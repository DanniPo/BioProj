# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Stage 7 · Machine learning
#
# **Task:** classify a tumour as TNBC or non-TNBC from the expression of the DNA-repair genes only. The receptor genes that *define* the label (*ESR1, PGR, ERBB2*) are not in the feature set, so the model cannot simply read off the answer.
#
# **Design:** a stratified 80/20 train/test split. All model selection uses 5-fold cross-validation on the training set only, and the test set is used once at the end. Scaling happens inside the pipeline, so no information leaks from validation folds. Classes are imbalanced (~16% TNBC), so we use `class_weight="balanced"` and report **ROC-AUC, PR-AUC, balanced accuracy, sensitivity and specificity**, not plain accuracy.
#
# **Baselines.** A model is only interesting if it beats these:
#
# 1. **Majority class:** always predict non-TNBC.
# 2. **Proliferation only:** logistic regression on the 1-number PAM50 proliferation score.
# 3. **Random genes:** the same model on 30 random sets of the same number of expressed non-repair genes. This tests whether repair genes are *special* or whether any genes would do.

# %%
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score,
                             confusion_matrix, roc_curve, precision_recall_curve, f1_score)

X_rep = logn.loc[repair_present].T.copy()
X_rep.columns = repair.loc[repair_present, "symbol"].values
X_pro = prolif_score.to_frame("proliferation")
idx_train, idx_test = train_test_split(X_rep.index, test_size=0.2, stratify=y, random_state=SEED)
y_train, y_test = y[idx_train], y[idx_test]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"train: {len(idx_train)} tumours ({y_train.sum()} TNBC) | test: {len(idx_test)} tumours ({y_test.sum()} TNBC)")

def logreg(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))

# Tune the two candidate models with 5-fold CV on the training set
lr_search = GridSearchCV(logreg(), {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}, cv=cv, scoring="roc_auc").fit(X_rep.loc[idx_train], y_train)
rf_search = GridSearchCV(RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
                         {"min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.3]}, cv=cv, scoring="roc_auc").fit(X_rep.loc[idx_train], y_train)
print("logistic regression best:", lr_search.best_params_, f"CV ROC-AUC {lr_search.best_score_:.3f}")
print("random forest best      :", rf_search.best_params_, f"CV ROC-AUC {rf_search.best_score_:.3f}")

# %%
# Baselines evaluated with the same CV folds on the training set
cv_rows = []
def cv_auc(model, X):
    return cross_val_score(model, X.loc[idx_train], y_train, cv=cv, scoring="roc_auc")

cv_rows.append({"model": "Majority class (dummy)", "CV ROC-AUC": cv_auc(DummyClassifier(strategy="most_frequent"), X_pro).mean()})
cv_rows.append({"model": "Proliferation score only (LR)", "CV ROC-AUC": cv_auc(logreg(), X_pro).mean()})
best_C = lr_search.best_params_["logisticregression__C"]
pool = logn.index.difference(repair_present).difference(list(MARKERS.values())[:3])
rand_aucs = []
rng = np.random.default_rng(SEED)
for _ in range(30):
    genes = rng.choice(pool, len(repair_present), replace=False)
    rand_aucs.append(cv_auc(logreg(best_C), logn.loc[genes].T).mean())
cv_rows.append({"model": f"Random {len(repair_present)} non-repair genes (LR, mean of 30 sets)", "CV ROC-AUC": np.mean(rand_aucs)})
cv_rows.append({"model": "DNA-repair genes (LR)", "CV ROC-AUC": lr_search.best_score_})
cv_rows.append({"model": "DNA-repair genes (random forest)", "CV ROC-AUC": rf_search.best_score_})
X_both = X_rep.join(X_pro)
cv_rows.append({"model": "DNA-repair genes + proliferation (LR)", "CV ROC-AUC": cv_auc(logreg(best_C), X_both).mean()})
cv_table = pd.DataFrame(cv_rows).set_index("model").round(3)
display(cv_table)
print(f"random gene sets: CV ROC-AUC range {min(rand_aucs):.3f}–{max(rand_aucs):.3f}; "
      f"repair genes beat {np.mean(np.array(rand_aucs) < lr_search.best_score_):.0%} of the 30 random sets")

# %% [markdown]
# ### Repeated cross-validation of the baselines
# One 5-fold split gives a single number per model. Repeating 5-fold CV five times with different shuffles (25 fits per model, training set only) shows how much each score varies. We need that to judge whether the small gap between repair genes and random genes is real. Every model is scored on the **same 25 splits**, so the scores can be compared fold by fold.

# %%
from sklearn.model_selection import RepeatedStratifiedKFold

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

def rcv_auc(model, X):
    return cross_val_score(model, X.loc[idx_train], y_train, cv=rcv, scoring="roc_auc", n_jobs=-1)

rcv_scores = {
    "Majority class (dummy)": rcv_auc(DummyClassifier(strategy="most_frequent"), X_pro),
    "Proliferation score only (LR)": rcv_auc(logreg(), X_pro),
    "DNA-repair genes (LR)": rcv_auc(logreg(best_C), X_rep),
    "DNA-repair genes (random forest)": rcv_auc(rf_search.best_estimator_, X_rep),
    "DNA-repair genes + proliferation (LR)": rcv_auc(logreg(best_C), X_both),
}

# 30 random sets of non-repair genes, each scored on the same 25 splits.
# A separate random generator keeps the shared `rng` (used later for the bootstrap CI) unchanged.
my_rng = np.random.default_rng(SEED)
random_fold_scores = np.array([rcv_auc(logreg(best_C), logn.loc[my_rng.choice(pool, len(repair_present), replace=False)].T)
                               for _ in range(30)])                     # shape: 30 sets x 25 folds
rcv_scores[f"Random {len(repair_present)} genes (LR, 30 sets)"] = random_fold_scores.ravel()

rcv_table = pd.DataFrame({name: {"mean ROC-AUC": s.mean(), "SD across folds": s.std(ddof=1),
                                 "min": s.min(), "max": s.max()} for name, s in rcv_scores.items()}).T.round(3)
display(rcv_table)

repair_lr = rcv_scores["DNA-repair genes (LR)"]
set_means = random_fold_scores.mean(axis=1)
diff = repair_lr - random_fold_scores                                   # paired: same folds
print(f"Repair-gene LR mean AUC {repair_lr.mean():.3f} is higher than {np.mean(set_means < repair_lr.mean()):.0%} "
      f"of the 30 random-set means (random sets: {set_means.mean():.3f} ± {set_means.std(ddof=1):.3f}).")
print(f"Paired difference, repair minus random (same folds): {diff.mean():+.4f} AUC "
      f"(SD {diff.std(ddof=1):.4f}); repair genes win in {np.mean(diff > 0):.0%} of set-fold pairs.")

fig, ax = plt.subplots(figsize=(10, 4.5))
order_m = list(rcv_scores)
sns.boxplot(data=[rcv_scores[m] for m in order_m], orient="h", ax=ax, color="#D6EAF8", fliersize=2)
ax.set_yticks(range(len(order_m)))
ax.set_yticklabels(order_m)
ax.set(xlabel="ROC-AUC (5 × 5 repeated CV, training set)", title="Model comparison with fold-to-fold variation")
ax.axvline(0.5, ls="--", c="grey", lw=0.8)
plt.tight_layout(); plt.show()

# %%
# Final evaluation on the untouched test set
final_models = {
    "Majority class (dummy)": (DummyClassifier(strategy="most_frequent"), X_pro),
    "Proliferation score only (LR)": (logreg(), X_pro),
    "DNA-repair genes (LR)": (lr_search.best_estimator_, X_rep),
    "DNA-repair genes (random forest)": (rf_search.best_estimator_, X_rep),
}
test_rows, curves = [], {}
for name, (model, X) in final_models.items():
    model.fit(X.loc[idx_train], y_train)
    prob = model.predict_proba(X.loc[idx_test])[:, 1]
    pred = model.predict(X.loc[idx_test])
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    test_rows.append({"model": name, "ROC-AUC": roc_auc_score(y_test, prob), "PR-AUC": average_precision_score(y_test, prob),
                      "balanced accuracy": balanced_accuracy_score(y_test, pred),
                      "sensitivity (TNBC recall)": recall_score(y_test, pred, zero_division=0),
                      "specificity": tn_ / (tn_ + fp_), "F1 (TNBC)": f1_score(y_test, pred, zero_division=0)})
    curves[name] = prob
test_table = pd.DataFrame(test_rows).set_index("model").round(3)
display(test_table)
test_table.to_csv(os.path.join(RESULTS_DIR, "ml_test_metrics.csv"))

# Bootstrap 95% CI for the test ROC-AUC of the repair-gene LR
boot = []
p_lr = curves["DNA-repair genes (LR)"]
yt = y_test.values
for _ in range(2000):
    b = rng.integers(0, len(yt), len(yt))
    if yt[b].min() != yt[b].max():
        boot.append(roc_auc_score(yt[b], p_lr[b]))
print(f"Repair-gene LR test ROC-AUC 95% bootstrap CI: {np.percentile(boot, 2.5):.3f}–{np.percentile(boot, 97.5):.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, prob in curves.items():
    if name.startswith("Majority"): continue
    fpr, tpr, _ = roc_curve(y_test, prob); axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, prob):.2f})")
    pr, rc, _ = precision_recall_curve(y_test, prob); axes[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y_test, prob):.2f})")
axes[0].plot([0, 1], [0, 1], ls="--", c="grey"); axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)")
axes[1].axhline(y_test.mean(), ls="--", c="grey", label=f"chance ({y_test.mean():.2f})")
axes[1].set(xlabel="recall (TNBC)", ylabel="precision", title="Precision–recall curves (test set)")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
cm = confusion_matrix(y_test, (p_lr >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
            xticklabels=["pred non-TNBC", "pred TNBC"], yticklabels=["true non-TNBC", "true TNBC"])
axes[2].set_title("Confusion matrix · DNA-repair LR (threshold 0.5)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Tuning the random forest's decision threshold
# At the default 0.5 threshold the random forest finds few TNBC tumours: with only ~16% TNBC its probabilities rarely reach 0.5. We choose a better threshold using **out-of-fold predictions on the training set only**, then apply that single threshold to the test set once. This keeps the test set untouched by the tuning.

# %%
from sklearn.model_selection import cross_val_predict

rf_oof = cross_val_predict(rf_search.best_estimator_, X_rep.loc[idx_train], y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
ba_oof = np.array([balanced_accuracy_score(y_train, (rf_oof >= t).astype(int)) for t in thresholds])
t_best = float(thresholds[ba_oof.argmax()])
print(f"threshold chosen on training out-of-fold predictions: {t_best:.2f} "
      f"(training balanced accuracy {ba_oof.max():.3f} vs {ba_oof[thresholds == 0.5][0]:.3f} at 0.50)")

p_rf = curves["DNA-repair genes (random forest)"]
thr_rows = []
for label, t in [("0.50 (default)", 0.5), (f"{t_best:.2f} (tuned on training data)", t_best)]:
    pred = (p_rf >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    thr_rows.append({"RF threshold": label, "balanced accuracy": balanced_accuracy_score(y_test, pred),
                     "sensitivity (TNBC recall)": tp_ / (tp_ + fn_), "specificity": tn_ / (tn_ + fp_),
                     "F1 (TNBC)": f1_score(y_test, pred, zero_division=0), "TNBC found": f"{tp_} of {tp_ + fn_}"})
thr_table = pd.DataFrame(thr_rows).set_index("RF threshold").round(3)
display(thr_table)
print(f"test ROC-AUC is unchanged by the threshold: {roc_auc_score(y_test, p_rf):.3f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(thresholds, ba_oof, c="#1F618D")
ax.axvline(t_best, ls="--", c="#C0392B", label=f"chosen threshold {t_best:.2f}")
ax.axvline(0.5, ls=":", c="grey", label="default 0.50")
ax.set(xlabel="decision threshold", ylabel="balanced accuracy (out-of-fold)", title="Random forest: choosing the threshold on training data")
ax.legend(); plt.tight_layout(); plt.show()

# %%
# Which repair genes does the logistic regression rely on? (standardised coefficients)
coef = pd.Series(lr_search.best_estimator_.named_steps["logisticregression"].coef_[0], index=X_rep.columns)
top = coef.reindex(coef.abs().sort_values(ascending=False).index[:20])
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top.index[::-1], top.values[::-1], color=["#C0392B" if v > 0 else "#2E86C1" for v in top.values[::-1]])
ax.axvline(0, c="k", lw=0.8)
ax.set(xlabel="standardised coefficient (positive = pushes towards TNBC)", title="Top 20 features · DNA-repair logistic regression")
plt.tight_layout(); plt.show()
print("Spearman correlation between LR coefficient and adjusted TNBC difference:",
      round(stats.spearmanr(coef, adj.set_index("symbol").loc[coef.index, "diff_adjusted"])[0], 2))

# %% [markdown]
# ### Permutation importance (test set)
# Coefficients show what the model learned. Permutation importance asks what it actually *uses* on unseen data: shuffle one gene in the test set and measure how much ROC-AUC drops (30 shuffles per gene).

# %%
from sklearn.inspection import permutation_importance

pi = permutation_importance(lr_search.best_estimator_, X_rep.loc[idx_test], y_test, scoring="roc_auc",
                            n_repeats=30, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"drop in ROC-AUC": pi.importances_mean, "SD": pi.importances_std}, index=X_rep.columns)
imp = imp.sort_values("drop in ROC-AUC", ascending=False)
top_pi = imp.head(15)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(top_pi.index[::-1], top_pi["drop in ROC-AUC"][::-1], xerr=top_pi["SD"][::-1], color="#5D6D7E")
ax.set(xlabel="mean drop in test ROC-AUC when the gene is shuffled", title="Top 15 genes by permutation importance · repair-gene LR")
plt.tight_layout(); plt.show()

overlap = sorted(set(top_pi.index) & set(top.index))
print(f"genes with any measurable importance (mean drop > 0): {int((imp['drop in ROC-AUC'] > 0).sum())} of {len(imp)}")
print(f"largest single-gene drop: {imp['drop in ROC-AUC'].iloc[0]:.4f} AUC ({imp.index[0]})")
print(f"{len(overlap)} of the top 15 permutation genes are also in the top 20 coefficients: {', '.join(overlap)}")

# %% [markdown]
# **Interpretation (Stage 7).** 
#
# * **Model selection (training set, 5-fold CV):** logistic regression with strong regularisation (C = 0.01) did best (CV ROC-AUC 0.970), slightly ahead of the random forest (0.950).
# * **Against the baselines:** the repair-gene model clearly beats the majority-class baseline (AUC 0.50) and the proliferation-only model (CV 0.84, test 0.83). So repair genes carry information beyond "this tumour divides fast", and adding the proliferation score to them adds nothing. **But 30 random sets of 148 non-repair genes do just as well.**
# * **Repeated cross-validation (5 × 5, same splits for every model):** this confirms the point above and makes it sharper. The repair-gene model scores 0.969 ± 0.013 and the random gene sets 0.967 ± 0.013. Scored on identical folds, the repair genes are ahead by only +0.002 AUC (SD 0.008) and win in just 61% of the comparisons. They beat 77% of the random-set means, down from 93% with the single 5-fold split, so the single split flattered the repair genes. The proliferation-only model is not only weaker (0.840) but also the least stable (SD 0.045, range 0.73–0.91).
# * **Test set (used once):** the repair-gene logistic regression reaches **ROC-AUC 0.93 (95% CI 0.87–0.97)** and PR-AUC 0.66 (chance level 0.16). Balanced accuracy is 0.87, with sensitivity 0.81 (26 of 32 TNBC found) and specificity 0.93 (158 of 170 non-TNBC correct). The test AUC is a little lower than CV (0.93 vs 0.97); with only 32 TNBC in the test set this is within the uncertainty, and nothing was tuned on the test set.
# * **Random forest threshold:** at the default 0.5 threshold the random forest finds only 14 of 32 TNBC, because its probabilities for the rare class are low. A threshold of 0.22, chosen on training out-of-fold predictions only, raises test sensitivity from 44% to 78% (25 of 32) and balanced accuracy from 0.70 to 0.85. Specificity drops slightly (0.97 → 0.92), and ROC-AUC is unchanged (0.919) because it does not depend on the threshold. Even tuned, the random forest stays below the logistic regression (balanced accuracy 0.85 vs 0.87), so logistic regression remains our preferred model.
# * **What the model uses:** the largest coefficients are *FANCE, SEM1, RAD51B, GTF2H1, ATM* (towards TNBC) and *ATRIP, RAD50, RAD17, POLL, POLD4, POLN* (towards non-TNBC). The coefficients correlate with the proliferation-adjusted differences from Stage 5 (ρ = 0.84). The model relies on the proliferation-independent part of the repair profile, the same genes our statistics highlighted.
# * **Permutation importance:** no single gene matters much. Shuffling the most important gene, *ATM*, lowers test AUC by only 0.006, and only 72 of the 148 genes have any measurable effect. 9 of the top 15 genes (*ATM, DCLRE1C, FANCA, PARP4, POLD4, POLL, POLN, RAD51B, XPA*) are also among the top 20 coefficients, so the two methods broadly agree. The information is spread across many correlated genes, which is also why random gene sets can do the same job.
# * **Honest conclusion:** DNA-repair expression classifies TNBC well, but that does **not** show repair genes are special. TNBC are so different transcriptome-wide that almost any 148 genes would do. The classifier confirms a distinct profile; it is not evidence of a repair-specific biomarker.