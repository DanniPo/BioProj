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
# **Interpretation (Stage 7).** 
#
# * **Model selection (training set, 5-fold CV):** logistic regression with strong regularisation (C = 0.01) did best (CV ROC-AUC 0.970), slightly ahead of the random forest (0.951).
# * **Against the baselines:** the repair-gene model clearly beats the majority-class baseline (AUC 0.50) and the proliferation-only model (CV 0.84, test 0.83). So repair genes carry information beyond "this tumour divides fast", and adding the proliferation score to them adds nothing (0.970). **But 30 random sets of 148 non-repair genes do almost as well** (mean CV AUC 0.966, range 0.960–0.972). The repair genes beat 93% of the random sets, but only by about 0.004 AUC, which is not a meaningful difference.
# * **Test set (used once):** the repair-gene logistic regression reaches **ROC-AUC 0.93 (95% CI 0.87–0.97)** and PR-AUC 0.66 (chance level 0.16). Balanced accuracy is 0.87, with sensitivity 0.81 (26 of 32 TNBC found) and specificity 0.93 (158 of 170 non-TNBC correct). The random forest has a similar AUC (0.92), but at the default 0.5 threshold it finds only 44% of TNBC: its probabilities for the rare class are poorly calibrated. We would have to tune its threshold inside cross-validation, so logistic regression is our preferred model. The test AUC is a little lower than CV (0.93 vs 0.97); with only 32 TNBC in the test set this is within the uncertainty, and nothing was tuned on the test set.
# * **What the model uses:** the largest coefficients are *FANCE, SEM1, RAD51B, GTF2H1, ATM* (towards TNBC) and *ATRIP, RAD50, RAD17, POLL, POLD4, POLN* (towards non-TNBC). The coefficients correlate with the proliferation-adjusted differences from Stage 5 (ρ = 0.84). The model relies on the proliferation-independent part of the repair profile, the same genes our statistics highlighted.
# * **Honest conclusion:** DNA-repair expression classifies TNBC well, but that does **not** show repair genes are special. TNBC are so different transcriptome-wide that almost any 148 genes would do. The classifier confirms a distinct profile; it is not evidence of a repair-specific biomarker.
