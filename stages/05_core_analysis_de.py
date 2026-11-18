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
# ## Stage 5 · Core analysis: differential expression
#
# **Primary method: DESeq2** (Python port *PyDESeq2*) on raw counts. It uses a negative-binomial model with median-of-ratios normalisation and a Wald test, plus **Benjamini–Hochberg FDR** correction. We test all 16k expressed genes so that dispersion estimates and the FDR are genome-wide. We then look at the 148 expressed repair genes and also re-adjust p-values within that pre-specified family.
#
# **Effect-size threshold.** With ~1,000 tumours even tiny differences become "significant", so we call a gene *strongly* differentially expressed only if FDR < 0.05 **and** |log2 fold change| ≥ 1 (≥ two-fold).
#
# **Robustness checks**
#
# 1. A **Wilcoxon rank-sum test** on normalised expression, which is distribution-free and recommended for large human cohorts (Li et al., *Genome Biology* 2022).
# 2. A **gene-set (competitive) test**: are repair genes shifted more than random sets of genes?
# 3. A **proliferation-adjusted model**: is the difference still there after accounting for how fast the tumour is dividing?

# %%
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

t0 = time.time()
meta = pd.DataFrame({"group": np.where(y == 1, "TNBC", "nonTNBC")}, index=counts_f.columns)
inference = DefaultInference(n_cpus=os.cpu_count())
dds = DeseqDataSet(counts=counts_f.T.astype(int), metadata=meta, design="~group",
                   refit_cooks=True, inference=inference, quiet=True)
dds.deseq2()
ds = DeseqStats(dds, contrast=["group", "TNBC", "nonTNBC"], inference=inference, quiet=True)
ds.summary()
de = ds.results_df.copy()
de.index = de.index.astype(int)
print(f"DESeq2 finished in {time.time()-t0:.0f}s")
print("size factors agree with Stage 3:", round(np.corrcoef(dds.obs["size_factors"] if "size_factors" in dds.obs
      else dds.obsm["size_factors"], size_factors)[0, 1], 4))

de["symbol"] = [SYMBOL.get(i, "") for i in de.index]
sig = (de.padj < 0.05)
print(f"\nGenome-wide: {int(sig.sum()):,} of {len(de):,} genes have FDR < 0.05 "
      f"({sig.mean():.0%}); {int((sig & (de.log2FoldChange.abs() >= 1)).sum()):,} also have |log2FC| ≥ 1.")
display(de.loc[[MARKERS[g] for g in ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "FOXC1", "KRT5"]],
               ["symbol", "baseMean", "log2FoldChange", "padj"]].round(3))

# %%
# Wilcoxon rank-sum test for every gene on log2 normalised expression (vectorised)
mw = stats.mannwhitneyu(logn.loc[:, y.values == 1].values, logn.loc[:, y.values == 0].values, axis=1)
de["wilcoxon_auc"] = mw.statistic / (y.sum() * (1 - y).sum())       # >0.5 = higher in TNBC
de["wilcoxon_padj"] = multipletests(mw.pvalue, method="fdr_bh")[1]

# DNA-repair results table
rep = de.loc[repair_present].copy()
rep["pathways"] = repair.loc[repair_present, "pathways"]
rep["label"] = repair.loc[repair_present, "label"]
rep["padj_repair_family"] = multipletests(rep.pvalue, method="fdr_bh")[1]
def call(r):
    if r.padj_repair_family >= 0.05: return "not significant"
    if abs(r.log2FoldChange) >= 1: return "UP ≥2x in TNBC" if r.log2FoldChange > 0 else "DOWN ≥2x in TNBC"
    return "up (<2x)" if r.log2FoldChange > 0 else "down (<2x)"
rep["call"] = rep.apply(call, axis=1)
rep = rep.sort_values("stat", ascending=False)
cols = ["symbol", "pathways", "baseMean", "log2FoldChange", "lfcSE", "padj", "padj_repair_family",
        "wilcoxon_auc", "wilcoxon_padj", "call"]
print(rep["call"].value_counts().to_string())
print("\nTop 15 up in TNBC:")
display(rep[cols].head(15).round(4))
print("Top 10 down in TNBC:")
display(rep[cols].tail(10).iloc[::-1].round(4))
rep[cols].to_csv(os.path.join(RESULTS_DIR, "dna_repair_DE_results.csv"))
de.to_csv(os.path.join(RESULTS_DIR, "genome_wide_DESeq2_results.csv"))

agree_sig = ((rep.padj_repair_family < 0.05) == (rep.wilcoxon_padj < 0.05)).mean()
agree_dir = (np.sign(rep.log2FoldChange) == np.sign(rep.wilcoxon_auc - 0.5)).mean()
print(f"DESeq2 vs Wilcoxon: same direction for {agree_dir:.0%} of repair genes; "
      f"same significance call for {agree_sig:.0%}.")

# %%
# Volcano plot (genome-wide), repair genes highlighted
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
nl = -np.log10(de.padj.clip(lower=1e-300))
ax.scatter(de.log2FoldChange, nl, s=3, c="#D5D8DC", label="other genes")
is_rep = de.index.isin(repair_present)
ax.scatter(de.log2FoldChange[is_rep], nl[is_rep], s=14, c=de.log2FoldChange[is_rep], cmap="RdBu_r",
           vmin=-2, vmax=2, edgecolor="k", linewidth=0.3, label="DNA-repair genes")
label_ids = list(rep.index[:6]) + list(rep.index[-3:]) + [MARKERS["ESR1"], MARKERS["FOXC1"]]
offsets = [(40, 30), (55, 5), (45, -20), (30, -40), (60, -35), (70, 20), (-60, 25), (-70, 0), (-55, -25), (15, -10), (20, 15)]
for eid, off in zip(label_ids, offsets):
    ax.annotate(de.symbol[eid], (de.log2FoldChange[eid], nl[eid]), fontsize=8, xytext=off, textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=0.8); ax.axvline(1, ls=":", c="grey"); ax.axvline(-1, ls=":", c="grey")
ax.set(xlabel="log2 fold change (TNBC vs non-TNBC)", ylabel="−log10 FDR", title="Volcano plot · genome-wide DESeq2")
ax.legend(loc="upper left")

# log2FC of repair genes by pathway label
order_lab = rep.groupby("label").log2FoldChange.median().sort_values(ascending=False).index
sns.boxplot(data=rep, y="label", x="log2FoldChange", order=order_lab, ax=axes[1], color="white", fliersize=0)
sns.stripplot(data=rep, y="label", x="log2FoldChange", order=order_lab, ax=axes[1], hue="call", size=5,
              palette={"UP ≥2x in TNBC": "#C0392B", "up (<2x)": "#F1948A", "not significant": "#AAB7B8",
                       "down (<2x)": "#85C1E9", "DOWN ≥2x in TNBC": "#1F618D"})
axes[1].axvline(0, c="k", lw=0.8)
axes[1].set(title="Repair-gene fold changes by pathway", xlabel="log2 fold change (TNBC vs non-TNBC)", ylabel="")
axes[1].legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()

# %%
# Gene-set (competitive) test: are repair genes more shifted than random gene sets of the same size?
wald = de["stat"]
others = wald.drop(repair_present)
obs_mean, obs_abs = wald[repair_present].mean(), wald[repair_present].abs().mean()
rng = np.random.default_rng(SEED)
n_perm = 10_000
perm = np.array([rng.choice(others.values, len(repair_present), replace=False) for _ in range(n_perm)])
p_mean = (np.sum(perm.mean(axis=1) >= obs_mean) + 1) / (n_perm + 1)
p_abs = (np.sum(np.abs(perm).mean(axis=1) >= obs_abs) + 1) / (n_perm + 1)
print(f"mean Wald statistic, repair genes: {obs_mean:.2f} vs random sets: {perm.mean():.2f} "
      f"(95% range {np.percentile(perm.mean(1), 2.5):.2f} to {np.percentile(perm.mean(1), 97.5):.2f}); "
      f"one-sided permutation p = {p_mean:.4f}")
print(f"mean |Wald|, repair genes: {obs_abs:.2f} vs random: {np.abs(perm).mean():.2f}; permutation p = {p_abs:.4f}")
print(f"fraction up in TNBC: repair {(wald[repair_present] > 0).mean():.0%} vs all other genes {(others > 0).mean():.0%}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(perm.mean(axis=1), bins=60, color="#AAB7B8", label="10,000 random gene sets")
ax.axvline(obs_mean, color="#C0392B", lw=2, label="DNA-repair genes")
ax.set(xlabel="mean DESeq2 Wald statistic (TNBC vs non-TNBC)", ylabel="count", title="Competitive gene-set test")
ax.legend(); plt.tight_layout(); plt.show()

# %%
# Proliferation-adjusted model for each repair gene:  expression ~ TNBC  vs  expression ~ TNBC + proliferation
rows = []
for eid in repair_present:
    yy = logn.loc[eid].values
    m0 = sm.OLS(yy, sm.add_constant(y.values.astype(float))).fit(cov_type="HC3")
    X1 = sm.add_constant(np.column_stack([y.values.astype(float), prolif_score.values]))
    m1 = sm.OLS(yy, X1).fit(cov_type="HC3")
    rows.append({"entrez": eid, "symbol": repair.loc[eid, "symbol"], "label": repair.loc[eid, "label"],
                 "diff_unadjusted": m0.params[1], "p_unadjusted": m0.pvalues[1],
                 "diff_adjusted": m1.params[1], "p_adjusted": m1.pvalues[1],
                 "prolif_coef": m1.params[2], "r_with_prolif": stats.spearmanr(yy, prolif_score)[0]})
adj = pd.DataFrame(rows).set_index("entrez")
adj["fdr_unadjusted"] = multipletests(adj.p_unadjusted, method="fdr_bh")[1]
adj["fdr_adjusted"] = multipletests(adj.p_adjusted, method="fdr_bh")[1]
adj["retained_pct"] = 100 * adj.diff_adjusted / adj.diff_unadjusted
up0 = (adj.fdr_unadjusted < 0.05) & (adj.diff_unadjusted > 0)
dn0 = (adj.fdr_unadjusted < 0.05) & (adj.diff_unadjusted < 0)
up1 = (adj.fdr_adjusted < 0.05) & (adj.diff_adjusted > 0)
dn1 = (adj.fdr_adjusted < 0.05) & (adj.diff_adjusted < 0)
print(f"Before adjustment: {int(up0.sum())} repair genes up, {int(dn0.sum())} down in TNBC (FDR < 0.05)")
print(f"After adjusting for proliferation: {int(up1.sum())} up, {int(dn1.sum())} down")
print(f"Median correlation of repair genes with the proliferation score: ρ = {adj.r_with_prolif.median():.2f}")
print(f"For genes up before adjustment, median share of the TNBC difference left after adjustment: "
      f"{adj.loc[up0, 'retained_pct'].median():.0f}%")
print("\nRepair genes still UP in TNBC after proliferation adjustment (largest effects):")
display(adj[up1].sort_values("diff_adjusted", ascending=False).head(12)[
    ["symbol", "label", "diff_unadjusted", "diff_adjusted", "fdr_adjusted", "r_with_prolif"]].round(3))
print("Repair genes DOWN in TNBC after adjustment (largest effects):")
display(adj[dn1].sort_values("diff_adjusted").head(12)[
    ["symbol", "label", "diff_unadjusted", "diff_adjusted", "fdr_adjusted", "r_with_prolif"]].round(3))
adj.to_csv(os.path.join(RESULTS_DIR, "dna_repair_proliferation_adjusted.csv"))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(adj.diff_unadjusted, adj.diff_adjusted, c=adj.r_with_prolif, cmap="viridis", s=22, edgecolor="k", lw=0.3)
lim = [min(adj.diff_unadjusted.min(), adj.diff_adjusted.min()) - 0.1, max(adj.diff_unadjusted.max(), adj.diff_adjusted.max()) + 0.1]
ax.plot(lim, lim, ls="--", c="grey", lw=0.8); ax.axhline(0, c="k", lw=0.6); ax.axvline(0, c="k", lw=0.6)
for eid in adj.diff_adjusted.abs().sort_values().index[-10:]:
    ax.annotate(adj.symbol[eid], (adj.diff_unadjusted[eid], adj.diff_adjusted[eid]), fontsize=7, xytext=(3, 2), textcoords="offset points")
sc = ax.collections[0]; plt.colorbar(sc, ax=ax, label="Spearman ρ with proliferation score")
ax.set(xlabel="TNBC difference, unadjusted (log2)", ylabel="TNBC difference, adjusted for proliferation (log2)",
       title="How much of each repair gene's TNBC difference survives\nadjustment for proliferation?")
plt.tight_layout(); plt.show()

# %%
# Key genes: the BRCAness-related genes plus the strongest up/down genes after adjustment
show = ["BRCA1", "BRCA2", "RAD51", "PARP1", "FANCD2", "EXO1"]
show += [s for s in list(adj[up1].sort_values("diff_adjusted", ascending=False).symbol[:2]) +
         list(adj[dn1].sort_values("diff_adjusted").symbol[:2]) if s not in show]
fig, axes = plt.subplots(2, 5, figsize=(18, 7)); axes = axes.ravel()
sym2id = {v: k for k, v in repair["symbol"].items()}
for ax, s in zip(axes, show[:10]):
    df = pd.DataFrame({"expr": logn.loc[sym2id[s]], "group": cohort["group"]})
    sns.violinplot(data=df, x="group", y="expr", order=["non-TNBC", "TNBC"], palette=GROUP_COLORS, ax=ax, cut=0, inner="quartile")
    r = rep.loc[sym2id[s]]
    ax.set(title=f"{s}  log2FC={r.log2FoldChange:.2f}, FDR={r.padj_repair_family:.1g}", xlabel="", ylabel="log2 norm. expr")
    ax.title.set_fontsize(9)
for ax in axes[len(show):]: ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Interpretation (Stage 5).** 
#
# * **Sanity:** the receptor genes come out as expected (*ESR1* log2FC −5.0, *PGR* −4.1, *ERBB2* −2.2), as do the basal markers (*FOXC1* +3.0, *KRT5* +2.5). So the contrast direction and the labels are correct.
# * **Genome-wide:** 13,377 of 16,697 genes (80%) have FDR < 0.05. With 1,006 tumours almost everything is statistically "different", so effect sizes and gene-set comparisons matter more than p-values.
# * **DNA-repair genes:** 116 of 148 differ at FDR < 0.05: **79 are higher and 37 lower in TNBC**. Only 11 change at least two-fold. Ten are up (*FANCA, RAD54L, CHEK1, RAD51AP1, EXO1, FANCE, RAD51B, FANCB, BLM, RFC4*) and one is down (*POLN*). The changes are real but moderate (from 0.45- to 2.6-fold). HR, FA and the shared replication factors lean up, while NER and parts of MMR (*MSH3, MLH1, MLH3*) lean down.
# * **Robustness:** the Wilcoxon test agrees with DESeq2 on direction for 94% of repair genes and on significance for 91%, so the results do not hinge on DESeq2's model assumptions. *RAD51B* is an exception worth noting. Its DESeq2 fold change is large (log2FC 1.14) but the Wilcoxon effect is small (AUC 0.64), because its mean is pulled up by a subset of TNBC with very high expression (long upper tail in the violin plot).
# * **Gene-set test:** as a set, repair genes are shifted towards TNBC much more than random genes (mean Wald statistic 2.64 vs 0.49, p = 0.0003; 66% up vs 54% of other genes). They are **not more strongly changed in absolute size** than random genes (mean |Wald| 6.63 vs 6.46, p = 0.33). So the repair set is distinctive in its *direction* (up in TNBC), not its *magnitude*.
# * **Proliferation (key result):** adjusting for proliferation halves the number of up-regulated repair genes (74 → 38). For genes that were up, a median of only 31% of the TNBC difference remains. *BRCA2* (−0.02, FDR 0.87) and *RAD51* (−0.22) lose their up-regulation entirely. So most of the "TNBC repair up-regulation" is the expected consequence of TNBC dividing faster, because replication-coupled repair genes are cell-cycle regulated. A proliferation-independent signal remains in both directions:
#   * Still **higher** in TNBC: *RAD51B, FANCE, FANCA, USP1, DCLRE1C, CHEK1, MSH2*.
#   * **Lower** than proliferation predicts (67 genes): led by ***BRCA1*** (−0.84 log2), then *POLN, POLD4, RAD50, EME2, ATRIP, MSH3*.
#
#   The adjusted estimate is the part of the TNBC difference that proliferation cannot explain. It is not the "true" effect: if TNBC biology drives both proliferation and repair expression, adjustment removes some real signal too.
