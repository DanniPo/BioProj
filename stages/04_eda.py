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
# ## Stage 4 · Exploratory data analysis

# %%
# Cohort description: age and race by group (race matters for the representation-bias limitation)
desc = cohort.groupby("group")["AGE"].describe()[["count", "mean", "50%", "min", "max"]].round(1)
display(desc)
race = pd.crosstab(cohort["RACE"].fillna("Not reported"), cohort["group"], normalize="columns").mul(100).round(1)
display(race)
print("age difference (Mann–Whitney) p =",
      f"{stats.mannwhitneyu(cohort.AGE[y == 1].dropna(), cohort.AGE[y == 0].dropna()).pvalue:.2g}")
bw = pd.crosstab(cohort["RACE"] == "BLACK OR AFRICAN AMERICAN", cohort["group"])
print("Black/African-American patients over-represented among TNBC (Fisher exact) p =",
      f"{stats.fisher_exact(bw.values)[1]:.2g}")

# %%
# Scores used throughout: z-score each gene, then average over the gene set
z = logn.sub(logn.mean(axis=1), axis=0).div(logn.std(axis=1), axis=0)
prolif_ids = [i for i in PROLIF.values() if i in z.index]
prolif_score = z.loc[prolif_ids].mean(axis=0)
repair_score = z.loc[repair_present].mean(axis=0)

from sklearn.decomposition import PCA
def pca_plot(mat, ax, title):
    X = (mat.T - mat.T.mean()) / mat.T.std()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    pcs = pca.transform(X)
    for g, col in GROUP_COLORS.items():
        m = (cohort["group"] == g).values
        ax.scatter(pcs[m, 0], pcs[m, 1], s=9, alpha=0.7, c=col, label=f"{g} (n={m.sum()})")
    ev = pca.explained_variance_ratio_ * 100
    ax.set(title=title, xlabel=f"PC1 ({ev[0]:.1f}% variance)", ylabel=f"PC2 ({ev[1]:.1f}% variance)")
    ax.legend(markerscale=2)
    return pcs, ev

top_var = logn.var(axis=1).sort_values(ascending=False).index[:2000]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
pcs_all, ev_all = pca_plot(logn.loc[top_var], axes[0], "PCA · 2,000 most variable genes")
pcs_rep, ev_rep = pca_plot(logn.loc[repair_present], axes[1], f"PCA · {len(repair_present)} DNA-repair genes")
axes[2].scatter(prolif_score, repair_score, s=9, alpha=0.7, c=cohort["group"].map(GROUP_COLORS))
r_pr = stats.spearmanr(prolif_score, repair_score)[0]
print(f"Spearman correlation, repair score vs proliferation score: {r_pr:.2f}")
axes[2].set(title=f"Repair score vs proliferation score (Spearman ρ = {r_pr:.2f})",
            xlabel="proliferation score (PAM50 11 genes, mean z)", ylabel="DNA-repair score (mean z)")
plt.tight_layout(); plt.show()

auc_pc1 = max(stats.mannwhitneyu(pcs_rep[y.values == 1, 0], pcs_rep[y.values == 0, 0]).statistic,
              stats.mannwhitneyu(pcs_rep[y.values == 0, 0], pcs_rep[y.values == 1, 0]).statistic) / (y.sum() * (1 - y).sum())
print(f"Repair-gene PC1 alone separates TNBC from non-TNBC with AUC = {auc_pc1:.2f}")
print(f"Correlation of repair-gene PC1 with the proliferation score: ρ = {abs(stats.spearmanr(pcs_rep[:, 0], prolif_score)[0]):.2f}")
for g in GROUP_COLORS:
    print(f"median proliferation score in {g}: {prolif_score[cohort.group == g].median():.2f}; "
          f"median repair score: {repair_score[cohort.group == g].median():.2f}")

# %%
# Clustered heatmap of DNA-repair genes (z-scores, clipped at ±3) with the TNBC annotation
order = cohort.sort_values("group").index
hm = z.loc[repair_present, order].clip(-3, 3)
hm.index = repair.loc[repair_present, "symbol"]
lab_colors = {"HR": "#1F618D", "FA": "#7D3C98", "MMR": "#117A65", "NER": "#B9770E", "BER": "#F4D03F",
              "NHEJ": "#566573", "Shared": "#17202A"}
cg = sns.clustermap(hm, cmap="RdBu_r", center=0, figsize=(14, 16), xticklabels=False, yticklabels=True,
                    col_colors=cohort.loc[order, "group"].map(GROUP_COLORS).rename("group"),
                    row_colors=pd.Series(repair.loc[repair_present, "label"].map(lab_colors).values, index=hm.index, name="pathway"),
                    method="average", metric="correlation", dendrogram_ratio=(0.12, 0.06),
                    cbar_kws={"label": "z-score"})
cg.ax_heatmap.tick_params(axis="y", labelsize=6)
from matplotlib.patches import Patch
handles = [Patch(color=c, label=l) for l, c in lab_colors.items()] + [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
cg.fig.legend(handles=handles, loc="lower center", ncol=9, bbox_to_anchor=(0.55, -0.02), fontsize=9,
              title="row colour = repair pathway  ·  column colour = group")
cg.fig.suptitle("DNA-repair gene expression across 1,006 tumours (rows and columns clustered)", y=1.01)
plt.show()

# %%
# Which genes drive PC1 of the repair-gene PCA? PC1 is the proliferation axis (see above),
# so its loadings name the repair genes that move with proliferation in TNBC.
Xr = (logn.loc[repair_present].T - logn.loc[repair_present].T.mean()) / logn.loc[repair_present].T.std()
rep_pca = PCA(n_components=2, random_state=SEED).fit(Xr)
load1 = pd.Series(rep_pca.components_[0], index=repair_present)
score1 = rep_pca.transform(Xr)[:, 0]
if stats.spearmanr(score1, prolif_score)[0] < 0:          # PC1 sign is arbitrary; orient it so that
    load1 = -load1                                         # the positive end = high-proliferation / TNBC
load1 = load1.sort_values()
top = pd.concat([load1.head(12), load1.tail(12)])          # 12 lowest + 12 highest loadings
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(repair.loc[top.index, "symbol"], top.values,
        color=["#C0392B" if v > 0 else "#2471A3" for v in top.values])
ax.axvline(0, color="k", lw=0.6)
ax.set(title="Genes loading on PC1 of the repair-gene PCA\n(positive end = high-proliferation / TNBC direction)",
       xlabel="PC1 loading")
plt.tight_layout(); plt.show()
print("High-proliferation end:", ", ".join(repair.loc[load1.tail(8).index[::-1], "symbol"]))
print("Opposite end:         ", ", ".join(repair.loc[load1.head(8).index, "symbol"]))

# %% [markdown]
# **Interpretation (Stage 4).** 
#
# * **Cohort:** TNBC patients are younger (median age 54 vs 60, p = 1 × 10⁻⁴) and much more often Black / African American (33.8% vs 13.7%, p ≈ 10⁻⁸). This matches the epidemiology of TNBC (Carey et al., 2006). It also shows the representation problem from our ethics presentation: most patients are White (57–70%), about 5% are Asian, and none were recruited in Africa.
# * **Genome-wide PCA:** TNBC tumours form their own cluster at the negative end of PC1. TNBC differs across the whole transcriptome, not just in a few genes; this is the basal-like phenotype.
# * **Repair-gene PCA:** using only the 148 repair genes, TNBC still shift along PC1 (AUC 0.78). But repair-gene PC1 is strongly correlated with the proliferation score (ρ = 0.81), so the main axis of variation among repair genes is a **proliferation axis**.
# * **What drives repair-gene PC1:** the positive (high-proliferation / TNBC) end of PC1 is built from replication-coupled HR and Fanconi-anaemia genes together with S-phase checkpoint and mismatch-repair genes (*RAD51AP1, BLM, EXO1, FANCB, FANCI, FANCD2, MSH2, MSH6, CHEK1, TOPBP1, DNA2, NEIL3*) — genes that are switched on in dividing cells. Because these same genes define PC1 and PC1 tracks the proliferation score (ρ = 0.81), the main "repair" axis is really a **cell-cycle axis**: the repair genes that look high in TNBC are largely the ones expressed in proliferating tumours. The opposite end carries base- and nucleotide-excision-repair genes and back-up / translesion polymerases (*NEIL1, MPG, POLM, POLL, POLN, POLD4, PARP3, FAAP20, ERCC1, XPA, CCNH, ATRIP*), repair activities that are not replication-coupled. This is exactly why Stage 5 removes the proliferation effect before trusting any repair-specific difference.
# * **Repair vs proliferation:** the average repair score tracks proliferation closely (ρ = 0.76). TNBC sit at the high-proliferation end (median proliferation z-score 0.95 vs −0.09), yet their mean repair score is only slightly higher (0.14 vs −0.04). Averaging over all repair genes hides genes that move in opposite directions.
# * **Heatmap:** clustering places most TNBC tumours together in one block of columns. In that block, a large group of HR, FA and replication genes (*BLM, FANCA, FANCD2, FANCI, RAD51, RAD51AP1, EXO1, CHEK1, BRCA2, UBE2T …*) is high. A smaller set (*RAD50, MSH3, POLK, XPC …*) is low in part of the same block. Stage 5 tests both patterns formally.