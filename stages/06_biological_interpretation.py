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
# ## Stage 6 · Biological interpretation
#
# We use three interpretation methods:
#
# 1. **Pathway enrichment (GSEA pre-ranked):** genes are ranked genome-wide by the DESeq2 Wald statistic. GSEA asks whether each repair pathway sits at the TNBC-up or TNBC-down end of the ranking more than chance allows. The PAM50 proliferation genes are included as a positive control.
# 2. **Co-expression network (NetworkX):** genes are linked when their expression is strongly correlated across tumours, then grouped into modules to see which repair genes behave as one programme.
# 3. **Literature comparison.**

# %%
import gseapy as gp
ranking = de["stat"].dropna().sort_values(ascending=False)
ranking.index = ranking.index.astype(str)
gene_sets = {p: [str(i) for i in repair_present if p in repair.loc[i, "pathways"].split(";")] for p in PATHWAYS}
gene_sets["ALL_DNA_REPAIR"] = [str(i) for i in repair_present]
gene_sets["PROLIFERATION_PAM50 (control)"] = [str(i) for i in prolif_ids]
pre = gp.prerank(rnk=ranking, gene_sets=gene_sets, permutation_num=1000, min_size=5, max_size=1000,
                 seed=SEED, threads=os.cpu_count(), outdir=None, verbose=False)
gsea = pre.res2d.copy()
gsea["NES"] = gsea["NES"].astype(float)
gsea["FDR q-val"] = gsea["FDR q-val"].astype(float)
gsea["leading edge (first 12)"] = gsea["Lead_genes"].apply(
    lambda s: ", ".join(SYMBOL.get(int(g), g) for g in str(s).split(";")[:12]))
gsea = gsea.sort_values("NES", ascending=False)
display(gsea[["Term", "NES", "NOM p-val", "FDR q-val", "Tag %", "leading edge (first 12)"]])
gsea.to_csv(os.path.join(RESULTS_DIR, "gsea_prerank_repair_pathways.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#C0392B" if n > 0 else "#2E86C1" for n in gsea["NES"]]
ax.barh(gsea["Term"], gsea["NES"], color=colors)
for i, (n, q) in enumerate(zip(gsea["NES"], gsea["FDR q-val"])):
    ax.text(n + (0.05 if n > 0 else -0.05), i, f"FDR={q:.2g}", va="center", ha="left" if n > 0 else "right", fontsize=8)
ax.axvline(0, c="k", lw=0.8); ax.invert_yaxis()
ax.set(xlabel="normalised enrichment score (positive = higher in TNBC)", title="GSEA pre-ranked: DNA-repair pathways")
plt.tight_layout(); plt.show()

# %%
# Co-expression network of DNA-repair genes across all tumours
corr = logn.loc[repair_present].T.corr(method="spearman")
corr.index = corr.columns = repair.loc[repair_present, "symbol"]
THRESH = 0.6
G = nx.Graph()
fc = rep.set_index("symbol")["log2FoldChange"]
adj_fc = adj.set_index("symbol")["diff_adjusted"]
for s in corr.index:
    G.add_node(s, log2fc=fc[s], label=repair.loc[sym2id[s], "label"])
iu = np.triu_indices(len(corr), 1)
for i, j in zip(*iu):
    if abs(corr.values[i, j]) >= THRESH:
        G.add_edge(corr.index[i], corr.index[j], rho=corr.values[i, j], weight=abs(corr.values[i, j]))
G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
n_neg = sum(d["rho"] < 0 for _, _, d in G.edges(data=True))
print(f"network: {G.number_of_nodes()} genes with ≥1 edge, {G.number_of_edges()} edges (|ρ| ≥ {THRESH}; {n_neg} negative)")

# modules = groups of genes that rise and fall together, so communities are found on positive correlations only
G_pos = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d["rho"] > 0]).copy()
communities = sorted(nx.community.greedy_modularity_communities(G_pos, weight="weight"),
                     key=lambda c: (-len(c), sorted(c)))          # deterministic order
mod_rows = []
for k, comm in enumerate(communities, 1):
    members = sorted(comm, key=lambda s: (-G_pos.degree(s), s))
    ids = [sym2id[s] for s in members]
    module_score = z.loc[ids].mean(axis=0)
    mod_rows.append({"module": k, "n_genes": len(members), "hub genes (by degree)": ", ".join(members[:8]),
                     "median log2FC": round(fc[members].median(), 2),
                     "median adjusted diff": round(adj_fc[members].median(), 2),
                     "ρ(module score, proliferation)": round(stats.spearmanr(module_score, prolif_score)[0], 2),
                     "AUC TNBC": round(stats.mannwhitneyu(module_score[y == 1], module_score[y == 0]).statistic
                                       / (y.sum() * (1 - y).sum()), 2)})
    for s in members: G.nodes[s]["module"] = k
modules = pd.DataFrame(mod_rows).set_index("module")
display(modules)
deg = pd.Series(dict(G.degree())).sort_values(ascending=False)
btw = pd.Series(nx.betweenness_centrality(G)).sort_values(ascending=False)
print("top hubs by degree:", ", ".join(f"{g} ({d})" for g, d in deg.head(10).items()))
print("top by betweenness:", ", ".join(btw.head(6).index))

fig, ax = plt.subplots(figsize=(14, 12))
components = sorted(nx.connected_components(G), key=lambda c: (-len(c), sorted(c)))
pos = nx.kamada_kawai_layout(G.subgraph(components[0]), weight=None)      # main component
x0 = -1.0
for comp in components[1:]:                                                 # small components in a row below
    sub = G.subgraph(comp)
    sub_pos = nx.spring_layout(sub, seed=SEED, scale=0.12)
    for n, xy in sub_pos.items():
        pos[n] = np.asarray(xy) + np.array([x0, -1.35])
    x0 += 0.45
pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rho"] > 0]
neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rho"] < 0]
nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax, alpha=0.15, width=0.6)
nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax, alpha=0.5, width=0.8, style="dashed", edge_color="#2E86C1")
nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[40 + 12 * G.degree(n) for n in G.nodes],
                               node_color=[G.nodes[n]["log2fc"] for n in G.nodes], cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                               edgecolors="k", linewidths=0.3)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
plt.colorbar(nodes, ax=ax, label="log2 fold change TNBC vs non-TNBC", shrink=0.6)
ax.set_title("Co-expression network of DNA-repair genes (Spearman |ρ| ≥ 0.6)\nnode size = degree · node colour = log2FC · dashed blue = negative correlation")
ax.axis("off"); plt.tight_layout(); plt.show()

# %%
# BRCA1: TNBC is known to contain BRCA1-deficient tumours (germline mutation or promoter methylation).
# Expression alone cannot show deficiency, but a low-expression tail would be consistent with silencing.
b1 = logn.loc[sym2id["BRCA1"]]
cut = b1[y == 0].quantile(0.05)
low_t, low_n = (b1[y == 1] < cut).mean(), (b1[y == 0] < cut).mean()
print(f"BRCA1 below the 5th percentile of non-TNBC: {low_t:.1%} of TNBC vs {low_n:.1%} of non-TNBC")
print(f"BRCA1 median log2 expression: TNBC {b1[y == 1].median():.2f}, non-TNBC {b1[y == 0].median():.2f}")
b1_res = pd.Series(sm.OLS(b1.values, sm.add_constant(prolif_score.values)).fit().resid, index=b1.index)
print(f"BRCA1 relative to its proliferation-expected level: TNBC median residual {b1_res[y == 1].median():.2f}, "
      f"non-TNBC {b1_res[y == 0].median():.2f}; "
      f"Mann–Whitney p = {stats.mannwhitneyu(b1_res[y == 1], b1_res[y == 0]).pvalue:.2g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for g, col in GROUP_COLORS.items():
    sns.kdeplot(b1[cohort.group == g], ax=axes[0], fill=True, alpha=0.3, color=col, label=g, common_norm=False)
    axes[1].scatter(prolif_score[cohort.group == g], b1[cohort.group == g], s=8, alpha=0.6, c=col, label=g)
axes[0].axvline(cut, ls="--", c="grey"); axes[0].set(title="BRCA1 expression", xlabel="log2 normalised expression"); axes[0].legend()
axes[1].set(title="BRCA1 vs proliferation", xlabel="proliferation score", ylabel="BRCA1 log2 expression"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Literature comparison
#
# | Our finding | What the literature says | Agreement |
# |---|---|---|
# | TNBC separate from non-TNBC across the whole transcriptome and are far more proliferative (PCA; proliferation z 0.95 vs −0.09) | Most TNBC are basal-like, the most proliferative intrinsic subtype (Perou et al., 2000; TCGA Network, 2012) | Agrees |
# | HR, FA and checkpoint genes (*FANCA, RAD54L, CHEK1, RAD51AP1, EXO1, BLM*) are the most up-regulated repair genes in TNBC | The basal-like TNBC subtypes BL1/BL2 have high cell-cycle and DNA-damage-response gene expression (Lehmann et al., 2011) | Agrees |
# | Most of that up-regulation disappears after adjusting for proliferation; the up-regulated genes form one module tied to proliferation | Many replication-coupled repair genes are periodically expressed in the cell cycle and appear in tumour proliferation signatures (Whitfield et al., 2002) | Agrees, and explains our result |
# | *BRCA1* is lower than proliferation predicts in TNBC, with a low-expression tail in 16.6% of TNBC | Sporadic TNBC often share features of *BRCA1*-mutant cancers ("BRCAness"), and *BRCA1* can be silenced by promoter methylation in sporadic breast tumours (Turner et al., 2004; Esteller et al., 2000; Foulkes et al., 2010) | Consistent; needs methylation or mutation data to confirm |
# | HR genes such as *RAD51* and *BRCA2* are *higher* in TNBC, the subtype most associated with HR deficiency | HR deficiency is a functional state, detected through mutations, methylation or genomic "scars" (Knijnenburg et al., 2018; Davies et al., 2017), not mRNA levels | Paradox explained: expression ≠ function |
# | TNBC patients are younger and more often Black / African American | Basal-like breast cancer is more prevalent in younger African-American women (Carey et al., 2006) | Agrees |
#
# **Interpretation (Stage 6).** 
#
# * **Pathway enrichment:** the proliferation control is the most enriched set (NES 2.19, FDR < 0.001), which shows the method works. Among repair pathways, only **homologous recombination is significantly enriched towards TNBC** (NES 1.81, FDR 0.009). MMR (NES 1.50, FDR 0.065), all repair genes together (NES 1.56, FDR 0.057) and FA (NES 1.40, FDR 0.10) point the same way without reaching significance. BER, NER and NHEJ show no enrichment (FDR ≥ 0.48). The HR leading-edge genes (*CHEK1, RFC4, RAD54L, RAD51AP1, RAD51B, POLD1 …*) are the same genes highlighted in Stage 5.
# * **Co-expression network:** 55 repair genes have at least one strong co-expression partner. Most of them form one dense core made of two modules (25 + 13 genes) whose hubs are *RAD51AP1, EXO1, FANCB, BLM, CHEK1, NEIL3, RAD54L* and *RAD51*. Both module scores correlate strongly with proliferation (ρ = 0.85 and 0.88) and are higher in TNBC (AUC 0.80 and 0.79). After proliferation adjustment, though, their median TNBC difference is close to zero (0.14 and −0.02). This is a **single replication-coupled HR/FA programme** that switches on in dividing cells, and TNBC divide more. A separate small module (*MSH3, RAD50, POLK, FAN1, RAD17*) behaves differently. It is **lower in TNBC** (AUC 0.16, i.e. higher in non-TNBC), not tied to proliferation (ρ = −0.31), and keeps its difference after adjustment (median −0.53). This is the clearest proliferation-independent repair difference in the data.
# * **BRCA1:** median *BRCA1* expression is the same in both groups (9.97 vs 9.95), but TNBC have a long low tail: 16.6% of TNBC fall below the 5th percentile of non-TNBC (5.1% expected). Given how proliferative TNBC are, *BRCA1* is lower than expected (p = 7.8 × 10⁻⁹). This fits a subset of TNBC in which *BRCA1* is silenced (the BRCAness phenotype), while most TNBC have normal *BRCA1* driven up by proliferation. Expression alone cannot prove repair deficiency.
# * **Overall:** our results support the literature on direction (HR/FA up in TNBC, a BRCA1-low subset) but show that **most of the repair difference is proliferation**, a point often left out when repair-gene expression is compared between subtypes.
