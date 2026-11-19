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
# 2. **Co-expression network (NetworkX):** genes are linked when their expression is strongly correlated across tumours, then grouped into modules to see which repair genes behave as one programme. We also build a protein–protein interaction (PPI) network from STRING for the significant repair genes and compare its hubs with the co-expression hubs — correlated expression and physical interaction are different kinds of evidence, so genes that are hubs in both are a stronger claim than either network alone.
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
# Protein-protein interaction (PPI) network from STRING, for the *significant* repair genes
# (rep.call != "not significant", i.e. the 116 genes that pass genome-wide FDR 0.05 — see Stage 5).
# This is independent evidence from the co-expression network above: STRING edges come from
# curated/experimental physical and functional interactions, not from this cohort's expression data,
# so genes that are hubs in BOTH networks are a stronger claim than either network alone.
from io import StringIO

sig_genes = sorted(rep.loc[rep.call != "not significant", "symbol"].unique())
print(f"{len(sig_genes)} significant repair genes queried against STRING")

string_url = "https://string-db.org/api/tsv/network"
string_params = {
    "identifiers": "%0d".join(sig_genes),   # STRING's documented separator between identifiers
    "species": 9606,                         # Homo sapiens
    "required_score": 700,                   # high-confidence interactions only (STRING score is 0-1000)
    "caller_identity": "bioproj_stage6_biological_interpretation",
}
req = urllib.request.Request(
    string_url,
    data=urllib.parse.urlencode(string_params).encode("utf-8"),
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    string_tsv = resp.read().decode("utf-8")

ppi_edges = pd.read_csv(StringIO(string_tsv), sep="\t")[["preferredName_A", "preferredName_B", "score"]].drop_duplicates()
print(f"STRING returned {len(ppi_edges)} high-confidence (score ≥ 700) interactions")

G_ppi = nx.Graph()
G_ppi.add_nodes_from(sig_genes)
for _, r in ppi_edges.iterrows():
    G_ppi.add_edge(r["preferredName_A"], r["preferredName_B"], score=r["score"] / 1000)
G_ppi.remove_nodes_from([n for n in list(G_ppi.nodes) if G_ppi.degree(n) == 0])
print(f"PPI network: {G_ppi.number_of_nodes()} genes with >=1 interaction, {G_ppi.number_of_edges()} edges")

deg_ppi = pd.Series(dict(G_ppi.degree())).sort_values(ascending=False)
btw_ppi = pd.Series(nx.betweenness_centrality(G_ppi)).sort_values(ascending=False)
print("top PPI hubs by degree:", ", ".join(f"{g} ({d})" for g, d in deg_ppi.head(10).items()))
print("top PPI hubs by betweenness:", ", ".join(btw_ppi.head(6).index))

# Compare against the co-expression hubs (deg, btw) computed in the cell above
K = 10
coexp_top, ppi_top = set(deg.head(K).index), set(deg_ppi.head(K).index)
shared = sorted(coexp_top & ppi_top)
print(f"\ntop-{K} hubs in BOTH networks: {', '.join(shared) if shared else '(none)'}")

hub_compare = pd.DataFrame({
    "co-expression degree": deg,
    "co-expression rank": deg.rank(ascending=False, method="min"),
    "PPI degree": deg_ppi.reindex(deg.index).fillna(0).astype(int),
    "PPI rank": deg_ppi.reindex(deg.index).rank(ascending=False, method="min"),
}).sort_values("co-expression rank").head(15)
display(hub_compare)

fig, ax = plt.subplots(figsize=(10, 9))
pos_ppi = nx.kamada_kawai_layout(G_ppi)
nx.draw_networkx_edges(G_ppi, pos_ppi, ax=ax, alpha=0.2, width=0.6)
node_colors = ["#8B1E3F" if n in coexp_top else "#AEB6C4" for n in G_ppi.nodes]
nx.draw_networkx_nodes(G_ppi, pos_ppi, ax=ax, node_size=[40 + 20 * G_ppi.degree(n) for n in G_ppi.nodes],
                        node_color=node_colors, edgecolors="k", linewidths=0.3)
nx.draw_networkx_labels(G_ppi, pos_ppi, ax=ax, font_size=7)
ax.set_title(f"STRING PPI network of significant repair genes (score ≥ 0.7)\nred = also a top-{K} co-expression hub")
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
# | *BRCA1* is lower than proliferation predicts in TNBC, with a low-expression tail in 16.6% of TNBC | Sporadic TNBC often share features of *BRCA1*-mutant cancers ("BRCAness"), and *BRCA1* can be silenced by promoter methylation in sporadic breast tumours (Turner et al., 2007; Esteller et al., 2000; Foulkes et al., 2010) | Consistent; needs methylation or mutation data to confirm |
# | HR genes such as *RAD51* and *BRCA2* are *higher* in TNBC, the subtype most associated with HR deficiency | HR deficiency is a functional state, detected through mutations, methylation or genomic "scars" (Knijnenburg et al., 2018; Davies et al., 2017), not mRNA levels | Paradox explained: expression ≠ function |
# | TNBC patients are younger and more often Black / African American | Basal-like breast cancer is more prevalent in younger African-American women (Carey et al., 2006) | Agrees |
#
# ### References
#
# Every reference below was checked against the publisher/journal record. One correction was found: the
# literature table previously cited the BRCA1-dysfunction paper as "Turner et al., 2004" — the paper was
# published online 2 Oct 2006 and in print in 2007, so it is cited here as **Turner et al., 2007**.
#
# 1. Perou CM, Sørlie T, Eisen MB, et al. Molecular portraits of human breast tumours. *Nature*. 2000;406(6797):747–752. doi: [10.1038/35021093](https://doi.org/10.1038/35021093)
# 2. The Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. *Nature*. 2012;490(7418):61–70. doi: [10.1038/nature11412](https://doi.org/10.1038/nature11412)
# 3. Whitfield ML, Sherlock G, Saldanha AJ, et al. Identification of genes periodically expressed in the human cell cycle and their expression in tumors. *Mol Biol Cell*. 2002;13(6):1977–2000. doi: [10.1091/mbc.02-02-0030](https://doi.org/10.1091/mbc.02-02-0030)
# 4. Lehmann BD, Bauer JA, Chen X, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. *J Clin Invest*. 2011;121(7):2750–2767. doi: [10.1172/JCI45014](https://doi.org/10.1172/JCI45014)
# 5. Turner NC, Reis-Filho JS, Russell AM, et al. BRCA1 dysfunction in sporadic basal-like breast cancer. *Oncogene*. 2007;26:2126–2132 (published online 2 Oct 2006). doi: [10.1038/sj.onc.1210014](https://doi.org/10.1038/sj.onc.1210014) — *corrected from "2004" in the table above.*
# 6. Esteller M, Silva JM, Dominguez G, et al. Promoter hypermethylation and BRCA1 inactivation in sporadic breast and ovarian tumors. *J Natl Cancer Inst*. 2000;92(7):564–569. doi: [10.1093/jnci/92.7.564](https://doi.org/10.1093/jnci/92.7.564)
# 7. Foulkes WD, Smith IE, Reis-Filho JS. Triple-negative breast cancer. *N Engl J Med*. 2010;363(20):1938–1948. doi: [10.1056/NEJMra1001389](https://doi.org/10.1056/NEJMra1001389)
# 8. Knijnenburg TA, Wang L, Zimmermann MT, et al. Genomic and molecular landscape of DNA damage repair deficiency across The Cancer Genome Atlas. *Cell Rep*. 2018;23(1):239–254.e6. doi: [10.1016/j.celrep.2018.03.076](https://doi.org/10.1016/j.celrep.2018.03.076)
# 9. Davies H, Glodzik D, Morganella S, et al. HRDetect is a predictor of BRCA1 and BRCA2 deficiency based on mutational signatures. *Nat Med*. 2017;23(4):517–525. doi: [10.1038/nm.4292](https://doi.org/10.1038/nm.4292)
# 10. Carey LA, Perou CM, Livasy CA, et al. Race, breast cancer subtypes, and survival in the Carolina Breast Cancer Study. *JAMA*. 2006;295(21):2492–2502. doi: [10.1001/jama.295.21.2492](https://doi.org/10.1001/jama.295.21.2492)
#
# **Interpretation (Stage 6).**
#
# * **Pathway enrichment:** the proliferation control is the most enriched set (NES 2.19, FDR < 0.001), which shows the method works. Among repair pathways, only **homologous recombination is significantly enriched towards TNBC** (NES 1.81, FDR 0.009). MMR (NES 1.50, FDR 0.065), all repair genes together (NES 1.56, FDR 0.057) and FA (NES 1.40, FDR 0.10) point the same way without reaching significance. BER, NER and NHEJ show no enrichment (FDR ≥ 0.48). The HR leading-edge genes (*CHEK1, RFC4, RAD54L, RAD51AP1, RAD51B, POLD1 …*) are the same genes highlighted in Stage 5.
# * **Co-expression network:** 55 repair genes have at least one strong co-expression partner. Most of them form one dense core made of two modules (25 + 13 genes) whose hubs are *RAD51AP1, EXO1, FANCB, BLM, CHEK1, NEIL3, RAD54L* and *RAD51*. Both module scores correlate strongly with proliferation (ρ = 0.85 and 0.88) and are higher in TNBC (AUC 0.80 and 0.79). After proliferation adjustment, though, their median TNBC difference is close to zero (0.14 and −0.02). This is a **single replication-coupled HR/FA programme** that switches on in dividing cells, and TNBC divide more. A separate small module (*MSH3, RAD50, POLK, FAN1, RAD17*) behaves differently. It is **lower in TNBC** (AUC 0.16, i.e. higher in non-TNBC), not tied to proliferation (ρ = −0.31), and keeps its difference after adjustment (median −0.53). This is the clearest proliferation-independent repair difference in the data.
# * **PPI vs co-expression hubs:** `# TODO (Kian) — replace with what your own run prints.` Run the new STRING cell above in Colab and read off: (a) how many of the top-10 co-expression hubs (`RAD51AP1, EXO1, FANCB, BLM, CHEK1, …`) are *also* top-10 STRING PPI hubs — printed as "top-10 hubs in BOTH networks"; (b) name any gene that is a hub in one network but not the other, and say what that means (a co-expression-only hub is "co-regulated but not known to physically interact"; a PPI-only hub is "interacts physically but isn't co-regulated here"). A gene that is a hub in *both* networks (shared regulation **and** shared physical interaction) is the strongest candidate for a real functional module — call that out by name once you have it.
# * **BRCA1:** median *BRCA1* expression is the same in both groups (9.97 vs 9.95), but TNBC have a long low tail: 16.6% of TNBC fall below the 5th percentile of non-TNBC (5.1% expected). Given how proliferative TNBC are, *BRCA1* is lower than expected (p = 7.8 × 10⁻⁹). This fits a subset of TNBC in which *BRCA1* is silenced (the BRCAness phenotype), while most TNBC have normal *BRCA1* driven up by proliferation. Expression alone cannot prove repair deficiency.
# * **Overall:** our results support the literature on direction (HR/FA up in TNBC, a BRCA1-low subset) but show that **most of the repair difference is proliferation**, a point often left out when repair-gene expression is compared between subtypes.
