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
# ## Stage 8 · Conclusions

# %%
summary = {
    "cohort": f"{len(cohort)} female primary tumours: {int(y.sum())} TNBC, {int((1-y).sum())} non-TNBC",
    "repair genes tested": len(repair_present),
    "repair genes FDR<0.05 (up/down)": f"{int(((rep.padj_repair_family<0.05)&(rep.log2FoldChange>0)).sum())} / {int(((rep.padj_repair_family<0.05)&(rep.log2FoldChange<0)).sum())}",
    "repair genes ≥2-fold (up/down)": f"{int((rep.call=='UP ≥2x in TNBC').sum())} / {int((rep.call=='DOWN ≥2x in TNBC').sum())}",
    "competitive set test p (shift towards TNBC)": f"{p_mean:.4f}",
    "after proliferation adjustment (up/down)": f"{int(up1.sum())} / {int(dn1.sum())}",
    "test ROC-AUC repair LR": round(test_table.loc['DNA-repair genes (LR)', 'ROC-AUC'], 3),
    "test ROC-AUC proliferation only": round(test_table.loc['Proliferation score only (LR)', 'ROC-AUC'], 3),
    "CV ROC-AUC random genes (mean)": round(np.mean(rand_aucs), 3),
}
display(pd.Series(summary, name="key result").to_frame())

# %% [markdown]
# ### Answer to the research question
# **Yes, TNBC show a distinct DNA-repair expression profile, but most of it reflects their higher proliferation.**
#
# 116 of 148 repair genes differ at FDR < 0.05, and the set as a whole is shifted towards higher expression in TNBC more than random genes (p = 0.0003), so we reject H0. The predicted direction (HR/FA/MMR up in TNBC) holds, but adjusting for proliferation removes most of it: only 36 genes stay higher, and more genes (71) become *lower* than proliferation predicts, with a median of just 26% of the TNBC difference retained. The proliferation-independent signal is:
#
# * FA-core and checkpoint genes (*FANCE, FANCA, USP1, CHEK1*) and *RAD51B* stay higher in TNBC.
# * ***BRCA1***, *POLN*, *POLD4* and a *RAD50/MSH3/POLK/RAD17/FAN1* module are lower than expected.
#
# ### One point per objective
# 1. **Cohort and data (O1):** we built a quality-controlled count matrix from open-access TCGA-BRCA RNA-seq counts (40,796 genes × 1,097 tumours, SHA-256-verified against a pinned GDC mirror and holding all 156 repair genes). Keeping one primary tumour per patient with determinable receptor status gives 1,006 female tumours (157 TNBC, 849 non-TNBC; 15.6% TNBC); a 15% expression filter retains 16,697 genes including 148 of 156 repair genes (stable at 10%/25%: 149/147). The labels passed biological checks (*ESR1* AUC 0.965, *PGR* 0.928, *ERBB2* 0.809), and only 5 TNBC (3.2%) look ER-positive at the mRNA level — kept and flagged, not reclassified.
# 2. **Differential expression (O2):** 79 repair genes are up and 37 down in TNBC (FDR < 0.05), and 11 change at least two-fold (10 up, led by *FANCA, RAD54L, CHEK1*; 1 down, *POLN*); the result is robust to age (log2FC *r* = 1.00). The Wilcoxon test confirms the DESeq2 direction for 94% of genes. As a set, repair genes are shifted *up* in TNBC more than random genes (mean Wald 2.64 vs 0.49, p = 0.0003), but in magnitude they are no more changed than random (mean |Wald| 6.63 vs 6.46, p = 0.33).
# 3. **Biological interpretation (O3):** homologous recombination is the only significantly enriched pathway (GSEA NES 1.80, FDR 0.010). The up-regulated genes form two co-expressed, replication-coupled HR/FA modules that track proliferation (ρ ≈ 0.84–0.91), and a STRING protein–protein network of the significant repair genes (114 genes, 1,523 high-confidence edges) shares its top hubs with the co-expression network — *EXO1, FANCD2* and *RAD51* rank in the top 10 of both. Two findings do not depend on proliferation: *BRCA1* is lower than expected in TNBC (median residual −0.25; a low tail in 16.6% of tumours, consistent with BRCAness), and a down-regulated *RAD50/MSH3/POLK* module.
# 4. **Machine learning (O4):** on the held-out test set (202 tumours, 32 TNBC), a logistic regression on repair genes reaches test ROC-AUC 0.93 (95% CI 0.87–0.97; 5 × 5 repeated CV 0.969 ± 0.013), balanced accuracy 0.87, sensitivity 0.81 and specificity 0.93, clearly beating the majority-class (0.50) and proliferation-only (0.83) baselines. But random sets of 148 non-repair genes do just as well (0.967 ± 0.013; repair genes ahead in only 61% of paired folds), and tuning the random-forest threshold to 0.22 lifts balanced accuracy from 0.70 to 0.85 (TNBC recall 0.44 → 0.78). So the classifier confirms a distinct profile but **does not show that repair genes are uniquely informative**.
#
# ### Limitations
# * **Expression ≠ function.** High HR-gene mRNA does not mean HR works, and low *BRCA1* mRNA does not prove deficiency. We used no mutation, methylation or genomic-scar data.
# * **Proliferation adjustment is imperfect.** The 11-gene score is a proxy, so some confounding may remain. Adjustment may also remove real biology if repair changes themselves drive proliferation.
# * **Bulk tumour RNA.** Each sample mixes tumour, stromal and immune cells. We did not adjust for tumour purity, and TNBC often have more immune infiltration.
# * **Label noise.** Receptor status was scored at many hospitals over many years under older guidelines. 71 tumours with incomplete status were excluded, and 5 TNBC have *ESR1* at ER-positive levels. Eight repair genes (e.g. *GTF2H4, TOP3B*) fell below the expression filter and were not tested.
# * **Representation.** Most patients are White (57–70%), about 5% Asian, and none were recruited in Africa. The results may not transfer to Kenyan or other African patients.
# * **One dataset.** The analysis is cross-sectional with no external validation, and the test set holds only 32 TNBC, so the metrics have wide confidence intervals.
#
# ### Future steps
# * Add TCGA mutation, *BRCA1* promoter-methylation and copy-number data to test whether the low-*BRCA1* TNBC are truly BRCA1-deficient, and compare with genomic HRD scores (e.g. HRDetect; Davies et al., 2017).
# * Adjust for tumour purity and immune infiltration, and compare IHC-defined TNBC with PAM50 basal-like subtypes.
# * Validate in an independent cohort (e.g. METABRIC) and, ideally, in an African breast-cancer cohort.
# * Relate the repair profile to treatment response (platinum, PARP inhibitors) and survival.
#
# ### Code and outputs
# This notebook runs top to bottom in Google Colab. It is stored on GitHub with its outputs, together with the result tables written to `results/` (repair-gene DE, shrunken and age- and proliferation-adjusted tables, the genome-wide DE table, the GSEA table and the ML test metrics).

# %% [markdown]
# ## References
#
# * Carey LA, et al. Race, breast cancer subtypes, and survival in the Carolina Breast Cancer Study. *JAMA* 2006;295:2492–2502.
# * Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. *Nature* 2012;490:61–70.
# * Cerami E, et al. The cBio cancer genomics portal. *Cancer Discov* 2012;2:401–404.
# * Davies H, et al. HRDetect is a predictor of BRCA1 and BRCA2 deficiency based on mutational signatures. *Nat Med* 2017;23:517–525.
# * Esteller M, et al. Promoter hypermethylation and BRCA1 inactivation in sporadic breast and ovarian tumors. *J Natl Cancer Inst* 2000;92:564–569.
# * Fang Z, Liu X, Peltz G. GSEApy: a comprehensive package for performing gene set enrichment analysis in Python. *Bioinformatics* 2023;39:btac757.
# * Foulkes WD, Smith IE, Reis-Filho JS. Triple-negative breast cancer. *N Engl J Med* 2010;363:1938–1948.
# * Grossman RL, et al. Toward a shared vision for cancer genomic data. *N Engl J Med* 2016;375:1109–1112.
# * Knijnenburg TA, et al. Genomic and molecular landscape of DNA damage repair deficiency across The Cancer Genome Atlas. *Cell Rep* 2018;23:239–254.
# * Lehmann BD, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. *J Clin Invest* 2011;121:2750–2767.
# * Li Y, Ge X, Peng F, Li W, Li JJ. Exaggerated false positives by popular differential expression methods when analyzing human population samples. *Genome Biol* 2022;23:79.
# * Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biol* 2014;15:550.
# * Muzellec B, et al. PyDESeq2: a python package for bulk RNA-seq differential expression analysis. *Bioinformatics* 2023;39:btad547.
# * Nielsen TO, et al. A comparison of PAM50 intrinsic subtyping with immunohistochemistry and clinical prognostic factors in tamoxifen-treated estrogen receptor-positive breast cancer. *Clin Cancer Res* 2010;16:5222–5232.
# * Perou CM, et al. Molecular portraits of human breast tumours. *Nature* 2000;406:747–752.
# * Robson M, et al. Olaparib for metastatic breast cancer in patients with a germline BRCA mutation. *N Engl J Med* 2017;377:523–533.
# * Subramanian A, et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *PNAS* 2005;102:15545–15550.
# * Turner N, Tutt A, Ashworth A. Hallmarks of 'BRCAness' in sporadic cancers. *Nat Rev Cancer* 2004;4:814–819.
# * Whitfield ML, et al. Identification of genes periodically expressed in the human cell cycle and their expression in tumors. *Mol Biol Cell* 2002;13:1977–2000.