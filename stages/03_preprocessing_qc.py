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
# ## Stage 3 · Preprocessing and quality control

# %% [markdown]
# ### 3.1 Integrity of the count matrix

# %%
print("missing values      :", int(counts_raw.isna().sum().sum()))
print("negative counts     :", int((counts_raw < 0).sum().sum()))
print("non-integer counts  :", int((counts_raw % 1 != 0).sum().sum()))
dup_genes = counts_raw.index.duplicated(keep=False)
print("duplicated gene IDs :", counts_raw.index[dup_genes].nunique(), "IDs appear on", int(dup_genes.sum()), "rows")
print("duplicated samples  :", int(counts_raw.columns.duplicated().sum()))

# Some Entrez IDs map to more than one GENCODE gene (e.g. X/Y pseudo-autosomal copies).
# Reads from those rows belong to the same Entrez gene, so we sum them.
counts_all = counts_raw.groupby(level=0).sum()
print("genes after merging duplicated IDs:", counts_all.shape[0])

# %% [markdown]
# ### 3.2 Sample selection and TNBC labels
#
# * Keep **primary tumours** only (barcode type `01`) and one sample per patient.
# * **HER2:** positive if IHC is 3+/positive **or** FISH shows amplification. Negative if FISH is negative, or IHC is negative with no positive FISH. Otherwise (e.g. IHC equivocal with no FISH) it is *unknown*.
# * **TNBC** = ER-negative **and** PR-negative **and** HER2-negative. **non-TNBC** = ER-, PR- or HER2-positive. Anything else (a negative receptor with missing or indeterminate partners) is *unknown* and excluded.
# * **Male patients are excluded**, because male breast cancer is biologically different and too rare (n = 12) to model.

# %%
def her2_status(row):
    ihc, fish = row["IHC_HER2"], row["HER2_FISH_STATUS"]
    if ihc == "Positive" or fish == "Positive":
        return "Positive"
    if fish == "Negative" or ihc == "Negative":
        return "Negative"
    return "Unknown"

def tnbc_group(row):
    if "Positive" in (row["ER"], row["PR"], row["HER2"]):
        return "non-TNBC"
    if row["ER"] == "Negative" and row["PR"] == "Negative" and row["HER2"] == "Negative":
        return "TNBC"
    return "Unknown"

clin = bcr.set_index("PATIENT_ID")
labels = pd.DataFrame({"ER": clin["ER_STATUS_BY_IHC"], "PR": clin["PR_STATUS_BY_IHC"]})
labels["HER2"] = clin.apply(her2_status, axis=1)
labels["group"] = labels.apply(tnbc_group, axis=1)
gp = gdc_patient.set_index("PATIENT_ID")
labels = labels.join(gp[["SEX", "AGE", "RACE"]], how="left")

flow = []
def log_step(step, n_samples, n_patients=None):
    flow.append({"step": step, "samples": n_samples, "patients": n_patients if n_patients is not None else n_samples})

log_step("All samples in count matrix", counts_all.shape[1], counts_all.columns.str[:12].nunique())
sample_type = counts_all.columns.str[13:15]
print("sample-type codes:", pd.Series(sample_type).value_counts().to_dict(), "(01 = primary tumour, 06 = metastasis)")
counts_pt = counts_all.loc[:, sample_type == "01"]
log_step("Primary tumours (type 01)", counts_pt.shape[1], counts_pt.columns.str[:12].nunique())
assert counts_pt.columns.str[:12].is_unique, "more than one primary tumour per patient"
counts_pt.columns = counts_pt.columns.str[:12]          # one sample per patient -> use patient ID

has_status = counts_pt.columns.intersection(labels.index)
log_step("With receptor-status record", len(has_status))
known = [p for p in has_status if labels.loc[p, "group"] != "Unknown"]
log_step("TNBC status determinable", len(known))
female = [p for p in known if labels.loc[p, "SEX"] == "Female"]
log_step("Female patients (final cohort)", len(female))

display(pd.DataFrame(flow).set_index("step"))
print("\nreceptor-status combinations among patients with RNA-seq:")
display(pd.crosstab([labels.loc[has_status, "ER"], labels.loc[has_status, "PR"]], labels.loc[has_status, "HER2"]))

# %%
cohort = labels.loc[female].copy()
counts = counts_pt[female]
y = (cohort["group"] == "TNBC").astype(int)
print(cohort["group"].value_counts().to_string())
print(f"TNBC fraction: {y.mean():.1%}")

# %% [markdown]
# ### 3.3 Library size and detected genes per sample

# %%
lib_size = counts.sum(axis=0)
detected = (counts > 0).sum(axis=0)
log_lib = np.log10(lib_size)
mad = stats.median_abs_deviation(log_lib)
low_lib = log_lib < (log_lib.median() - 4 * mad)
print(f"library size: median {lib_size.median()/1e6:.1f} M reads (range {lib_size.min()/1e6:.1f}–{lib_size.max()/1e6:.1f} M)")
print(f"detected genes: median {detected.median():,.0f} (range {detected.min():,}–{detected.max():,})")
print(f"samples more than 4 MADs below the median log-library size: {int(low_lib.sum())} "
      f"(smallest library {lib_size.min()/1e6:.1f} M reads)")
print("library size by group (M reads):")
display((lib_size / 1e6).groupby(cohort["group"]).describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for g, col in GROUP_COLORS.items():
    axes[0].hist(lib_size[cohort["group"] == g] / 1e6, bins=40, alpha=0.6, color=col, label=g)
axes[0].set(xlabel="library size (million reads)", ylabel="samples", title="Library size per sample")
axes[0].legend()
axes[1].scatter(lib_size / 1e6, detected, s=8, c=cohort["group"].map(GROUP_COLORS))
axes[1].set(xlabel="library size (million reads)", ylabel="genes with ≥1 read", title="Detected genes vs depth")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.4 Filtering low-expressed genes
# A gene is kept if it has at least 1 count per million (CPM) in at least 15% of samples (≈ the size of the TNBC group). Genes below this are mostly noise and cannot be tested reliably.

# %%
cpm = counts / lib_size * 1e6
keep_gene = (cpm >= 1).sum(axis=1) >= 0.15 * counts.shape[1]
counts_f = counts.loc[keep_gene]
print(f"genes before filter: {counts.shape[0]:,}  after: {counts_f.shape[0]:,}")
repair_present = repair.index[repair.index.isin(counts_f.index)]
repair_dropped = repair.index.difference(repair_present)
print(f"DNA-repair genes kept: {len(repair_present)} of {len(repair)}; dropped (not expressed in breast tumours):",
      ", ".join(repair.loc[repair_dropped, "symbol"]))

# %% [markdown]
# ### 3.5 Normalisation
# We use DESeq2's **median-of-ratios** size factors (the same method PyDESeq2 uses in Stage 5) and then `log2(normalised count + 1)`. This removes differences in sequencing depth and compresses the range, so samples can be compared in plots, PCA and machine learning.

# %%
def median_of_ratios(cnt):
    log_c = np.log(cnt.replace(0, np.nan))
    log_geo_mean = log_c.mean(axis=1)
    usable = log_c.notna().all(axis=1)
    return np.exp((log_c.loc[usable].sub(log_geo_mean[usable], axis=0)).median(axis=0))

size_factors = median_of_ratios(counts_f)
logn = np.log2(counts_f / size_factors + 1)            # genes x samples
print(f"size factors: median {size_factors.median():.2f}, range {size_factors.min():.2f}–{size_factors.max():.2f}")
print("correlation of size factor with library size:", round(np.corrcoef(size_factors, lib_size)[0, 1], 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rng = np.random.default_rng(SEED)
for s in rng.choice(logn.columns, 60, replace=False):
    sns.kdeplot(np.log2(counts_f[s] + 1), ax=axes[0], lw=0.6, color="grey", alpha=0.5)
    sns.kdeplot(logn[s], ax=axes[1], lw=0.6, color="teal", alpha=0.5)
axes[0].set(title="Before normalisation: log2(count+1)", xlabel="log2 expression")
axes[1].set(title="After median-of-ratios normalisation", xlabel="log2 normalised expression")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.6 Biological sanity checks
# Some biology must hold if the labels and expression data are matched correctly:
#
# * *ESR1* mRNA should be much lower in ER-negative tumours, *PGR* lower in PR-negative tumours, and *ERBB2* higher in HER2-positive tumours.
# * Y-chromosome genes (*RPS4Y1, DDX3Y, KDM5D*) should be essentially silent in female patients. The 12 male patients excluded above serve as a positive control.

# %%
def gene(sym_or_id):
    eid = sym_or_id if isinstance(sym_or_id, (int, np.integer)) else {**MARKERS, **PROLIF,
           **{v: k for k, v in repair["symbol"].items()}}[sym_or_id]
    return logn.loc[eid] if eid in logn.index else np.log2(counts.loc[eid] / size_factors + 1)

checks = [("ESR1", "ER"), ("PGR", "PR"), ("ERBB2", "HER2")]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
rows = []
for ax, (sym, receptor) in zip(axes, checks):
    status = cohort[receptor].where(cohort[receptor].isin(["Positive", "Negative"]))
    df = pd.DataFrame({"expr": gene(sym), "status": status}).dropna()
    sns.boxplot(data=df, x="status", y="expr", order=["Negative", "Positive"], ax=ax, fliersize=1,
                palette={"Negative": "#AAB7B8", "Positive": "#F5B041"})
    auc = stats.mannwhitneyu(df.expr[df.status == "Positive"], df.expr[df.status == "Negative"]).statistic / (
        (df.status == "Positive").sum() * (df.status == "Negative").sum())
    rows.append({"gene": sym, "receptor": receptor, "AUC (expr separates status)": round(auc, 3),
                 "median log2 expr in negative": round(df.expr[df.status == "Negative"].median(), 2),
                 "median log2 expr in positive": round(df.expr[df.status == "Positive"].median(), 2)})
    ax.set(title=f"{sym} by {receptor} status", xlabel=f"{receptor} (clinical)", ylabel="log2 normalised expression")
# Sex check on all primary tumours: the 12 male patients (excluded above) act as a positive control
cpm_pt = counts_pt / counts_pt.sum(axis=0) * 1e6
y_score = np.log2(cpm_pt.loc[[MARKERS[g] for g in ["RPS4Y1", "DDX3Y", "KDM5D"]]] + 1).mean(axis=0)
sex_df = pd.DataFrame({"Y-gene score": y_score, "sex": gp["SEX"].reindex(counts_pt.columns)}).dropna()
sns.boxplot(data=sex_df, x="sex", y="Y-gene score", order=["Female", "Male"], ax=axes[3], fliersize=0,
            palette={"Female": "#D7BDE2", "Male": "#A9CCE3"})
sns.stripplot(data=sex_df, x="sex", y="Y-gene score", order=["Female", "Male"], ax=axes[3], size=2, color="k", alpha=0.4)
axes[3].set(title="Y-chromosome genes (RPS4Y1, DDX3Y, KDM5D)", ylabel="mean log2(CPM + 1)", xlabel="recorded sex")
plt.tight_layout(); plt.show()
display(pd.DataFrame(rows))

male_min = sex_df.loc[sex_df.sex == "Male", "Y-gene score"].min()
fem_max = sex_df.loc[sex_df.sex == "Female", "Y-gene score"].max()
print(f"Y-gene score: highest female {fem_max:.2f} vs lowest male {male_min:.2f}; "
      f"female samples in the male range: {int((sex_df.loc[sex_df.sex == 'Female', 'Y-gene score'] >= male_min).sum())}")

# Suspicious labels: ER-negative by IHC but ESR1 mRNA at a typical ER-positive level
esr1 = gene("ESR1")
er_pos_q10 = esr1[cohort["ER"] == "Positive"].quantile(0.10)
discordant = cohort.index[(cohort["group"] == "TNBC") & (esr1 > er_pos_q10)]
print(f"TNBC tumours with ESR1 above the 10th percentile of ER-positive tumours: {len(discordant)} "
      f"of {int(y.sum())} ({len(discordant)/y.sum():.1%})")

# %% [markdown]
# **Interpretation (Stage 3).** 
#
# * **Integrity:** no missing, negative or non-integer values, as expected for raw counts. 34 Entrez IDs appeared on two rows (genes annotated twice in GENCODE); we summed them, giving 40,761 unique genes.
# * **Cohort:** 1,097 samples → 1,090 primary tumours (7 metastases removed) → 1,089 with a receptor record → 1,018 with a determinable TNBC status (71 with incomplete or indeterminate receptors excluded) → **1,006 female patients: 157 TNBC (15.6%) and 849 non-TNBC**. The 15.6% TNBC share matches the 15–20% usually reported, a first sign that the labels are sensible. The classes are imbalanced (about 1 : 5), which matters for Stage 7.
# * **Sequencing depth:** median 57.5 M reads per tumour, almost the same in TNBC (59.4 M) and non-TNBC (57.1 M), so depth cannot explain group differences. 16 samples have small libraries for this cohort, but even the smallest has 19.2 M reads, plenty for gene-level counts. We kept them and let normalisation correct for depth.
# * **Filtering:** 16,697 genes pass the filter, and 148 of the 156 repair genes are expressed. Some exclusions are biological: *DNTT* is only active in developing lymphocytes. Others are technical. *SLX1A/SLX1B* are duplicated copies of each other, so their reads cannot be placed uniquely. **Suspicious:** *GTF2H4* and *TOP3B* are housekeeping genes expressed in essentially every cell, yet they have a median of only ~4 reads per tumour. This is most likely a gene-annotation or read-assignment artefact (reads that overlap more than one annotated gene are discarded by gene-level counting), not true absence. We note it as a limitation rather than force them back in.
# * **Normalisation:** size factors (0.28–2.10) track library size (r = 0.93), and the per-sample expression distributions line up after normalisation.
# * **Biology checks pass:** *ESR1* separates ER-positive from ER-negative tumours almost perfectly (AUC 0.97; median log2 14.9 vs 8.4, about 90-fold), and *PGR* separates PR status (AUC 0.93). *ERBB2* separates HER2 status less sharply (AUC 0.81). That is expected: clinical HER2 positivity is defined by gene amplification or protein level, and equivocal cases sit in between. Every female sample's Y-gene score (maximum 2.3) is below every male sample's (minimum 5.0), so recorded sex agrees with the data.
# * **Suspicious labels:** 5 of 157 TNBC tumours (3.2%) express *ESR1* at a level typical of ER-positive tumours. They may be IHC false negatives or ER-low tumours. We keep them, because dropping samples on the basis of expression would bias the comparison, and we list label noise as a limitation.
