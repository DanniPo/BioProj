
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
#
# This stage performs:
# 1. Count-matrix integrity checks.
# 2. Primary tumour selection and TNBC classification.
# 3. Sequencing-depth quality control.
# 4. Low-expression gene filtering.
# 5. Filtering sensitivity analysis.
# 6. Count normalisation.
# 7. Biological sanity checks.
# 8. ESR1-discordant TNBC investigation.

# %%
# ============================================================
# 3.0 IMPORTS AND INPUT VALIDATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from IPython.display import display

required_objects = [
    "counts_raw",
    "bcr",
    "gdc_patient",
    "repair",
    "MARKERS",
    "PROLIF",
    "GROUP_COLORS",
    "SEED"
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        "Stage 3 is missing outputs from earlier stages: "
        + ", ".join(missing_objects)
        + ". Run the earlier stages before Stage 3."
    )

for name, obj in [
    ("counts_raw", counts_raw),
    ("bcr", bcr),
    ("gdc_patient", gdc_patient),
    ("repair", repair)
]:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

if counts_raw.empty:
    raise ValueError("The raw count matrix is empty.")

if "symbol" not in repair.columns:
    raise KeyError(
        "The repair DataFrame must contain a 'symbol' column."
    )

print("Stage 3 inputs successfully loaded.")
print("Raw count matrix shape:", counts_raw.shape)


# %% [markdown]
# ### 3.1 Integrity of the count matrix
#
# Check missing values, negative values, non-integer counts
# and duplicated gene or sample identifiers.
#
# Duplicated gene IDs are combined by summing their counts.
# To reduce memory use, only duplicated rows are grouped.

# %%
# ============================================================
# 3.1 COUNT MATRIX INTEGRITY
# ============================================================

print("COUNT MATRIX INTEGRITY")
print("=" * 45)

counts_raw = counts_raw.copy()
counts_raw.columns = counts_raw.columns.astype(str)

# Check that the matrix contains numeric values.

if not all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in counts_raw.dtypes
):
    raise TypeError(
        "The count matrix contains non-numeric columns."
    )

# Check count values.

missing_values = int(
    counts_raw.isna().to_numpy().sum()
)

if missing_values > 0:
    raise ValueError(
        f"The count matrix contains {missing_values:,} missing values."
    )

negative_counts = int(
    (counts_raw < 0).to_numpy().sum()
)

if negative_counts > 0:
    raise ValueError(
        f"The count matrix contains {negative_counts:,} negative values."
    )

non_integer_counts = int(
    (counts_raw % 1 != 0).to_numpy().sum()
)

if non_integer_counts > 0:
    raise ValueError(
        f"The count matrix contains {non_integer_counts:,} non-integer values."
    )

duplicate_sample_columns = int(
    counts_raw.columns.duplicated().sum()
)

if duplicate_sample_columns > 0:
    raise ValueError(
        "The count matrix contains duplicated sample columns."
    )

duplicate_mask = counts_raw.index.duplicated(keep=False)

duplicate_gene_ids = int(
    counts_raw.index[duplicate_mask].nunique()
)

duplicate_gene_rows = int(
    duplicate_mask.sum()
)

print("Missing values:", missing_values)
print("Negative counts:", negative_counts)
print("Non-integer counts:", non_integer_counts)
print("Duplicated gene IDs:", duplicate_gene_ids)
print("Rows belonging to duplicated gene IDs:", duplicate_gene_rows)
print("Duplicated sample columns:", duplicate_sample_columns)

# Use int32 where possible to reduce memory consumption.
# Fall back to int64 if the raw counts are too large.

maximum_count = int(counts_raw.to_numpy().max())

if maximum_count <= np.iinfo(np.int32).max:
    counts_raw = counts_raw.astype(np.int32)
    print("Count storage: int32 (memory-efficient)")
else:
    counts_raw = counts_raw.astype(np.int64)
    print("Count storage: int64")

# Aggregate only duplicated gene IDs.

if duplicate_gene_rows > 0:

    # Keep the rows with unique gene IDs.
    counts_unique = counts_raw.loc[
        ~duplicate_mask
    ]

    # Select only duplicated rows.
    counts_duplicates = counts_raw.loc[
        duplicate_mask
    ].astype(np.int64)

    # Sum the duplicated gene IDs.
    counts_duplicates_aggregated = (
        counts_duplicates
        .groupby(level=0, sort=False)
        .sum()
    )

    if (
        counts_duplicates_aggregated.to_numpy().max()
        > np.iinfo(np.int32).max
        and counts_raw.dtypes.iloc[0] == np.dtype("int32")
    ):
        raise OverflowError(
            "Aggregated counts exceed the int32 limit. "
            "Use int64 for the count matrix."
        )

    counts_duplicates_aggregated = (
        counts_duplicates_aggregated
        .astype(counts_raw.dtypes.iloc[0])
    )

    # Combine unique and aggregated rows.
    counts_all = pd.concat([
        counts_unique,
        counts_duplicates_aggregated
    ])

else:
    counts_all = counts_raw

# Sort the final gene index.
counts_all = counts_all.sort_index()

if counts_all.index.has_duplicates:
    raise ValueError(
        "Duplicate gene IDs remain after aggregation."
    )

print("\nDuplicate gene IDs aggregated successfully.")
print("Updated count matrix shape:", counts_all.shape)
print("Remaining duplicated gene IDs:", counts_all.index.duplicated().sum())
print("Unique genes:", f"{counts_all.shape[0]:,}")
print("Samples:", f"{counts_all.shape[1]:,}")


# %% [markdown]
# ### 3.2 Sample selection and TNBC classification
#
# Selection criteria:
# - Retain primary tumours only (TCGA sample type 01).
# - Retain one sample per patient.
# - Determine HER2 status using IHC and FISH.
# - Classify TNBC as ER-negative, PR-negative and HER2-negative.
# - Classify non-TNBC when at least one receptor is positive.
# - Exclude unknown receptor status.
# - Retain female patients in the final cohort.

# %%
# ============================================================
# 3.2 SAMPLE SELECTION AND TNBC CLASSIFICATION
# ============================================================

required_bcr_columns = [
    "PATIENT_ID",
    "ER_STATUS_BY_IHC",
    "PR_STATUS_BY_IHC",
    "IHC_HER2",
    "HER2_FISH_STATUS"
]

missing_bcr_columns = [
    col for col in required_bcr_columns
    if col not in bcr.columns
]

if missing_bcr_columns:
    raise KeyError(
        "Missing columns in BCR clinical data: "
        + ", ".join(missing_bcr_columns)
    )

required_gdc_columns = [
    "PATIENT_ID",
    "SEX",
    "AGE",
    "RACE"
]

missing_gdc_columns = [
    col for col in required_gdc_columns
    if col not in gdc_patient.columns
]

if missing_gdc_columns:
    raise KeyError(
        "Missing columns in GDC patient data: "
        + ", ".join(missing_gdc_columns)
    )


# Standardise clinical text values.

def clean_status(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if value in ["positive", "pos", "1", "yes"]:
        return "Positive"

    if value in ["negative", "neg", "0", "no"]:
        return "Negative"

    return "Unknown"


def her2_status(row):

    ihc = clean_status(row["IHC_HER2"])
    fish = clean_status(row["HER2_FISH_STATUS"])

    # A positive result from either test indicates HER2-positive.
    if ihc == "Positive" or fish == "Positive":
        return "Positive"

    # If one test is negative and the other is unknown,
    # retain the available negative result.
    if ihc == "Negative" or fish == "Negative":
        return "Negative"

    return "Unknown"


def tnbc_group(row):

    receptors = (
        row["ER"],
        row["PR"],
        row["HER2"]
    )

    if "Positive" in receptors:
        return "non-TNBC"

    if all(value == "Negative" for value in receptors):
        return "TNBC"

    return "Unknown"


# Prepare BCR clinical information.

clin = bcr.copy()

clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str).str.strip()
clin = clin.set_index("PATIENT_ID")

if clin.index.has_duplicates:
    raise ValueError(
        "The BCR clinical table contains duplicated patient IDs. "
        "Resolve these before continuing."
    )

labels = pd.DataFrame(index=clin.index)

labels["ER"] = clin["ER_STATUS_BY_IHC"].apply(clean_status)
labels["PR"] = clin["PR_STATUS_BY_IHC"].apply(clean_status)

labels["HER2"] = clin.apply(
    her2_status,
    axis=1
)

labels["group"] = labels.apply(
    tnbc_group,
    axis=1
)


# Prepare GDC demographic information.

gp = gdc_patient.copy()

gp["PATIENT_ID"] = gp["PATIENT_ID"].astype(str).str.strip()
gp = gp.set_index("PATIENT_ID")

if gp.index.has_duplicates:
    raise ValueError(
        "The GDC patient table contains duplicated patient IDs."
    )

# Join demographic information to receptor information.
labels = labels.join(
    gp[["SEX", "AGE", "RACE"]],
    how="left"
)

# Standardise recorded sex.
labels["SEX"] = (
    labels["SEX"]
    .astype("string")
    .str.strip()
    .str.capitalize()
)


# Record cohort filtering steps.

flow = []


def log_step(step, n_samples, n_patients=None):

    flow.append({
        "step": step,
        "samples": int(n_samples),
        "patients": int(
            n_patients if n_patients is not None else n_samples
        )
    })


# All samples.

all_patient_ids = counts_all.columns.str[:12]

log_step(
    "All samples in count matrix",
    counts_all.shape[1],
    all_patient_ids.nunique()
)


# Select primary tumours.

if not all(
    len(sample) >= 15
    for sample in counts_all.columns
):
    raise ValueError(
        "Some sample barcodes are too short to identify "
        "their TCGA sample type."
    )

sample_type = counts_all.columns.str[13:15]

print(
    "\nSample-type codes:",
    pd.Series(sample_type).value_counts().to_dict()
)

counts_pt = counts_all.loc[
    :,
    sample_type == "01"
].copy()

log_step(
    "Primary tumours (type 01)",
    counts_pt.shape[1],
    counts_pt.columns.str[:12].nunique()
)

if counts_pt.empty:
    raise ValueError(
        "No primary tumour samples were found."
    )


# Retain one sample per patient.

patient_ids = counts_pt.columns.str[:12]

if patient_ids.has_duplicates:
    duplicated_patients = (
        patient_ids[patient_ids.duplicated()]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Multiple primary tumour samples were found for "
        f"some patients: {duplicated_patients[:10]}"
    )

counts_pt.columns = patient_ids


# Match RNA-seq samples to clinical records.

has_status = counts_pt.columns.intersection(
    labels.index
)

log_step(
    "With receptor-status record",
    len(has_status)
)

known = [
    patient
    for patient in has_status
    if labels.loc[patient, "group"] != "Unknown"
]

log_step(
    "TNBC status determinable",
    len(known)
)


# Retain female patients.

female = [
    patient
    for patient in known
    if labels.loc[patient, "SEX"] == "Female"
]

log_step(
    "Female patients (final cohort)",
    len(female)
)


# Display filtering flow.

print("\nCOHORT FILTERING FLOW")

display(
    pd.DataFrame(flow).set_index("step")
)


# Display receptor combinations.

print("\nRECEPTOR-STATUS COMBINATIONS")

display(
    pd.crosstab(
        [
            labels.loc[has_status, "ER"],
            labels.loc[has_status, "PR"]
        ],
        labels.loc[has_status, "HER2"]
    )
)


# Define final cohort.

if len(female) == 0:
    raise ValueError(
        "No female patients with known receptor status were found."
    )

cohort = labels.loc[female].copy()
counts = counts_pt[female].copy()

y = (
    cohort["group"] == "TNBC"
).astype(int)

print("\nFINAL COHORT COMPOSITION")

display(
    cohort["group"].value_counts().to_frame("Patients")
)

print(f"Final cohort size: {len(cohort):,}")
print(f"TNBC fraction: {y.mean():.1%}")


# %% [markdown]
# ### 3.3 Library size and detected genes per sample
#
# Examine sequencing depth and the number of detected genes.
# Samples with unusually small libraries are flagged for review.
# They are not automatically removed.

# %%
# ============================================================
# 3.3 LIBRARY SIZE AND DETECTED GENES
# ============================================================

lib_size = counts.sum(axis=0)
detected = (counts > 0).sum(axis=0)

if (lib_size <= 0).any():
    bad_samples = lib_size.index[
        lib_size <= 0
    ].tolist()

    raise ValueError(
        "Samples with zero or negative library size: "
        f"{bad_samples[:10]}"
    )

log_lib = np.log10(lib_size.astype(float))

mad = stats.median_abs_deviation(
    log_lib,
    scale=1
)

if mad == 0:
    low_lib = pd.Series(False, index=log_lib.index)
else:
    low_lib = log_lib < (
        log_lib.median() - 4 * mad
    )

print(
    f"Library size: median {lib_size.median()/1e6:.1f} million reads"
)

print(
    f"Library size range: "
    f"{lib_size.min()/1e6:.1f}–{lib_size.max()/1e6:.1f} million reads"
)

print(
    f"Detected genes: median {detected.median():,.0f}"
)

print(
    f"Detected genes range: {detected.min():,}–{detected.max():,}"
)

print("Samples flagged for low library size:", int(low_lib.sum()))


# Library size by TNBC group.

print("\nLIBRARY SIZE BY GROUP (MILLION READS)")

display(
    (lib_size / 1e6)
    .groupby(cohort["group"])
    .describe()
    .round(2)
)


# Plot library size and detected genes.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

for group, color in GROUP_COLORS.items():

    group_mask = cohort["group"] == group

    axes[0].hist(
        lib_size[group_mask] / 1e6,
        bins=40,
        alpha=0.6,
        color=color,
        label=group
    )

axes[0].set(
    xlabel="Library size (million reads)",
    ylabel="Samples",
    title="Library size per sample"
)

axes[0].legend()

axes[1].scatter(
    lib_size / 1e6,
    detected,
    s=8,
    c=cohort["group"].map(GROUP_COLORS)
)

axes[1].set(
    xlabel="Library size (million reads)",
    ylabel="Genes with at least one read",
    title="Detected genes vs sequencing depth"
)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 3.4 Filtering low-expressed genes
#
# Retain a gene if it has at least 1 count per million (CPM)
# in at least 15% of the samples.
#
# The CPM matrix is stored as float32 to reduce memory use.

# %%
# ============================================================
# 3.4 LOW-EXPRESSION GENE FILTERING
# ============================================================

# Calculate CPM using float32 to reduce memory requirements.

counts_float = counts.astype(np.float32)

lib_size_float = lib_size.astype(np.float32)

cpm = counts_float.div(
    lib_size_float,
    axis=1
) * np.float32(1e6)

baseline_threshold = 0.15

minimum_samples = int(
    np.ceil(baseline_threshold * counts.shape[1])
)

keep_gene = (
    (cpm >= 1).sum(axis=1)
    >= minimum_samples
)

counts_f = counts.loc[keep_gene].copy()

print(f"Genes before filtering: {counts.shape[0]:,}")
print(f"Genes after filtering: {counts_f.shape[0]:,}")
print(f"Genes removed: {counts.shape[0] - counts_f.shape[0]:,}")
print(f"Minimum samples required: {minimum_samples:,}")


# Check retained DNA-repair genes.

repair_ids = repair.index

repair_present = repair_ids.intersection(
    counts_f.index
)

repair_dropped = repair_ids.difference(
    repair_present
)

print(
    f"\nDNA-repair genes retained: "
    f"{len(repair_present)} of {len(repair)}"
)

if len(repair_dropped) > 0:

    dropped_symbols = (
        repair.loc[repair_dropped, "symbol"]
        .astype(str)
        .tolist()
    )

    print(
        "DNA-repair genes not retained:",
        ", ".join(dropped_symbols)
    )

else:
    print("All DNA-repair genes were retained.")


# %% [markdown]
# ### 3.4.1 Sensitivity analysis of gene filtering
#
# Compare 10%, 15% and 25% sample-prevalence thresholds.
# The 15% threshold remains the baseline.

# %%
# ============================================================
# 3.4.1 FILTERING SENSITIVITY ANALYSIS
# ============================================================

thresholds = [0.10, 0.15, 0.25]

sensitivity_results = []

for threshold in thresholds:

    required_samples = int(
        np.ceil(threshold * counts.shape[1])
    )

    keep = (
        (cpm >= 1).sum(axis=1)
        >= required_samples
    )

    genes_kept = counts.index[keep]

    repair_kept = repair_ids.intersection(
        genes_kept
    )

    sensitivity_results.append({
        "Threshold": f"{threshold:.0%}",
        "Total genes retained": len(genes_kept),
        "DNA repair genes retained": len(repair_kept)
    })

sensitivity_df = pd.DataFrame(
    sensitivity_results
)

print("GENE-FILTERING SENSITIVITY ANALYSIS")
display(sensitivity_df)


# Compare alternatives with baseline.

baseline_row = sensitivity_df.loc[
    sensitivity_df["Threshold"] == "15%"
].iloc[0]

baseline_genes = int(
    baseline_row["Total genes retained"]
)

baseline_repair = int(
    baseline_row["DNA repair genes retained"]
)

print("\nDIFFERENCE COMPARED WITH 15% BASELINE")

for _, row in sensitivity_df.iterrows():

    print(
        f"{row['Threshold']}: "
        f"{int(row['Total genes retained']) - baseline_genes:+,} "
        "total genes; "
        f"{int(row['DNA repair genes retained']) - baseline_repair:+} "
        "DNA-repair genes"
    )


# %% [markdown]
# ### 3.5 Normalisation
#
# Use a positive-count median-of-ratios approach.
#
# For each gene, calculate its geometric mean using positive
# counts. Calculate ratios for positive observations, then
# take the median ratio for each sample.
#
# Size factors are scaled to have a geometric mean of 1.

# %%
# ============================================================
# 3.5 MEDIAN-OF-RATIOS NORMALISATION
# ============================================================

def median_of_ratios(cnt):

    if cnt.empty:
        raise ValueError(
            "Cannot normalise an empty count matrix."
        )

    if (cnt < 0).to_numpy().any():
        raise ValueError(
            "Negative counts found during normalisation."
        )

    # Convert to float64 for numerical stability.
    values = cnt.astype(np.float64)

    # Replace zeros with NaN for geometric mean calculation.
    log_counts = np.log(
        values.where(values > 0)
    )

    # Calculate geometric means from positive observations.
    log_geo_mean = log_counts.mean(
        axis=1,
        skipna=True
    )

    geo_means = np.exp(log_geo_mean)

    usable = (
        geo_means.notna()
        & np.isfinite(geo_means)
        & (geo_means > 0)
    )

    print(
        "Genes usable for normalisation:",
        int(usable.sum())
    )

    if not usable.any():
        raise ValueError(
            "No genes are usable for median-of-ratios normalisation."
        )

    # Calculate ratios only where counts are positive.
    ratios = (
        values.loc[usable]
        .where(values.loc[usable] > 0)
        .div(geo_means.loc[usable], axis=0)
    )

    ratios = ratios.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Calculate the median ratio for each sample.
    factors = ratios.median(
        axis=0,
        skipna=True
    )

    if factors.isna().any():
        bad = factors.index[
            factors.isna()
        ].tolist()

        raise ValueError(
            "Size factors could not be calculated for samples: "
            f"{bad[:10]}"
        )

    if (factors <= 0).any():
        raise ValueError(
            "Non-positive size factors were calculated."
        )

    # Scale factors to geometric mean 1.
    factors = factors / np.exp(
        np.log(factors).mean()
    )

    return factors


# Calculate size factors using the filtered count matrix.

size_factors = median_of_ratios(counts_f)

if (
    size_factors.isna().any()
    or not np.isfinite(size_factors).all()
    or (size_factors <= 0).any()
):
    raise ValueError(
        "Invalid size factors were produced."
    )


# Apply size factors to the full cohort count matrix.

normalised_counts = counts.astype(np.float64).div(
    size_factors,
    axis=1
)

logn = np.log2(
    normalised_counts + 1
)

sf_lib_correlation = np.corrcoef(
    size_factors.to_numpy(),
    lib_size.to_numpy()
)[0, 1]

print(
    f"\nMedian size factor: {size_factors.median():.3f}"
)

print(
    f"Size factor range: "
    f"{size_factors.min():.3f}–{size_factors.max():.3f}"
)

print(
    "Correlation of size factors with library size:",
    round(sf_lib_correlation, 3)
)


# Plot expression distributions before and after normalisation.
# Use a sample of genes to reduce plotting memory requirements.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

rng = np.random.default_rng(SEED)

number_to_plot = min(
    30,
    len(logn.columns)
)

selected_samples = rng.choice(
    logn.columns.to_numpy(),
    number_to_plot,
    replace=False
)

for sample in selected_samples:

    sns.kdeplot(
        np.log2(counts[sample].to_numpy(dtype=np.float32) + 1),
        ax=axes[0],
        lw=0.6,
        color="grey",
        alpha=0.5
    )

    sns.kdeplot(
        logn[sample].to_numpy(dtype=np.float32),
        ax=axes[1],
        lw=0.6,
        color="teal",
        alpha=0.5
    )

axes[0].set(
    title="Before normalisation",
    xlabel="log2(count + 1)"
)

axes[1].set(
    title="After median-of-ratios normalisation",
    xlabel="log2(normalised count + 1)"
)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 3.6 Biological sanity checks
#
# Examine whether gene expression agrees with clinical labels.
#
# Expected relationships:
# - ESR1 expression should generally be higher in ER-positive tumours.
# - PGR expression should generally be higher in PR-positive tumours.
# - ERBB2 expression should generally be higher in HER2-positive tumours.
# - Y-chromosome genes should generally be higher in male samples.

# %%
# ============================================================
# 3.6 BIOLOGICAL SANITY CHECKS
# ============================================================

# Build a gene-symbol-to-Entrez-ID dictionary.

symbol_to_id = {}

symbol_to_id.update(MARKERS)
symbol_to_id.update(PROLIF)

repair_mapping = {
    str(symbol): entrez_id
    for entrez_id, symbol in repair["symbol"].items()
    if pd.notna(symbol)
}

symbol_to_id.update(repair_mapping)


# Map string gene IDs to their actual index values.
# This handles cases where one table uses strings and
# another uses integers for Entrez Gene IDs.

index_lookup = {
    str(gene_id): gene_id
    for gene_id in logn.index
}


def resolve_gene_id(symbol_or_id):

    if symbol_or_id in symbol_to_id:
        gene_id = symbol_to_id[symbol_or_id]
    else:
        gene_id = symbol_or_id

    return index_lookup.get(
        str(gene_id),
        None
    )


def gene(sym_or_id):

    gene_id = resolve_gene_id(sym_or_id)

    if gene_id is None:
        raise KeyError(
            f"Gene {sym_or_id!r} was not found in the expression matrix."
        )

    return logn.loc[gene_id]


# Compare receptor-associated gene expression.

checks = [
    ("ESR1", "ER"),
    ("PGR", "PR"),
    ("ERBB2", "HER2")
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4)
)

rows = []

for ax, (symbol, receptor) in zip(axes, checks):

    try:
        expression = gene(symbol)

    except KeyError as error:
        print("Warning:", error)
        ax.set_visible(False)
        continue

    status = cohort[receptor].where(
        cohort[receptor].isin(["Positive", "Negative"])
    )

    df = pd.DataFrame({
        "expression": expression,
        "status": status
    }).dropna()

    positive = df.loc[
        df["status"] == "Positive",
        "expression"
    ]

    negative = df.loc[
        df["status"] == "Negative",
        "expression"
    ]

    if positive.empty or negative.empty:
        print(
            f"Warning: cannot compare {symbol}; "
            "one receptor-status group is empty."
        )
        ax.set_visible(False)
        continue

    sns.boxplot(
        data=df,
        x="status",
        y="expression",
        order=["Negative", "Positive"],
        ax=ax,
        fliersize=1
    )

    # AUC based on the Mann-Whitney U statistic.
    auc = stats.mannwhitneyu(
        positive,
        negative,
        alternative="two-sided"
    ).statistic / (
        len(positive) * len(negative)
    )

    rows.append({
        "gene": symbol,
        "receptor": receptor,
        "AUC": round(auc, 3),
        "median negative": round(negative.median(), 2),
        "median positive": round(positive.median(), 2)
    })

    ax.set(
        title=f"{symbol} by {receptor} status",
        xlabel="Clinical receptor status",
        ylabel="log2 normalised expression"
    )


# Y-chromosome gene check.

y_gene_symbols = [
    "RPS4Y1",
    "DDX3Y",
    "KDM5D"
]

available_y_genes = [
    symbol
    for symbol in y_gene_symbols
    if resolve_gene_id(symbol) is not None
]

if len(available_y_genes) == 0:

    axes[3].set_visible(False)

    print(
        "Warning: no Y-chromosome genes were found "
        "in MARKERS, PROLIF or repair."
    )

else:

    y_gene_ids = [
        resolve_gene_id(symbol)
        for symbol in available_y_genes
    ]

    missing_y_genes = [
        symbol
        for symbol in available_y_genes
        if resolve_gene_id(symbol) not in counts_pt.index
    ]

    available_y_genes = [
        symbol
        for symbol in available_y_genes
        if resolve_gene_id(symbol) in counts_pt.index
    ]

    y_gene_ids = [
        resolve_gene_id(symbol)
        for symbol in available_y_genes
    ]

    if len(y_gene_ids) == 0:

        axes[3].set_visible(False)

        print(
            "Warning: Y-chromosome genes were not present "
            "in the primary tumour count matrix."
        )

    else:

        # Calculate CPM for the primary tumour samples.
        primary_lib_size = counts_pt.sum(axis=0).astype(np.float64)

        cpm_pt = (
            counts_pt.astype(np.float32)
            .div(primary_lib_size.astype(np.float32), axis=1)
            * np.float32(1e6)
        )

        y_score = np.log2(
            cpm_pt.loc[y_gene_ids].astype(np.float32) + 1
        ).mean(axis=0)

        sex_df = pd.DataFrame({
            "Y-gene score": y_score,
            "sex": gp["SEX"].reindex(counts_pt.columns)
        }).dropna()

        if (
            (sex_df["sex"] == "Female").any()
            and (sex_df["sex"] == "Male").any()
        ):

            sns.boxplot(
                data=sex_df,
                x="sex",
                y="Y-gene score",
                order=["Female", "Male"],
                ax=axes[3],
                fliersize=0
            )

            sns.stripplot(
                data=sex_df,
                x="sex",
                y="Y-gene score",
                order=["Female", "Male"],
                ax=axes[3],
                size=2,
                color="black",
                alpha=0.4
            )

            axes[3].set(
                title="Y-chromosome gene expression",
                ylabel="Mean log2(CPM + 1)",
                xlabel="Recorded sex"
            )

        else:

            axes[3].set_visible(False)

            print(
                "Warning: male and female primary tumour samples "
                "are both required for the Y-gene comparison."
            )


plt.tight_layout()
plt.show()


print("\nRECEPTOR-EXPRESSION CHECKS")

if rows:
    display(pd.DataFrame(rows))
else:
    print("No receptor-expression comparisons were completed.")


# Compare male and female Y-gene scores.

if "sex_df" in globals():

    male_scores = sex_df.loc[
        sex_df["sex"] == "Male",
        "Y-gene score"
    ]

    female_scores = sex_df.loc[
        sex_df["sex"] == "Female",
        "Y-gene score"
    ]

    if male_scores.empty or female_scores.empty:

        print(
            "Male or female samples are missing "
            "from the Y-gene comparison."
        )

    else:

        male_min = male_scores.min()
        female_max = female_scores.max()

        print(
            f"Highest female Y-gene score: {female_max:.2f}"
        )

        print(
            f"Lowest male Y-gene score: {male_min:.2f}"
        )

        print(
            "Female samples in the male score range:",
            int((female_scores >= male_min).sum())
        )


# %% [markdown]
# ### 3.6.1 Investigation of ESR1-discordant TNBC tumours
#
# Identify TNBC tumours whose ESR1 expression is above
# the 10th percentile of ER-positive tumours.
#
# These samples are flagged for review. They are not
# automatically removed or reclassified.

# %%
# ============================================================
# 3.6.1 ESR1-DISCORDANT TNBC INVESTIGATION
# ============================================================

esr1 = gene("ESR1")
pgr = gene("PGR")

er_positive = cohort["ER"] == "Positive"

if not er_positive.any():
    raise ValueError(
        "No ER-positive tumours are available to calculate "
        "the ESR1 reference percentile."
    )

er_pos_q10 = esr1.loc[
    er_positive
].quantile(0.10)

discordant = cohort.index[
    (cohort["group"] == "TNBC")
    & (esr1 > er_pos_q10)
]

tnbc_count = int(y.sum())

discordant_fraction = (
    len(discordant) / tnbc_count
    if tnbc_count > 0
    else 0
)

print(
    "TNBC tumours with ESR1 above the 10th percentile "
    "of ER-positive tumours:"
)

print(
    f"{len(discordant)} of {tnbc_count} "
    f"({discordant_fraction:.1%})"
)


# Expression and assigned receptor labels.

esr1_qc = pd.DataFrame({
    "ESR1 expression": esr1.reindex(discordant),
    "PGR expression": pgr.reindex(discordant),
    "ER status used": cohort.loc[discordant, "ER"],
    "PR status used": cohort.loc[discordant, "PR"],
    "HER2 status used": cohort.loc[discordant, "HER2"]
})


# Original clinical receptor records.

clinical_records = clin.reindex(discordant)[[
    "ER_STATUS_BY_IHC",
    "PR_STATUS_BY_IHC",
    "IHC_HER2",
    "HER2_FISH_STATUS"
]].rename(columns={
    "ER_STATUS_BY_IHC": "Original ER IHC record",
    "PR_STATUS_BY_IHC": "Original PR IHC record",
    "IHC_HER2": "Original HER2 IHC record",
    "HER2_FISH_STATUS": "Original HER2 FISH record"
})

esr1_qc = esr1_qc.join(
    clinical_records
)

esr1_qc.index.name = "Patient ID"

print("\nESR1-DISCORDANT TNBC CASES")

if esr1_qc.empty:
    print("No ESR1-discordant TNBC cases were identified.")
else:
    display(esr1_qc.round(3))


# %% [markdown]
# ### 3.7 Stage 3 summary
#
# Summarise the actual outputs generated during preprocessing
# and quality control.

# %%
# ============================================================
# 3.7 STAGE 3 SUMMARY
# ============================================================

print("STAGE 3 SUMMARY")
print("=" * 50)


# 1. Integrity.

print("\n1. COUNT MATRIX INTEGRITY")

print("Missing values:", missing_values)
print("Negative counts:", negative_counts)
print("Non-integer counts:", non_integer_counts)
print("Duplicated gene IDs in original matrix:", duplicate_gene_ids)

print(
    "Unique genes after merging:",
    f"{counts_all.shape[0]:,}"
)


# 2. Cohort.

print("\n2. COHORT SELECTION")

display(
    pd.DataFrame(flow).set_index("step")
)

print("\nFinal group composition:")

display(
    cohort["group"].value_counts().to_frame("Patients")
)

print(f"Final cohort: {len(cohort):,} patients")

print(
    f"TNBC: {int(y.sum()):,} ({y.mean():.1%})"
)

print(
    f"non-TNBC: {int((y == 0).sum()):,} "
    f"({(y == 0).mean():.1%})"
)


# 3. Sequencing depth.

print("\n3. SEQUENCING DEPTH")

print(
    f"Median library size: "
    f"{lib_size.median()/1e6:.2f} million reads"
)

print(
    "Low-library samples flagged:",
    int(low_lib.sum())
)

print(
    f"Median detected genes: {detected.median():,.0f}"
)


# 4. Gene filtering.

print("\n4. GENE FILTERING")

print(f"Genes before filtering: {counts.shape[0]:,}")
print(f"Genes after 15% filtering: {counts_f.shape[0]:,}")

display(sensitivity_df)

print(
    f"DNA-repair genes retained: "
    f"{len(repair_present)} of {len(repair)}"
)


# 5. Normalisation.

print("\n5. NORMALISATION")

print(
    f"Median size factor: {size_factors.median():.3f}"
)

print(
    f"Size factor range: "
    f"{size_factors.min():.3f}–{size_factors.max():.3f}"
)

print(
    "Correlation with library size:",
    round(sf_lib_correlation, 3)
)


# 6. Biological checks.

print("\n6. BIOLOGICAL SANITY CHECKS")

if rows:
    display(pd.DataFrame(rows))
else:
    print("No receptor-expression comparisons were completed.")


# 7. ESR1-discordant cases.

print("\n7. ESR1-DISCORDANT TNBC CASES")

print(
    f"{len(discordant)} of {tnbc_count} TNBC tumours "
    f"({discordant_fraction:.1%})"
)

print(
    "These cases should be reviewed using their "
    "clinical receptor records."
)

print(
    "Do not remove or reclassify samples based on expression alone."
)


# Final output variables available to later stages.

print("\n8. OUTPUTS AVAILABLE FOR LATER STAGES")

stage3_outputs = [
    "counts_all",
    "counts_pt",
    "counts",
    "counts_f",
    "cohort",
    "y",
    "cpm",
    "size_factors",
    "normalised_counts",
    "logn",
    "sensitivity_df",
    "esr1_qc",
    "discordant"
]

for output in stage3_outputs:
    print(" -", output)


print("\nStage 3 preprocessing and quality control completed.")
