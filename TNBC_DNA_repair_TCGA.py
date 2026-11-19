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
# # DNA-repair gene expression in triple-negative breast cancer (TCGA-BRCA)
#
# **Computing Bioinformatics — Group Project · ICS A · Group 2**
# Wanyoike Samuel Macharia · Arop Mary Abiong · Dylan Saisi · Pete Njagi · Danni Podho · Kian Mwiberi
#
# This notebook runs top to bottom in Google Colab (*Runtime → Run all*, about 5–8 minutes on the free CPU runtime). It follows the eight stages of the project brief in order:
#
# 1. Question and objectives
# 2. Data acquisition
# 3. Preprocessing and quality control
# 4. Exploratory data analysis
# 5. Core analysis (differential expression)
# 6. Biological interpretation
# 7. Machine learning
# 8. Conclusions
#
# Every stage ends with a short **Interpretation** of our own results.

# %% [markdown]
# ## Stage 1 · Question and objectives
#
# *Stage lead: Wanyoike Samuel Macharia*
#
# **Research question.** Do triple-negative breast cancers (TNBC: ER−, PR−, HER2−) show distinct DNA-repair gene expression compared with non-TNBC breast cancers in TCGA-BRCA RNA-seq data?
#
# **Why it matters.** TNBC has no hormone-receptor or HER2 target, so treatment relies on chemotherapy. Many TNBCs resemble *BRCA1*-mutant tumours, a pattern called "BRCAness" (Turner et al., 2004; Foulkes et al., 2010). That resemblance is why DNA-damaging drugs (platinum) and PARP inhibitors are used in TNBC (Robson et al., 2017). If DNA-repair genes are expressed differently in TNBC, that matters for this treatment logic.
#
# **Objectives**
#
# 1. **Assemble the cohort:** define TNBC and non-TNBC tumours from clinical ER/PR/HER2 status and build a quality-controlled RNA-seq count matrix.
# 2. **Test differential expression:** test 156 curated DNA-repair genes (six pathways: HR, FA, MMR, NER, BER, NHEJ) for differential expression between TNBC and non-TNBC, with FDR correction, and test whether DNA-repair genes *as a group* differ more than other genes.
# 3. **Interpret biologically:** identify which repair pathways and gene modules drive the difference (pathway enrichment, co-expression network, literature), including whether the signal is independent of tumour proliferation.
# 4. **Classify:** build and evaluate a machine-learning classifier that separates TNBC from non-TNBC using DNA-repair gene expression, against proper baselines.
#
# **Hypotheses**
#
# * **H0:** DNA-repair gene expression does not differ between TNBC and non-TNBC.
# * **H1:** TNBC show a distinct DNA-repair expression profile. Based on BRCAness biology and the high proliferation of TNBC, we expect replication-coupled repair genes (HR, FA, MMR) to be **higher** in TNBC.

# %% [markdown]
# ### Setup
# Installs the two packages Colab does not ship with (`pydeseq2`, `gseapy`), imports libraries and fixes random seeds so results are reproducible.

# %%
import importlib, subprocess, sys

def ensure(pip_name, module_name=None):
    """Install a package with pip only if it is not already importable."""
    try:
        importlib.import_module(module_name or pip_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

# versions pinned to the ones this notebook was run with
for pip_name, module_name in [("pydeseq2==0.6.0", "pydeseq2"), ("gseapy==1.3.2", "gseapy"),
                              ("statsmodels", "statsmodels"), ("networkx", "networkx")]:
    ensure(pip_name, module_name)

import hashlib, json, os, time, urllib.parse, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from IPython.display import display

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")
GROUP_COLORS = {"TNBC": "#C0392B", "non-TNBC": "#2E86C1"}

IN_COLAB = "google.colab" in sys.modules
SAVE_TO_DRIVE = False   # set True in Colab to cache data and results in Google Drive (our cloud pipeline)

if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/TNBC_DNA_repair"
else:
    BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import sklearn, pydeseq2, gseapy, matplotlib
print("Running in Colab:", IN_COLAB, "| data folder:", os.path.abspath(DATA_DIR))
print({m.__name__: m.__version__ for m in [np, pd, sklearn, pydeseq2, gseapy, nx, matplotlib, sns]})

# %% [markdown]
# ## Stage 2 · Data acquisition
#
# *Stage lead: Wanyoike Samuel Macharia*
#
# ### Where the data comes from
#
# | Data | Original source | How we download it |
# |---|---|---|
# | RNA-seq read counts for TCGA-BRCA tumours | **NCI Genomic Data Commons (GDC)**, harmonised pipeline, workflow *STAR – Counts* (`*.rna_seq.augmented_star_gene_counts.tsv`, column `unstranded`, GENCODE v36 genes) | cBioPortal study *Invasive Breast Carcinoma (TCGA GDC, 2025)*, which collects the GDC files into one matrix (one primary aliquot per sample) |
# | Sex, age, race (patient level) | GDC clinical data | same cBioPortal GDC study |
# | ER, PR and HER2 status (IHC and FISH) | TCGA Biospecimen Core Resource clinical records (on GDC as the *BCR Biotab* clinical supplement) | cBioPortal study *Breast Invasive Carcinoma (TCGA, Firehose Legacy)* |
#
# **Why a mirror?** The GDC distributes one file per sample (~1,100 files, ~4 GB). cBioPortal's datahub on GitHub publishes the same GDC STAR counts as one 135 MB matrix. That is faster and more reliable inside a Colab session. We pin the download to a fixed datahub commit and check every file's SHA-256 hash, so anyone re-running this gets byte-identical inputs. A provenance check below queries the GDC API directly and confirms that our patients match GDC's own file listing.
#
# The data is **open-access and de-identified** (TCGA barcodes only, no names or contact details), published for secondary research. No controlled-access data is used.

# %%
DATAHUB_COMMIT = "684be495d96265c76491e1330294dfedba6dd6f4"   # cBioPortal/datahub, 17 Nov 2026
MIRROR = f"https://media.githubusercontent.com/media/cBioPortal/datahub/{DATAHUB_COMMIT}"

FILES = {
    # key: (path in datahub, local file name, expected sha256, expected size in bytes)
    "counts":     ("crdc/gdc/brca_tcga_gdc/data_mrna_seq_read_counts.txt", "gdc_star_unstranded_counts.txt",
                   "1bf6c3323766c334b3e08de1ab9972a2ea374c8682f4ebd705b9e0a12a8c3e51", 135327800),
    "gdc_patient": ("crdc/gdc/brca_tcga_gdc/data_clinical_patient.txt", "gdc_clinical_patient.txt",
                   "ed23eb74d829ceccb236c03fce6501909452da47fbf3b6f1e23fda4ba276bf66", 364401),
    "gdc_sample":  ("crdc/gdc/brca_tcga_gdc/data_clinical_sample.txt", "gdc_clinical_sample.txt",
                   "2486dcaf4bd3f0d3851e80ddef5b675bc47b27f298f7f59063a57408a6b25051", 172380),
    "receptors":   ("public/brca_tcga/data_clinical_patient.txt", "tcga_bcr_clinical_patient.txt",
                   "973b6e03436330e0563a15989d04fb3566a2c7cdc66db9aaa3badbb32c58badc", 1703839),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(key, retries=3):
    remote, local, expected_hash, expected_size = FILES[key]
    path = os.path.join(DATA_DIR, local)
    if not (os.path.exists(path) and os.path.getsize(path) == expected_size):
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                urllib.request.urlretrieve(f"{MIRROR}/{remote}", path)
                print(f"downloaded {local} ({os.path.getsize(path)/1e6:.1f} MB, {time.time()-t0:.0f}s)")
                break
            except Exception as err:
                print(f"attempt {attempt} failed for {local}: {err}")
                if attempt == retries:
                    raise
                time.sleep(5)
    ok = sha256(path) == expected_hash
    print(f"{local:36s} sha256 {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise ValueError(f"{local} does not match the pinned version - delete it and re-run.")
    return path

paths = {key: fetch(key) for key in FILES}

# %%
# Load the four files
counts_raw = pd.read_csv(paths["counts"], sep="\t", index_col=0)      # genes x samples
gdc_patient = pd.read_csv(paths["gdc_patient"], sep="\t", comment="#")
gdc_sample = pd.read_csv(paths["gdc_sample"], sep="\t", comment="#")
bcr = pd.read_csv(paths["receptors"], sep="\t", comment="#", low_memory=False)

print("counts      :", counts_raw.shape, "(rows = genes as Entrez Gene IDs, columns = tumour samples)")
print("gdc_patient :", gdc_patient.shape)
print("gdc_sample  :", gdc_sample.shape)
print("bcr (receptor status):", bcr.shape)
display(counts_raw.iloc[:5, :5])

# %% [markdown]
# ### Description of every file and the variables we use
#
# | File | Rows × columns | Variables used | Meaning |
# |---|---|---|---|
# | `gdc_star_unstranded_counts.txt` | 40,796 genes × 1,097 samples | index `Entrez_Gene_Id`; one column per sample barcode (`TCGA-XX-XXXX-01A`) | Raw integer read counts per gene from STAR alignment to GRCh38 (GDC *unstranded* column). Raw counts are what DESeq2 expects. |
# | `gdc_clinical_patient.txt` | 1,098 patients × 31 | `PATIENT_ID`, `SEX`, `AGE`, `RACE` | Patient-level GDC clinical fields. Used to exclude male patients and to describe the cohort. |
# | `gdc_clinical_sample.txt` | 1,102 samples × 10 | `SAMPLE_ID`, `SAMPLE_TYPE` | Sample type (Primary Tumor vs Metastatic). Barcode positions 14–15 encode the same (`01` = primary tumour, `06` = metastasis). |
# | `tcga_bcr_clinical_patient.txt` | 1,097 patients × 111 | `PATIENT_ID`, `ER_STATUS_BY_IHC`, `PR_STATUS_BY_IHC`, `IHC_HER2`, `HER2_FISH_STATUS` | Receptor status from pathology at the tissue source site. IHC = immunohistochemistry; FISH = fluorescence in-situ hybridisation for *ERBB2* amplification. |
#
# **Gene annotation.** The count matrix is keyed by Entrez Gene ID. The DNA-repair genes, proliferation genes and marker genes this analysis needs are listed below with their symbols and Entrez IDs, taken from MSigDB v7.5.1 (KEGG and Reactome collections).
#
# **DNA-repair gene set (156 genes).** We defined it before looking at any results:
#
# * **HR** (homologous recombination) = KEGG_HOMOLOGOUS_RECOMBINATION ∪ REACTOME_HDR_THROUGH_HOMOLOGOUS_RECOMBINATION_HRR. Reactome is added because KEGG's legacy HR map omits *BRCA1*, *BARD1*, *PALB2*, *BRIP1*, *ATM* and *CHEK1*.
# * **FA** = REACTOME_FANCONI_ANEMIA_PATHWAY
# * **MMR / NER / BER / NHEJ** = the KEGG mismatch-repair, nucleotide-excision-repair, base-excision-repair and non-homologous-end-joining sets
# * **Removed:** ubiquitin precursor genes (*UBA52, UBB, UBC, RPS27A*) and two *HMGB1* pseudogenes. They are pathway participants in the databases but are not repair-specific.
#
# A gene can belong to several pathways. Core replication factors (PCNA, RPA, RFC, POLD/POLE) appear in 3–4 pathways, so for plots we label them **"Shared"**.

# %%
REPAIR_TABLE = """
APEX1:328:BER  APEX2:27301:BER  ATM:472:HR  ATR:545:HR;FA
ATRIP:84126:HR;FA  BARD1:580:HR  BLM:641:HR  BRCA1:672:HR
BRCA2:675:HR  BRIP1:83990:HR  CCNH:902:NER  CDK7:1022:NER
CENPS:378708:FA  CENPX:201254:FA  CETN2:1069:NER  CHEK1:1111:HR
CUL4A:8451:NER  CUL4B:8450:NER  DCLRE1A:9937:FA  DCLRE1B:64858:FA
DCLRE1C:64421:NHEJ  DDB1:1642:NER  DDB2:1643:NER  DNA2:1763:HR
DNTT:1791:NHEJ  EME1:146956:HR;FA  EME2:197342:HR;FA  ERCC1:2067:FA;NER
ERCC2:2068:NER  ERCC3:2071:NER  ERCC4:2072:FA;NER  ERCC5:2073:NER
ERCC6:2074:NER  ERCC8:1161:NER  EXO1:9156:HR;MMR  FAAP100:80233:FA
FAAP20:199990:FA  FAAP24:91442:FA  FAN1:22909:FA  FANCA:2175:FA
FANCB:2187:FA  FANCC:2176:FA  FANCD2:2177:FA  FANCE:2178:FA
FANCF:2188:FA  FANCG:2189:FA  FANCI:55215:FA  FANCL:55120:FA
FANCM:57697:FA  FEN1:2237:BER;NHEJ  GEN1:348654:HR  GTF2H1:2965:NER
GTF2H2:2966:NER  GTF2H3:2967:NER  GTF2H4:2968:NER  GTF2H5:404672:NER
HMGB1:3146:BER  HUS1:3364:HR  KAT5:10524:HR  LIG1:3978:MMR;NER;BER
LIG3:3980:BER  LIG4:3981:NHEJ  MBD4:8930:BER  MLH1:4292:MMR
MLH3:27030:MMR  MNAT1:4331:NER  MPG:4350:BER  MRE11:4361:HR;NHEJ
MSH2:4436:MMR  MSH3:4437:MMR  MSH6:2956:MMR  MUS81:80198:HR;FA
MUTYH:4595:BER  NBN:4683:HR  NEIL1:79661:BER  NEIL2:252969:BER
NEIL3:55247:BER  NHEJ1:79840:NHEJ  NTHL1:4913:BER  OGG1:4968:BER
PALB2:79728:HR  PARP1:142:BER  PARP2:10038:BER  PARP3:10039:BER
PARP4:143:BER  PCNA:5111:HR;MMR;NER;BER  PMS2:5395:MMR  POLB:5423:BER
POLD1:5424:HR;MMR;NER;BER  POLD2:5425:HR;MMR;NER;BER  POLD3:10714:HR;MMR;NER;BER  POLD4:57804:HR;MMR;NER;BER
POLE:5426:HR;NER;BER  POLE2:5427:HR;NER;BER  POLE3:54107:HR;NER;BER  POLE4:56655:HR;NER;BER
POLH:5429:HR  POLK:51426:HR  POLL:27343:BER;NHEJ  POLM:27434:NHEJ
POLN:353497:FA  PRKDC:5591:NHEJ  RAD1:5810:HR  RAD17:5884:HR
RAD23A:5886:NER  RAD23B:5887:NER  RAD50:10111:HR;NHEJ  RAD51:5888:HR
RAD51AP1:10635:HR  RAD51B:5890:HR  RAD51C:5889:HR  RAD51D:5892:HR
RAD52:5893:HR  RAD54B:25788:HR  RAD54L:8438:HR  RAD9A:5883:HR
RAD9B:144715:HR  RBBP8:5932:HR  RBX1:9978:NER  RFC1:5981:HR;MMR;NER
RFC2:5982:HR;MMR;NER  RFC3:5983:HR;MMR;NER  RFC4:5984:HR;MMR;NER  RFC5:5985:HR;MMR;NER
RHNO1:83695:HR  RMI1:80010:HR  RMI2:116028:HR  RPA1:6117:HR;FA;MMR;NER
RPA2:6118:HR;FA;MMR;NER  RPA3:6119:HR;FA;MMR;NER  RPA4:29935:HR;MMR;NER  RTEL1:51750:HR
SEM1:7979:HR  SLX1A:548593:HR;FA  SLX1B:79008:HR;FA  SLX4:84464:HR;FA
SMUG1:23583:BER  SPIDR:23514:HR  SSBP1:6742:HR;MMR  TDG:6996:BER
TOP3A:7156:HR  TOP3B:8940:HR  TOPBP1:11073:HR  UBE2T:29089:FA
UNG:7374:BER  USP1:7398:FA  WDR48:57599:FA  WRN:7486:HR
XPA:7507:NER  XPC:7508:NER  XRCC1:7515:BER  XRCC2:7516:HR
XRCC3:7517:HR  XRCC4:7518:NHEJ  XRCC5:7520:NHEJ  XRCC6:2547:NHEJ
"""
repair = pd.DataFrame([tok.split(":") for tok in REPAIR_TABLE.split()], columns=["symbol", "entrez", "pathways"])
repair["entrez"] = repair["entrez"].astype(int)
repair["n_pathways"] = repair["pathways"].str.count(";") + 1
repair["label"] = np.where(repair["n_pathways"] >= 3, "Shared", repair["pathways"].str.split(";").str[0])
repair = repair.set_index("entrez")
PATHWAYS = ["HR", "FA", "MMR", "NER", "BER", "NHEJ"]

# 11 proliferation genes of the PAM50 proliferation score (Nielsen et al. 2010); none is in the repair set
PROLIF = {"BIRC5": 332, "CCNB1": 891, "CDC20": 991, "NUF2": 83540, "CEP55": 55165, "NDC80": 10403,
          "MKI67": 4288, "PTTG1": 9232, "RRM2": 6241, "TYMS": 7298, "UBE2C": 11065}
# marker genes used for biological sanity checks
MARKERS = {"ESR1": 2099, "PGR": 5241, "ERBB2": 2064, "FOXA1": 3169, "GATA3": 2625, "KRT5": 3852,
           "FOXC1": 2296, "XIST": 7503, "RPS4Y1": 6192, "DDX3Y": 8653, "KDM5D": 8284}
assert not set(PROLIF.values()) & set(repair.index), "proliferation genes must not overlap repair genes"

SYMBOL = {**repair["symbol"].to_dict(), **{v: k for k, v in PROLIF.items()}, **{v: k for k, v in MARKERS.items()}}
print(f"DNA-repair genes: {len(repair)}")
print("genes per pathway:", {p: int(repair['pathways'].str.contains(rf'\b{p}\b').sum()) for p in PATHWAYS})
print("all repair genes present in the count matrix:", repair.index.isin(counts_raw.index).all())

# %% [markdown]
# ### Provenance check against the GDC API
# This cell asks the GDC API for its list of open-access TCGA-BRCA *STAR – Counts* files from primary tumours and compares the patients with our matrix. It needs internet access to `api.gdc.cancer.gov` (Colab has it). It is skipped if GDC cannot be reached, and nothing later depends on it.

# %%
def gdc_star_counts_cases():
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
    params = {"filters": json.dumps(filters), "fields": "file_id,cases.submitter_id",
              "format": "JSON", "size": "5000"}
    url = "https://api.gdc.cancer.gov/files?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        hits = json.load(response)["data"]["hits"]
    return len(hits), {c["submitter_id"] for h in hits for c in h.get("cases", [])}

try:
    n_files, gdc_cases = gdc_star_counts_cases()
    ours = set(counts_raw.columns.str[:12])
    print(f"GDC lists {n_files} primary-tumour STAR-count files for {len(gdc_cases)} TCGA-BRCA patients.")
    print(f"Our matrix: {len(ours)} patients, {len(ours & gdc_cases)} of them in GDC's listing.")
except Exception as err:
    print(f"GDC API not reachable from this machine ({type(err).__name__}); provenance check skipped.")

# %% [markdown]
# **Interpretation (Stage 2).** All four files downloaded and matched their pinned SHA-256 hashes, so these are exactly the inputs we analysed. The expression matrix covers **40,796 genes in 1,097 TCGA-BRCA tumour samples** and contains all 156 DNA-repair genes. The receptor-status table covers 1,097 patients. So the data can answer our question: it holds both the grouping variable (ER/PR/HER2) and genome-wide expression for the same patients. One weakness is visible already. Receptor status was scored by pathologists at many hospitals over several years, so some label noise is expected; we check this in Stage 3. When run in Colab, the GDC check listed 1,111 primary-tumour STAR-count files for 1,095 TCGA-BRCA patients, and all 1,090 distinct patients in our matrix (1,097 sample columns) were in that listing. That confirms the mirror really carries GDC data; five GDC patients are not in our matrix.

# %% [markdown]
# ## Stage 3 · Preprocessing and quality control
#
# *Stage lead: Arop Mary Abiong*

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

# %% [markdown]
# ## Stage 4 · Exploratory data analysis
#
# *Stage lead: Dylan Saisi*

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

# %% [markdown]
# ## Stage 5 · Core analysis: differential expression
#
# *Stage lead: Pete Njagi*
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
# ### 5.5 Robustness to age
#
# TNBC patients are younger than non-TNBC patients (median 54 vs 60, Stage 4), and age itself changes gene expression. So part of a "TNBC difference" could be an age difference. We re-run DESeq2 with age as a covariate, `design="~AGE + group"`, on the patients whose age is recorded. Age is standardised (mean 0, SD 1) so the model fits well; this does not change the TNBC coefficient.
#
# Dropping patients also changes the estimates a little, so we fit a third model, `~group` on the **same** patients. Comparing the three separates the effect of the smaller sample from the effect of adjusting for age.

# %%
age = pd.to_numeric(cohort["AGE"], errors="coerce").reindex(counts_f.columns)
has_age = age.notna().values
print(f"{int((~has_age).sum())} patients without a recorded age are dropped; "
      f"{int(has_age.sum())} remain ({int(y.values[has_age].sum())} TNBC, {int((1 - y.values)[has_age].sum())} non-TNBC)")
meta_age = meta.loc[has_age].copy()
meta_age["AGE"] = ((age[has_age] - age[has_age].mean()) / age[has_age].std()).values
counts_age = counts_f.loc[:, has_age].T.astype(int)

def run_deseq(design):
    """Fit DESeq2 on the patients with age and return the TNBC-vs-non-TNBC results table."""
    d = DeseqDataSet(counts=counts_age, metadata=meta_age, design=design,
                     refit_cooks=True, inference=inference, quiet=True)
    d.deseq2()
    s = DeseqStats(d, contrast=["group", "TNBC", "nonTNBC"], inference=inference, quiet=True)
    s.summary()
    r = s.results_df.copy()
    r.index = r.index.astype(int)
    return d, r

t0 = time.time()
dds_age, de_age = run_deseq("~AGE + group")          # age-adjusted
_, de_same = run_deseq("~group")                     # unadjusted, same patients
print(f"two extra DESeq2 fits finished in {time.time()-t0:.0f}s")

# Age effect itself (natural-log LFC per SD of age -> log2 per 10 years)
age_lfc = dds_age.varm["LFC"]["AGE"] / np.log(2) / age[has_age].std() * 10
age_lfc.index = age_lfc.index.astype(int)            # DESeq2 stores gene IDs as text
print(f"median |age effect| on repair genes: {age_lfc.reindex(counts_f.index).loc[repair_present].abs().median():.3f} "
      f"log2 per 10 years (TNBC effect: {rep.log2FoldChange.abs().median():.3f})")

cmp_age = pd.DataFrame({
    "symbol": rep.symbol,
    "lfc_original": rep.log2FoldChange,
    "lfc_same_patients": de_same.loc[rep.index, "log2FoldChange"],
    "lfc_age_adjusted": de_age.loc[rep.index, "log2FoldChange"],
    "padj_age_adjusted": de_age.loc[rep.index, "padj"],
})
cmp_age["padj_repair_family_age"] = multipletests(de_age.loc[rep.index, "pvalue"], method="fdr_bh")[1]
cmp_age["call_age_adjusted"] = [call(pd.Series({"padj_repair_family": p, "log2FoldChange": l}))
                                for p, l in zip(cmp_age.padj_repair_family_age, cmp_age.lfc_age_adjusted)]
cmp_age["call_original"] = rep["call"]
cmp_age["change"] = cmp_age.lfc_age_adjusted - cmp_age.lfc_original

r_p = stats.pearsonr(cmp_age.lfc_original, cmp_age.lfc_age_adjusted)[0]
r_s = stats.spearmanr(cmp_age.lfc_original, cmp_age.lfc_age_adjusted)[0]
r_sub = stats.pearsonr(cmp_age.lfc_same_patients, cmp_age.lfc_age_adjusted)[0]
sig0 = rep.padj_repair_family < 0.05
sig1 = cmp_age.padj_repair_family_age < 0.05
print(f"\nRepair-gene log2FC, original vs age-adjusted: Pearson r = {r_p:.3f}, Spearman ρ = {r_s:.3f}")
print(f"same patients without vs with age: Pearson r = {r_sub:.3f}")
print(f"median |change| in log2FC: {cmp_age.change.abs().median():.3f}; largest: {cmp_age.change.abs().max():.3f} "
      f"({cmp_age.loc[cmp_age.change.abs().idxmax(), 'symbol']})")
print(f"significant (repair-family FDR < 0.05): {int(sig0.sum())} originally, {int(sig1.sum())} age-adjusted; "
      f"same call for {(sig0 == sig1).mean():.0%} of genes; same direction for "
      f"{(np.sign(cmp_age.lfc_original) == np.sign(cmp_age.lfc_age_adjusted)).mean():.0%}")
print(f"up / down after age adjustment: {int((sig1 & (cmp_age.lfc_age_adjusted > 0)).sum())} / "
      f"{int((sig1 & (cmp_age.lfc_age_adjusted < 0)).sum())} (originally {int((sig0 & (rep.log2FoldChange > 0)).sum())} / "
      f"{int((sig0 & (rep.log2FoldChange < 0)).sum())})")
TWOFOLD = ["UP ≥2x in TNBC", "DOWN ≥2x in TNBC"]
print("≥2-fold calls, original:", ", ".join(cmp_age.symbol[cmp_age.call_original.isin(TWOFOLD)]))
print("≥2-fold calls, age-adjusted:", ", ".join(cmp_age.symbol[cmp_age.call_age_adjusted.isin(TWOFOLD)]))
changed = cmp_age[cmp_age.call_original != cmp_age.call_age_adjusted]
print(f"\n{len(changed)} repair genes change their call after age adjustment:")
display(changed[["symbol", "lfc_original", "lfc_age_adjusted", "call_original", "call_age_adjusted"]].round(3))
print("Largest shifts in log2FC:")
display(cmp_age.reindex(cmp_age.change.abs().sort_values(ascending=False).index).head(8)[
    ["symbol", "lfc_original", "lfc_same_patients", "lfc_age_adjusted", "change"]].round(3))
cmp_age.to_csv(os.path.join(RESULTS_DIR, "dna_repair_age_adjusted.csv"))

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(cmp_age.lfc_original, cmp_age.lfc_age_adjusted, s=20, c=np.where(sig1, "#C0392B", "#AAB7B8"),
           edgecolor="k", lw=0.3)
lim = [cmp_age[["lfc_original", "lfc_age_adjusted"]].min().min() - 0.1,
       cmp_age[["lfc_original", "lfc_age_adjusted"]].max().max() + 0.1]
ax.plot(lim, lim, ls="--", c="grey", lw=0.8); ax.axhline(0, c="k", lw=0.6); ax.axvline(0, c="k", lw=0.6)
for eid in cmp_age.change.abs().sort_values().index[-6:]:
    ax.annotate(cmp_age.symbol[eid], (cmp_age.lfc_original[eid], cmp_age.lfc_age_adjusted[eid]),
                fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.scatter([], [], c="#C0392B", s=20, label="FDR < 0.05 after age adjustment")
ax.scatter([], [], c="#AAB7B8", s=20, label="not significant")
ax.set(xlabel="log2FC, original model (~group)", ylabel="log2FC, age-adjusted model (~AGE + group)",
       title=f"Repair-gene fold changes with and without age\nPearson r = {r_p:.3f}")
ax.legend(fontsize=8, loc="upper left"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 5.6 Shrunken fold changes
#
# Raw DESeq2 fold changes are noisy for genes with low counts or high dispersion. `lfc_shrink()` (apeGLM prior) pulls those estimates towards zero while leaving well-measured genes almost unchanged. This makes the ranking by effect size more reliable. We use a separate `DeseqStats` object so the original results used later in the notebook stay untouched.

# %%
t0 = time.time()
ds_shr = DeseqStats(dds, contrast=["group", "TNBC", "nonTNBC"], inference=inference, quiet=True)
ds_shr.summary()
# The design's reference level is "TNBC" (alphabetical), so the model coefficient is group[T.nonTNBC].
# lfc_shrink() writes that raw coefficient into results_df, so we flip its sign back to TNBC vs non-TNBC.
LFC_COEF = [c for c in dds.varm["LFC"].columns if c.startswith("group")][0]
ds_shr.lfc_shrink(coeff=LFC_COEF)
shr = ds_shr.results_df.copy()
shr.index = shr.index.astype(int)
if LFC_COEF.endswith("nonTNBC]"):
    shr["log2FoldChange"] = -shr["log2FoldChange"]
assert stats.spearmanr(shr.loc[rep.index, "log2FoldChange"], rep.log2FoldChange)[0] > 0.9, "shrunken LFC sign check failed"
print(f"LFC shrinkage finished in {time.time()-t0:.0f}s")

rep_shr = rep[["symbol", "pathways", "baseMean", "log2FoldChange", "lfcSE", "padj_repair_family", "call"]].copy()
rep_shr["log2FC_shrunk"] = shr.loc[rep.index, "log2FoldChange"]
rep_shr["lfcSE_shrunk"] = shr.loc[rep.index, "lfcSE"]
rep_shr["shrink"] = rep_shr.log2FC_shrunk - rep_shr.log2FoldChange
print(f"Repair genes, raw vs shrunken log2FC: Pearson r = {stats.pearsonr(rep_shr.log2FoldChange, rep_shr.log2FC_shrunk)[0]:.4f}; "
      f"median |shrinkage| = {rep_shr.shrink.abs().median():.4f}; largest = {rep_shr.shrink.abs().max():.3f} "
      f"({rep_shr.loc[rep_shr.shrink.abs().idxmax(), 'symbol']})")
print(f"genes with |shrunken log2FC| ≥ 1 and repair-family FDR < 0.05: "
      f"{int(((rep_shr.log2FC_shrunk.abs() >= 1) & (rep_shr.padj_repair_family < 0.05)).sum())} "
      f"(raw: {int(rep_shr.call.isin(TWOFOLD).sum())})")
print("\nRepair genes ranked by shrunken log2FC (top 12 up, top 5 down):")
rep_shr = rep_shr.sort_values("log2FC_shrunk", ascending=False)
display(rep_shr.head(12).round(3))
display(rep_shr.tail(5).iloc[::-1].round(3))
print("Genes most affected by shrinkage:")
display(rep_shr.reindex(rep_shr.shrink.abs().sort_values(ascending=False).index).head(5).round(3))
rep_shr.to_csv(os.path.join(RESULTS_DIR, "dna_repair_DE_shrunken.csv"))

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
# * **Age (robustness check):** TNBC patients are younger, so we re-ran DESeq2 with age as a covariate (`~AGE + group`) on the 992 patients with a recorded age (14 dropped; 157 TNBC, 835 non-TNBC). The repair-gene fold changes are practically identical to the original ones (Pearson r = 1.000, Spearman ρ = 0.999). The median change is 0.009 log2 units and the largest is 0.072 (*NTHL1*, a gene with almost no TNBC effect). 115 instead of 116 repair genes stay significant (79 up, 36 down). Only *RAD51C* changes its call, from slightly down (log2FC −0.16) to not significant (−0.14). The same eleven genes change at least two-fold. Fitting `~group` on the same 992 patients gives the same fold changes too (r = 1.000), so neither dropping the patients without age nor adjusting for age changes anything. **The TNBC repair-gene differences are not an age effect.** The model only adjusts for a linear age trend, which is the usual assumption.
# * **Shrunken fold changes:** apeGLM shrinkage hardly moves any repair-gene estimate (r = 1.0000 with the raw values; median shift 0.002, largest 0.012 for *POLN*). The ranking and the eleven ≥2-fold genes stay the same, led by *FANCA* (1.37), *RAD54L* (1.36) and *CHEK1* (1.28), with *POLN* (−1.14) the only one down. This is expected: with ~1,000 tumours and well-expressed genes the standard errors are small (lfcSE 0.06–0.10), so the prior has little to correct. The raw fold changes reported above can be trusted as effect sizes.

# %% [markdown]
# ## Stage 6 · Biological interpretation
#
# *Stage lead: Kian Mwiberi*
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

# %% [markdown]
# ## Stage 7 · Machine learning
#
# *Stage lead: Danni Podho*
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
# * **Model selection (training set, 5-fold CV):** logistic regression with strong regularisation (C = 0.01) did best (CV ROC-AUC 0.970), slightly ahead of the random forest (0.951).
# * **Against the baselines:** the repair-gene model clearly beats the majority-class baseline (AUC 0.50) and the proliferation-only model (CV 0.84, test 0.83). So repair genes carry information beyond "this tumour divides fast", and adding the proliferation score to them adds nothing. **But 30 random sets of 148 non-repair genes do just as well.**
# * **Repeated cross-validation (5 × 5, same splits for every model):** this confirms the point above and makes it sharper. The repair-gene model scores 0.969 ± 0.013 and the random gene sets 0.967 ± 0.013. Scored on identical folds, the repair genes are ahead by only +0.002 AUC (SD 0.008) and win in just 60% of the comparisons. They beat 77% of the random-set means, down from 93% with the single 5-fold split, so the single split flattered the repair genes. The proliferation-only model is not only weaker (0.843) but also the least stable (SD 0.045, range 0.74–0.91).
# * **Test set (used once):** the repair-gene logistic regression reaches **ROC-AUC 0.93 (95% CI 0.87–0.97)** and PR-AUC 0.66 (chance level 0.16). Balanced accuracy is 0.87, with sensitivity 0.81 (26 of 32 TNBC found) and specificity 0.93 (158 of 170 non-TNBC correct). The test AUC is a little lower than CV (0.93 vs 0.97); with only 32 TNBC in the test set this is within the uncertainty, and nothing was tuned on the test set.
# * **Random forest threshold:** at the default 0.5 threshold the random forest finds only 14 of 32 TNBC, because its probabilities for the rare class are low. A threshold of 0.25, chosen on training out-of-fold predictions only, raises test sensitivity from 44% to 75% (24 of 32) and balanced accuracy from 0.70 to 0.84. Specificity drops slightly (0.96 → 0.92), and ROC-AUC is unchanged (0.915) because it does not depend on the threshold. Even tuned, the random forest stays below the logistic regression (balanced accuracy 0.84 vs 0.87), so logistic regression remains our preferred model.
# * **What the model uses:** the largest coefficients are *FANCE, SEM1, RAD51B, GTF2H1, ATM* (towards TNBC) and *ATRIP, RAD50, RAD17, POLL, POLD4, POLN* (towards non-TNBC). The coefficients correlate with the proliferation-adjusted differences from Stage 5 (ρ = 0.84). The model relies on the proliferation-independent part of the repair profile, the same genes our statistics highlighted.
# * **Permutation importance:** no single gene matters much. Shuffling the most important gene, *ATM*, lowers test AUC by only 0.006, and only 67 of the 148 genes have any measurable effect. 9 of the top 15 genes (*ATM, DCLRE1C, FANCA, PARP4, POLD4, POLL, POLN, RAD51B, XPA*) are also among the top 20 coefficients, so the two methods broadly agree. The information is spread across many correlated genes, which is also why random gene sets can do the same job.
# * **Honest conclusion:** DNA-repair expression classifies TNBC well, but that does **not** show repair genes are special. TNBC are so different transcriptome-wide that almost any 148 genes would do. The classifier confirms a distinct profile; it is not evidence of a repair-specific biomarker.

# %% [markdown]
# ## Stage 8 · Conclusions
#
# *Stage lead: Dylan Saisi*

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
# 116 of 148 repair genes differ at FDR < 0.05, and the set as a whole is shifted towards higher expression in TNBC more than random genes (p = 0.0003), so we reject H0. The predicted direction (HR/FA/MMR up in TNBC) holds, but adjusting for proliferation removes most of it: only 38 genes stay higher, and more genes (67) become *lower* than proliferation predicts. The proliferation-independent signal is:
#
# * FA-core and checkpoint genes (*FANCE, FANCA, USP1, CHEK1*) and *RAD51B* stay higher in TNBC.
# * ***BRCA1***, a *RAD50/MSH3/POLK/RAD17/FAN1* module and *POLN* are lower than expected.
#
# ### One point per objective
# 1. **Cohort (O1):** we built a quality-controlled count matrix for 1,006 female primary tumours (157 TNBC, 849 non-TNBC). The labels passed biological checks (*ESR1* AUC 0.97, *PGR* 0.93, recorded sex consistent with Y-gene expression). Only 5 TNBC (3.2%) look ER-positive at the mRNA level.
# 2. **Differential expression (O2):** 79 repair genes are up and 37 down in TNBC (FDR < 0.05), and 11 change at least two-fold (10 up, led by *FANCA, RAD54L, CHEK1*; 1 down, *POLN*). The Wilcoxon test confirms the direction for 94% of genes. As a set, repair genes are shifted *up* in TNBC, but they are not more strongly changed than random genes.
# 3. **Biological interpretation (O3):** homologous recombination is the only significantly enriched pathway (GSEA NES 1.81, FDR 0.009). The up-regulated genes form one co-expressed, replication-coupled HR/FA module that tracks proliferation (ρ ≈ 0.85–0.88). Two findings do not depend on proliferation: *BRCA1* is lower than expected in TNBC, with a low tail in 16.6% of tumours (consistent with BRCAness), and a down-regulated *RAD50/MSH3/POLK* module.
# 4. **Machine learning (O4):** a logistic regression on repair genes reaches test ROC-AUC 0.93 (95% CI 0.87–0.97), balanced accuracy 0.87, sensitivity 0.81 and specificity 0.93, clearly beating the majority-class (0.50) and proliferation-only (0.83) baselines. Random gene sets of the same size perform just as well (CV AUC 0.966 vs 0.970). So the classifier confirms a distinct profile but **does not show that repair genes are uniquely informative**.
#
# ### Limitations
# * **Expression ≠ function.** High HR-gene mRNA does not mean HR works, and low *BRCA1* mRNA does not prove deficiency. We used no mutation, methylation or genomic-scar data.
# * **Proliferation adjustment is imperfect.** The 11-gene score is a proxy, so some confounding may remain. Adjustment may also remove real biology if repair changes themselves drive proliferation.
# * **Bulk tumour RNA.** Each sample mixes tumour, stromal and immune cells. We did not adjust for tumour purity, and TNBC often have more immune infiltration.
# * **Label noise.** Receptor status was scored at many hospitals over many years under older guidelines. 71 tumours with incomplete status were excluded, and 5 TNBC have *ESR1* at ER-positive levels. Two housekeeping repair genes (*GTF2H4, TOP3B*) had implausibly low counts, a likely annotation artefact.
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
# This notebook runs top to bottom in Google Colab. It is stored on GitHub with its outputs, together with the result tables written to `results/` (repair-gene DE table, proliferation-adjusted table, GSEA table, ML test metrics).

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
