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
# ## Stage 2 · Data acquisition
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
# **Interpretation (Stage 2).** All four files downloaded and matched their pinned SHA-256 hashes, so these are exactly the inputs we analysed. The expression matrix covers **40,796 genes in 1,097 TCGA-BRCA tumour samples** and contains all 156 DNA-repair genes. The receptor-status table covers 1,097 patients. So the data can answer our question: it holds both the grouping variable (ER/PR/HER2) and genome-wide expression for the same patients. One weakness is visible already. Receptor status was scored by pathologists at many hospitals over several years, so some label noise is expected; we check this in Stage 3. When run in Colab, the GDC check above prints how many of our patients appear in GDC's own STAR-Counts listing. That confirms the mirror really carries GDC data.
