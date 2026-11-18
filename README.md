# DNA-repair gene expression in triple-negative breast cancer (TCGA-BRCA)

**Computing Bioinformatics · Group Project · ICS A · Group 2**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-GITHUB-ACCOUNT/YOUR-REPO/blob/main/TNBC_DNA_repair_TCGA.ipynb)
<!-- Replace YOUR-GITHUB-ACCOUNT/YOUR-REPO with this repository's path so the badge opens the notebook in Colab. -->

## Research question
Do triple-negative breast cancers (TNBC: ER−, PR−, HER2−) show distinct DNA-repair gene expression compared with non-TNBC breast cancers in TCGA-BRCA RNA-seq data?

## Team and roles
Each member leads one part of the analysis. Who did what, with links to commits, pull requests and reviews, is in **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)**. The commit-by-commit record is in `CONTRIBUTIONS_LOG.md`, generated from the git history.

| Member | Leads |
|---|---|
| Wanyoike Samuel Macharia | Stages 1–2 · Question, objectives and data acquisition |
| Arop Mary Abiong | Stage 3 · Preprocessing and quality control |
| Dylan Saisi | Stage 4 · Exploratory data analysis; Stage 8 · Conclusions |
| Pete Njagi | Stage 5 · Differential expression |
| Kian Mwiberi | Stage 6 · Biological interpretation |
| Danni Podho | Stage 7 · Machine learning; repository maintainer |

## How to run
Open `TNBC_DNA_repair_TCGA.ipynb` in Google Colab and choose **Runtime → Run all**. On the free CPU runtime it takes about 5–8 minutes. The notebook installs its two extra packages, downloads its own data (~140 MB) and writes result tables to `results/`.

## How the repository is organised
```
TNBC_DNA_repair_TCGA.ipynb   final notebook with outputs (generated, do not edit by hand)
TNBC_DNA_repair_TCGA.py      same notebook as a .py script (generated)
stages/                      one file per stage, each edited by its stage lead
  01_question_and_objectives.py ... 08_conclusions.py
build_notebook.py            assembles stages/ into the notebook (--execute runs it)
team.json                    who leads each stage (names appear under each stage heading)
results/                     tables written by the notebook
tools/contribution_report.py writes CONTRIBUTIONS_LOG.md from git history
WORKPLAN.md                  task for each member and definition of done
CONTRIBUTING.md              step-by-step git / pull-request workflow
CONTRIBUTIONS.md             who did what, with evidence links
.github/                     PR template, stage owners, and a check that runs the notebook on every PR
```

## Data
| Data | Original source | Downloaded from |
|---|---|---|
| RNA-seq raw read counts (STAR – Counts, GENCODE v36) for 1,097 TCGA-BRCA tumours | NCI Genomic Data Commons (GDC) | cBioPortal datahub mirror of the GDC data, pinned to commit `684be495`, each file checked against its SHA-256 hash |
| ER / PR / HER2 status (IHC, FISH) | TCGA Biospecimen Core Resource clinical records | cBioPortal *Breast Invasive Carcinoma (TCGA, Firehose Legacy)* |
| DNA-repair gene set (156 genes, 6 pathways) | MSigDB v7.5.1 (KEGG and Reactome) | embedded in the notebook |

All data is open-access and de-identified.

## Key results (baseline analysis)
* **Cohort:** 1,006 female primary tumours (157 TNBC, 849 non-TNBC).
* **Differential expression:** 116 of 148 expressed repair genes differ (FDR < 0.05; 79 up, 37 down in TNBC). As a set, repair genes are shifted up in TNBC more than random genes (p = 0.0003).
* **Proliferation explains most of it:** adjusting for proliferation cuts the up-regulated genes from 74 to 38. Lower-than-expected *BRCA1* in a subset of TNBC and a down-regulated *RAD50/MSH3/POLK* module remain.
* **Pathway enrichment:** homologous recombination is the only significantly enriched repair pathway (GSEA NES 1.81, FDR 0.009).
* **Classifier:** repair-gene logistic regression reaches test ROC-AUC 0.93, but random gene sets do as well, so repair genes are not uniquely informative.

## Reproducibility
Input files are pinned and hash-checked, random seeds are fixed (42), and `pydeseq2==0.6.0` and `gseapy==1.3.2` are pinned. A GitHub Actions check rebuilds the notebook from `stages/` and runs it top to bottom on every pull request.

## Use of AI tools
The first draft of the analysis notebook (the `Baseline` commit) was prepared with the help of an AI assistant (Claude). Each stage was then checked, extended and presented by its stage lead, as recorded in the commit history, the pull requests and [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
