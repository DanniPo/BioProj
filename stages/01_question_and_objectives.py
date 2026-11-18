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
