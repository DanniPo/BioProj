# Work plan: who does what

The analysis already runs end to end (the **baseline** commit). Each member now **leads one stage**. You are responsible for understanding it, checking it, improving it with a concrete task, keeping its interpretation correct, and presenting it. Your evidence is whatever you do from your own GitHub account: commits, a pull request (PR), a review of a teammate's PR, and the slides you present.

Swap roles freely. If you do, update `team.json` so the stage-lead names in the notebook stay accurate.

| Stage file | Stage lead | Reviews the PR of |
|---|---|---|
| `01_question_and_objectives.py`, `02_data_acquisition.py` | Wanyoike Samuel Macharia | Arop |
| `03_preprocessing_qc.py` | Arop Mary Abiong | Wanyoike |
| `04_eda.py`, `08_conclusions.py` | Dylan Saisi | Pete |
| `05_core_analysis_de.py` | Pete Njagi | Dylan |
| `06_biological_interpretation.py` | Kian Mwiberi | Danni |
| `07_machine_learning.py` + repo maintainer | Danni Podho | Kian |

## Stage tasks

Each lead does the **core task** at minimum. Stretch tasks are optional. Whatever you add, update the **Interpretation** text of your stage so it matches the new output.

### Stages 1–2 · Question and data: Wanyoike
- **Core:** replace the objectives in Stage 1 with the exact wording the lecturer approved. Run the notebook in Colab (the GDC provenance cell only works there), then add a sentence to the Stage 2 interpretation saying what the check printed.
- **Core:** check the "Description of every file and variable" table against the actual loaded data (row and column counts, variable names) and fix anything that is wrong.
- **Stretch:** add a short cell that lists the data release or date of each source (cBioPortal study metadata, GDC data release) so the data version is documented.
- **Present:** the research question, why it matters, where the data comes from, and why we use a pinned mirror plus hash checks.

### Stage 3 · Preprocessing and QC: Arop
- **Core:** sensitivity check on the gene filter. Re-run the filter at 10% and 25% of samples instead of 15%, and report how many genes and repair genes are kept each time. Does it change anything important?
- **Core:** look closer at the 5 TNBC tumours whose *ESR1* looks ER-positive. List their IDs, *ESR1*/*PGR* levels and receptor records, then write a short justification for keeping (or excluding) them.
- **Stretch:** check for outlier samples on the genome-wide PCA (e.g. more than 4 SD from the centre) and say whether any should be removed.
- **Present:** how the cohort went from 1,097 samples to 1,006, and the biological sanity checks.

### Stage 4 · EDA: Dylan
- **Core:** add one more informative plot, e.g. age distribution by group, or the top-loading genes on PC1 of the repair-gene PCA (which genes drive the axis?), and interpret it.
- **Stretch:** colour the genome-wide PCA by race or age as well, and say whether either drives the main axes.
- **Present:** what the PCA, heatmap and repair-vs-proliferation plot show.

### Stage 5 · Differential expression: Pete
- **Core:** robustness to age. TNBC patients are younger, so re-run DESeq2 with age as a covariate (`design="~AGE + group"`, dropping the patients with missing age). Compare the repair-gene log2 fold changes with the current ones (scatter plot + correlation).
- **Stretch:** report shrunken fold changes with PyDESeq2's `lfc_shrink()` for the repair-gene table.
- **Present:** the DE results, the gene-set test, and why proliferation adjustment changes the picture.

### Stage 6 · Biological interpretation: Kian
- **Core:** add a protein–protein interaction network from the STRING API (`https://string-db.org/api/tsv/network`, species 9606) for the significant repair genes. Build it in NetworkX and compare its hubs with the co-expression hubs.
- **Core:** check every reference in the literature table and the reference list (authors, year, journal), and add DOI links.
- **Stretch:** run GSEA with MSigDB Hallmark gene sets (download the GMT in Colab) to put the repair result in context.
- **Present:** the pathway enrichment, the network modules and the BRCA1 finding.

### Stage 7 · Machine learning: Danni (also maintainer)
- **Core:** replace the single 5-fold CV with repeated stratified CV (e.g. 5 × 5) for the baseline comparison, and report mean ± SD so "repair genes vs random genes" is judged with its uncertainty.
- **Core:** tune the random-forest decision threshold using out-of-fold predictions on the training set only, then re-evaluate it on the test set.
- **Stretch:** permutation importance on the test set for the logistic regression.
- **Present:** the model results against the baselines, and why "good AUC" does not mean "repair genes are special".

### Stage 8 · Conclusions: Dylan (with every lead)
- **Core:** after the other PRs are merged, each lead sends Dylan one updated sentence for their objective. Dylan updates Stage 8 so every number matches the final notebook.

### Maintainer jobs: Danni
- Create the GitHub repo, add everyone as collaborators, and protect `main` (require 1 approving review).
- Merge PRs. When everything is merged, run the notebook once in Colab with **Runtime → Run all** and commit the executed notebook.
- Run `python tools/contribution_report.py`, commit `CONTRIBUTIONS_LOG.md`, and tag the submission commit (`v1.0-submission`).

## Definition of done (each member)
- [ ] At least one PR **from your own GitHub account** that changes your stage file, reviewed and merged
- [ ] The notebook check (GitHub Actions) is green on your PR
- [ ] Your stage's Interpretation text matches the new output
- [ ] You reviewed your partner's PR on GitHub (comment or approval)
- [ ] `contributions/<your-first-name>.md` filled in with links
- [ ] Slides for your stage
- [ ] Your individual reflection (1-page PDF), written by you about what you actually did
