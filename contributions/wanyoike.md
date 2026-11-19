# Contribution: Wanyoike Samuel Macharia

## Stage(s) I led
Stages 1-2 · Question and objectives; Data acquisition

## What I did
- Ran the full notebook in Colab and checked the Stage 2 "Description of every file and variable" table against the loaded data: all row/column counts match (counts 40,796 × 1,097; gdc_patient 1,098 × 31; gdc_sample 1,102 × 10; bcr 1,097 × 111) and every variable named in the table exists. No corrections were needed. [PR #2](https://github.com/DanniPo/BioProj/pull/2)
- Ran the GDC provenance check and added its output to the Stage 2 Interpretation. [commit](https://github.com/DanniPo/BioProj/pull/2/changes/6a020c4e6822c4fbdeda3bdc9ad2676184a90ee2)
- Checked the Stage 1 objectives; the lecturer had no comments on them, so they were left unchanged. [PR #2](https://github.com/DanniPo/BioProj/pull/2)

## What changed in the results
The Stage 2 Interpretation now reports what the GDC check printed: GDC lists 1,111 primary-tumour STAR-count files for 1,095 TCGA-BRCA patients, and all 1,090 distinct patients in our matrix (1,097 sample columns) are in that listing; five GDC patients are not in our matrix.

## How I checked it
Ran the notebook in Colab (Runtime > Run all) from main and compared the printed shapes and column names with the table. The GitHub Actions notebook check on the PR passed.

## Review I gave
(to be added after I review Arop's Stage 3 PR)

## Presentation
Slides for Stages 1-2: (slide numbers to be confirmed)

## Evidence links
- Pull request: https://github.com/DanniPo/BioProj/pull/2
- Commits: https://github.com/DanniPo/BioProj/pull/2/changes/6a020c4e6822c4fbdeda3bdc9ad2676184a90ee2
- Review: (to be added)
- CI run: https://github.com/DanniPo/BioProj/actions/runs/44343190329
