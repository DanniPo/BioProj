<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/62b67d97-42f7-4881-a647-c438a2baab7e" /># Contribution: Wanyoike Samuel Macharia

## Stage(s) I led
Stages 1-2 · Question and objectives; Data acquisition

## What I did
- Ran the full notebook in Colab and checked the Stage 2 "Description of every file and variable" table against the loaded data: all row/column counts match (counts 40,796 × 1,097; gdc_patient 1,098 × 31; gdc_sample 1,102 × 10; bcr 1,097 × 111) and every variable named in the table exists. No corrections were needed. [PR #2](https://github.com/DanniPo/BioProj/pull/2)
- Ran the GDC provenance check and added its output to the Stage 2 Interpretation. [commit](https://github.com/DanniPo/BioProj/pull/2/changes/6a020c4e6822c4fbdeda3bdc9ad2676184a90ee2)
- Checked the Stage 1 objectives; the lecturer had no comments on them, so they were left unchanged. [PR #2](https://github.com/DanniPo/BioProj/pull/2)
-  Reviewed Arop's Stage 3 PR (#6) and requested changes because the notebook check was failing. [review](https://github.com/DanniPo/BioProj/pull/6#pullrequestreview-5973041660)

## What changed in the results
The Stage 2 Interpretation now reports what the GDC check printed: GDC lists 1,111 primary-tumour STAR-count files for 1,095 TCGA-BRCA patients, and all 1,090 distinct patients in our matrix (1,097 sample columns) are in that listing; five GDC patients are not in our matrix.

## How I checked it
Ran the notebook in Colab (Runtime > Run all) from main and compared the printed shapes and column names with the table. The GitHub Actions notebook check on the PR passed.

## Review I gave
I reviewed Arop's Stage 3 PR (#6) and requested changes. The notebook check was failing with a length-mismatch error (40,761 values vs an index of 16,697) in the Stage 5 Wilcoxon cell, which suggested the new filter-sensitivity code was overwriting a variable used later. I asked her to give the loop its own variable names and re-run. The PR was then updated (it now has 2 commits) and merged by Danni.

## Presentation
Slides for Stages 1-2: (Slide 1-5)

## Evidence links
- Pull request: https://github.com/DanniPo/BioProj/pull/2
- Commits: https://github.com/DanniPo/BioProj/pull/2/changes/6a020c4e6822c4fbdeda3bdc9ad2676184a90ee2
- Review: https://github.com/DanniPo/BioProj/pull/6
- CI run: https://github.com/DanniPo/BioProj/actions/runs/44343190329
