# Contribution: Dylan Saisi

## Stage(s) I led
Stage 4 · Exploratory data analysis; Stage 8 · Conclusions

## What I did
- **Stage 4 (core task):** added a plot of the top ±12 gene loadings on PC1 of the repair-gene PCA, oriented so the positive end is the high-proliferation / TNBC direction, plus an Interpretation bullet reading off both ends. It names the genes that drive PC1 and shows that the "repair" axis is really a proliferation axis, not a repair-specific signal. [PR #1](https://github.com/DanniPo/BioProj/pull/1), [commit a3793a8](https://github.com/DanniPo/BioProj/commit/a3793a8)
- **Stage 8 (core task):** after all stage PRs merged, I collected one updated sentence from each lead and reconciled the conclusions against the final executed notebook. I corrected the proliferation-adjustment counts (36 up / 71 down, 26% retained), dropped an unsupported "Y-gene sex" claim that the rewritten Stage 3 no longer produces, fixed the GSEA numbers (NES 1.80, FDR 0.010) and the proliferation-tracking range (ρ ≈ 0.84–0.91), and wove in the new Stage 6 STRING network and Stage 7 repeated-CV / threshold results. Only the prose changed; the auto-computing `summary` dict cell is untouched. [PR #8](https://github.com/DanniPo/BioProj/pull/8), [commit ebf2990](https://github.com/DanniPo/BioProj/commit/ebf2990)
- Reviewed Pete's Stage 5 PR (see "Review I gave").

## What changed in the results
- **Stage 4:** the repair-gene PC1 is now shown to be driven by replication-coupled HR / FA / checkpoint genes (*RAD51AP1, BLM, EXO1, FANCB, FANCI, FANCD2, CHEK1*), confirming it tracks the proliferation score (ρ = 0.81) rather than a repair-specific axis; the opposite end carries base-/nucleotide-excision repair and backup polymerases (*NEIL1, MPG, POLM, POLL, POLN, ERCC1, XPA*).
- **Stage 8:** every figure in the conclusions now matches the executed notebook — cohort 1,006 (157 TNBC / 849 non-TNBC), DE 79 up / 37 down, proliferation adjustment 36 / 71 (median 26% retained), HR GSEA NES 1.80 / FDR 0.010, STRING PPI 114 genes / 1,523 edges (hubs shared with co-expression: EXO1, FANCD2, RAD51), and repair-gene LR test ROC-AUC 0.93 vs random-gene 0.967.

## How I checked it
- Ran the full notebook in Colab (Runtime → Run all); prototyped the Stage 4 cell on the live cohort and read the gene lists off its printed output before committing.
- For Stage 8, cross-checked every number against the executed-notebook artifact from GitHub Actions on `main` (41 cells, 0 errors), using the Stage 8 `summary` dict output as the anchor.
- The "Notebook runs top to bottom" check passed on both PRs.

## Review I gave
- **Pete's Stage 5 PR (#4):** reviewed and commented. [review](https://github.com/DanniPo/BioProj/pull/4#pullrequestreview-5970058638) — All checks succesful. Ready to merge.

## Presentation
Slides 7, 8 and 18 — Stage 4 (genome-wide and repair-gene PCA, the PC1-loadings proliferation finding, the heatmap, repair-vs-proliferation) and Stage 8 (answer per objective, limitations).

## Evidence links
- Pull requests: Stage 4 https://github.com/DanniPo/BioProj/pull/1 ; Stage 8 https://github.com/DanniPo/BioProj/pull/8
- Commits: [a3793a8](https://github.com/DanniPo/BioProj/commit/a3793a8) Stage 4; [ebf2990](https://github.com/DanniPo/BioProj/commit/ebf2990) Stage 8
- Review: https://github.com/DanniPo/BioProj/pull/4#pullrequestreview-5970058638

