# Contribution: Pete Njagi

## Stage(s) I led
Stage 5 · Core analysis: differential expression

## What I did
- Added an age-robustness check (core task). TNBC patients are younger, so I re-ran PyDESeq2 with `design="~AGE + group"` on the 992 patients with a recorded age (age standardised). I also fitted `~group` on the same 992 patients, to separate the effect of dropping 14 patients from the effect of adjusting for age. The check compares the repair-gene log2 fold changes with a scatter plot and correlations, lists the genes whose call changes, and writes `results/dna_repair_age_adjusted.csv`. [PR #4](https://github.com/DanniPo/BioProj/pull/4), [commit cdf1eda](https://github.com/DanniPo/BioProj/commit/cdf1eda)
- Added shrunken fold changes (stretch task) with `lfc_shrink()` (apeGLM) for the repair-gene table, written to `results/dna_repair_DE_shrunken.csv`. It uses a separate `DeseqStats` object, so the `de`, `rep` and `adj` tables that Stages 6–8 use are unchanged. [commit cdf1eda](https://github.com/DanniPo/BioProj/commit/cdf1eda)
- Found and fixed a sign error. The design's reference level is "TNBC" (alphabetical), so `lfc_shrink()` returns the non-TNBC-vs-TNBC coefficient. The code flips the sign back and asserts that the direction agrees with the original results. [commit cdf1eda](https://github.com/DanniPo/BioProj/commit/cdf1eda)
- Fixed two printout bugs found in my first Colab run (an age-effect line printing `nan`, and the "≥2-fold" lists also matching "<2x" genes), and added two Interpretation bullets (age, shrinkage) to Stage 5. [commit 8bac0e3](https://github.com/DanniPo/BioProj/commit/8bac0e3)
- Reviewed Dylan's Stage 4 PR and Arop's Stage 3 PR (see "Review I gave").

## What changed in the results
- **Age adjustment:** the repair-gene log2 fold changes are practically identical to the original ones (Pearson r = 1.000, Spearman ρ = 0.999; median change 0.009, largest 0.072 for *NTHL1*). 115 instead of 116 repair genes stay significant (79 up / 36 down). Only *RAD51C* changes its call (slightly down → not significant), and the same 11 genes change at least two-fold. The `~group` fit on the same patients also gives r = 1.000. Age shifts repair genes by a median of only 0.022 log2 per 10 years, against a median TNBC effect of 0.304. **The TNBC repair differences are not an age effect.**
- **Shrinkage:** r = 1.0000 with the raw fold changes; median shift 0.002, largest 0.012 (*POLN*). The ranking and the 11 ≥2-fold genes are unchanged, because ~1,000 tumours give small standard errors.
- The Stage 5 Interpretation has two new bullets describing these results.

## How I checked it
- Ran the full notebook in Colab (Runtime → Run all) and checked the new 5.5 and 5.6 outputs. That run showed the two printout bugs, which I fixed in 8bac0e3.
- Confirmed the fixed printouts in the executed notebook from GitHub Actions on `main`: the age-effect line prints 0.022, and the ≥2-fold lists contain the expected 11 genes.
- Checked that `build_notebook.py --check` passes and that the new variable names do not clash with anything used in Stages 6–8.
- The "Notebook runs top to bottom" check passed on my PR: [CI run](https://github.com/DanniPo/BioProj/actions/runs/44467902914)

## Review I gave
- **Dylan's Stage 4 PR (#1):** reviewed and approved. [review](https://github.com/DanniPo/BioProj/pull/1#pullrequestreview-5958710330)
- **Arop's Stage 3 PR (#6):** the notebook check was failing in Stage 5 with `Length of values (40761) does not match length of index (16697)`. I traced it to the normalisation step, which used `counts` (all 40,761 genes) instead of the filtered `counts_f` (16,697 genes). I suggested the one-word fix, noted that the bug would also affect Stage 4 (PCA) and Stage 7 (ML), and asked for the removed Stage 3 Interpretation to be restored. The fix and the restored Interpretation were in the next commit (198ac11), and the check passed. [review](https://github.com/DanniPo/BioProj/pull/6#pullrequestreview-5973043495)

## Presentation
Slides 8–10 (Stage 5): DESeq2 results (116 of 148 repair genes, 11 two-fold), the proliferation adjustment, and the robustness checks (age, gene-set test, shrinkage).

## Evidence links
- Pull request: https://github.com/DanniPo/BioProj/pull/4
- Commits: [cdf1eda](https://github.com/DanniPo/BioProj/commit/cdf1eda) Stage 5: add age-adjusted DESeq2 and LFC shrinkage; [8bac0e3](https://github.com/DanniPo/BioProj/commit/8bac0e3) Stage 5: interpret age and shrinkage checks
- Reviews: https://github.com/DanniPo/BioProj/pull/1#pullrequestreview-5958710330, https://github.com/DanniPo/BioProj/pull/6#pullrequestreview-5973043495
- CI run: https://github.com/DanniPo/BioProj/actions/runs/44467902914
