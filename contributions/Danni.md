@@ -1,29 +0,0 @@
# Contribution: Danni

# Contribution: Danni Podho

## Stage(s) I led
- **Stage 7 · Machine learning**
- **Repository maintainer:** set up the repository, merged pull requests, and produced the final notebook and results.

## What I did
- **Created the repository and uploaded the shared starting draft** of the full analysis, which was prepared with AI assistance and is labelled as such: commit [`cc4cbe1`](https://github.com/DanniPo/BioProj/commit/cc4cbe1), *"Baseline: AI-assisted draft of the full analysis"*.
- **Extended Stage 7** in [PR #3](https://github.com/DanniPo/BioProj/pull/3), commit [`975df5c`](https://github.com/DanniPo/BioProj/commit/975df5c), with three additions:
  - **5 × 5 repeated cross-validation** comparing every model on the same 25 splits, including 30 random gene sets;
  - **random-forest decision threshold** tuned on training out-of-fold predictions only, then applied once to the test set;
  - **permutation importance** for the logistic regression on the test set.

  The code for these additions was written with help from an AI assistant (Claude). I integrated it, ran it, and checked its outputs.
- **Maintained the repository:**
  - merged pull requests [#2](https://github.com/DanniPo/BioProj/pull/2), [#5](https://github.com/DanniPo/BioProj/pull/5), [#6](https://github.com/DanniPo/BioProj/pull/6), [#8](https://github.com/DanniPo/BioProj/pull/8) and [#9](https://github.com/DanniPo/BioProj/pull/9);
  - kept `team.json` up to date: [`c82002d`](https://github.com/DanniPo/BioProj/commit/c82002d), [`5cef374`](https://github.com/DanniPo/BioProj/commit/5cef374);
  - restored the contributions template: [`00cb685`](https://github.com/DanniPo/BioProj/commit/00cb685), [`9dca123`](https://github.com/DanniPo/BioProj/commit/9dca123).
- **Produced the final submission notebook and results:**
  - ran the notebook in Colab: [`c0e52d6`](https://github.com/DanniPo/BioProj/commit/c0e52d6);
  - replaced it with the complete GitHub Actions run, in which all 41 code cells have outputs, and fixed the `results/` folder: [`cff2776`](https://github.com/DanniPo/BioProj/commit/cff2776), [`ebec1a9`](https://github.com/DanniPo/BioProj/commit/ebec1a9), [`70ab2e7`](https://github.com/DanniPo/BioProj/commit/70ab2e7).
- **Updated the Stage 7 interpretation** to the final numbers after the Stage 3 changes shifted them slightly: commit `<hash>`.

## What changed in the results
- **Repeated CV:** the repair-gene model scores 0.969 ± 0.013 and the random gene sets 0.967 ± 0.013. Repair genes win only 61% of the paired comparisons and beat 77% of the random-set means; the single 5-fold split had said 93%. So the single split flattered the repair genes, and the "not repair-specific" conclusion is firmer.
- **Random forest:** tuning the threshold to 0.22 raised test sensitivity from 44% to 78% (14 → 25 of 32 TNBC) and balanced accuracy from 0.70 to 0.85. The logistic regression is still the preferred model (balanced accuracy 0.87).
- **Permutation importance:** no single gene matters much. The largest drop, from shuffling *ATM*, is 0.006 AUC, and only 72 of 148 genes have any effect. The signal is spread across many correlated genes, which explains why random genes perform as well.
- **Interpretation:** I updated the Stage 7 text with these numbers and sent Dylan the Objective 4 sentence for the conclusions.

## How I checked it
- **Notebook check (GitHub Actions)** passed on my Stage 7 branch: [run 44344268107](https://github.com/DanniPo/BioProj/actions/runs/44344268107).
- **Earlier numbers unchanged:** confirmed my new cells didn't change anything already in the notebook. The bootstrap CI stayed the same, because my code uses its own random generator.
- **Final notebook:** `python build_notebook.py --check` reports that the notebook matches `stages/`. All 41 code cells have outputs, with no errors. Final run: [run 45631750622](https://github.com/DanniPo/BioProj/actions/runs/45631750622).
- **Colab:** ran the notebook top to bottom. Screenshot: `<link or slide number>`.



## Presentation
- Stage 7 setup and leakage control, baselines (repeated-CV boxplot), test performance and threshold tuning, take-home message.

## Evidence links
- Pull request: https://github.com/DanniPo/BioProj/pull/3
- Commits: `cc4cbe1` (baseline), `975df5c` (Stage 7), `c82002d`, `5cef374`, `00cb685`, `9dca123`, `c0e52d6`, `cff2776`, `ebec1a9`, `70ab2e7`, `<hash>`
- Merges: #2, #5, #6, #8, #9
- Review: `<link>`
- CI runs: https://github.com/DanniPo/BioProj/actions/runs/44344268107, https://github.com/DanniPo/BioProj/actions/runs/45631750622
