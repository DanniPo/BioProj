# How to contribute

**Rule 1:** edit only your own file in `stages/`. The notebook `TNBC_DNA_repair_TCGA.ipynb` is *generated* from those files by `build_notebook.py`, so never edit it by hand.

**Rule 2:** commit from **your own GitHub account**. The commit history is our evidence of who did what, and it only counts if it is yours.

## One-time setup
1. Accept the GitHub invitation to the repository.
2. Make sure your commits are linked to your account: set the same email as your GitHub account (or your GitHub "noreply" address).
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```

## Option A: work on your own computer
```bash
git clone https://github.com/<account>/<repo>.git
cd <repo>
pip install -r requirements.txt
git checkout -b stage-05-pete            # one branch per task: stage-<nn>-<name>
```
1. Open `TNBC_DNA_repair_TCGA.ipynb` in Jupyter or VS Code, choose **Run All**, and try your new code in extra cells under your stage until it works.
2. Copy the finished cells into your stage file (e.g. `stages/05_core_analysis_de.py`). Code cells start with `# %%`. Markdown cells start with `# %% [markdown]`, and each of their text lines starts with `# `.
3. Rebuild the notebook:
   ```bash
   python build_notebook.py --execute    # rebuild and run everything (about 5 min)
   ```
4. Commit and push, then open a pull request on GitHub:
   ```bash
   git add stages/05_core_analysis_de.py TNBC_DNA_repair_TCGA.ipynb TNBC_DNA_repair_TCGA.py
   git commit -m "Stage 5: add age-adjusted DESeq2 sensitivity check"
   git push -u origin stage-05-pete
   ```

## Option B: browser only (Colab + GitHub)
1. Open the notebook in Colab (badge in the README), go to your stage, and try your new code in extra cells until it works.
2. On GitHub, open your stage file, click the pencil (or press `.` for the web editor), paste in your finished cells using the same `# %%` format, and **commit to a new branch** (`stage-05-pete`).
3. Open a pull request. GitHub Actions rebuilds the notebook from the stage files and runs it top to bottom. The executed notebook appears as a downloadable artifact on the PR's check run, so you can see your outputs.

## Pull requests and reviews
- Title: `Stage N: what you did`. Fill in the PR template (what changed, how you checked it, key output).
- Your review partner (see `WORKPLAN.md`) reviews on GitHub: leave at least one real comment or question, then approve. Reviews are part of the evidence too.
- The maintainer merges once the check is green and the review is approved.

## Recording your evidence
When your PR is merged, fill in `contributions/<your-first-name>.md` (copy `contributions/TEMPLATE.md`) with links to your PR, commits, review and slides. Before submission the maintainer runs:
```bash
python tools/contribution_report.py      # writes CONTRIBUTIONS_LOG.md from the git history
```
