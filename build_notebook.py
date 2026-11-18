"""Assemble the eight stage files in stages/ into the final notebook.

    python build_notebook.py              # rebuild TNBC_DNA_repair_TCGA.ipynb/.py, keeping outputs of unchanged cells
    python build_notebook.py --execute    # rebuild and run top to bottom (needs the packages in requirements.txt)
    python build_notebook.py --check      # exit 1 if the committed notebook is out of date with stages/

Each member edits only their own file in stages/. The final notebook is generated, never edited by hand.
"""
import argparse, json, re, sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent
STAGES = ROOT / "stages"
NOTEBOOK = ROOT / "TNBC_DNA_repair_TCGA.ipynb"
SCRIPT = ROOT / "TNBC_DNA_repair_TCGA.py"
HEADER = re.compile(r"\A# ---\n.*?\n# ---\n\n?", re.S)


def assemble():
    team = json.loads((ROOT / "team.json").read_text())
    parts, first_header = [], None
    for path in sorted(STAGES.glob("[0-9][0-9]_*.py")):
        text = path.read_text()
        header = HEADER.match(text)
        if header and first_header is None:
            first_header = header.group(0)
        body = HEADER.sub("", text).strip("\n")
        lead_key = team["stages"].get(path.stem)
        if lead_key:
            lead = team["members"][lead_key]["name"]
            body = re.sub(r"^(# ## Stage \d · .*)$", rf"\1\n#\n# *Stage lead: {lead}*", body, count=1, flags=re.M)
        parts.append(body)
    if not parts:
        sys.exit("no stage files found in stages/")
    return (first_header or "") + "\n\n".join(parts) + "\n"


def sources(nb):
    return [(c.cell_type, c.source.strip()) for c in nb.cells]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="run the notebook top to bottom after building")
    ap.add_argument("--check", action="store_true", help="only check that the committed notebook matches stages/")
    ap.add_argument("--output", type=Path, default=NOTEBOOK, help="where to write the notebook")
    args = ap.parse_args()

    text = assemble()
    new_nb = jupytext.reads(text, fmt="py:percent")

    if args.check:
        old_nb = nbformat.read(NOTEBOOK, as_version=4)
        if sources(old_nb) != sources(new_nb):
            sys.exit("TNBC_DNA_repair_TCGA.ipynb is out of date: run  python build_notebook.py  and commit the result.")
        print("notebook matches stages/")
        return

    # keep cell ids, and saved outputs, for cells that have not changed (keeps git diffs small)
    if args.output.exists():
        old_nb = nbformat.read(args.output, as_version=4)
        old = {}
        for c in old_nb.cells:
            old.setdefault((c.cell_type, c.source.strip()), c)
        kept = 0
        for c in new_nb.cells:
            match = old.pop((c.cell_type, c.source.strip()), None)
            if match is None:
                continue
            if "id" in match:
                c["id"] = match["id"]
            if c.cell_type == "code":
                c.outputs, c.execution_count = match.get("outputs", []), match.get("execution_count")
                kept += 1
        print(f"kept saved outputs for {kept} unchanged code cells")
    new_nb.metadata.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3", "language": "python"})

    if args.execute:
        from nbclient import NotebookClient
        print("running the notebook top to bottom ...")
        NotebookClient(new_nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()

    nbformat.write(new_nb, args.output)
    if args.output == NOTEBOOK:
        SCRIPT.write_text(text)
    print(f"wrote {args.output.name}" + (f" and {SCRIPT.name}" if args.output == NOTEBOOK else ""))


if __name__ == "__main__":
    main()
