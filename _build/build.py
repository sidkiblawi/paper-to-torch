"""Generate exercises/*.ipynb and solutions/*.ipynb from _build/src/*.txt.

Source format:
    #%% md              -> markdown cell (both versions)
    #%% md solution     -> markdown cell only in the solution notebook
    #%% code            -> code cell
Inside code cells, lines between `### BEGIN SOLUTION` and `### END SOLUTION` are
replaced in the exercise version by `# YOUR CODE HERE` + `raise NotImplementedError`.

Usage:  uv run python _build/build.py [--execute]
"""
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_build" / "src"


def parse(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        m = re.match(r"^#%% (md|code)( solution)?\s*$", line)
        if m:
            if kind:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = m.group(1) + ("_sol" if m.group(2) else "")
            buf = []
        else:
            buf.append(line)
    if kind:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def strip_solutions(code):
    out, skipping = [], False
    for line in code.splitlines():
        if line.strip() == "### BEGIN SOLUTION":
            indent = line[: len(line) - len(line.lstrip())]
            out += [f"{indent}# YOUR CODE HERE", f"{indent}raise NotImplementedError"]
            skipping = True
        elif line.strip() == "### END SOLUTION":
            skipping = False
        elif not skipping:
            out.append(line)
    return "\n".join(out)


def unmark(code):
    return "\n".join(l for l in code.splitlines()
                     if l.strip() not in ("### BEGIN SOLUTION", "### END SOLUTION"))


def build(path):
    cells = parse(path.read_text())
    ex, sol = [], []
    for kind, body in cells:
        if kind == "md":
            ex.append(new_markdown_cell(body)); sol.append(new_markdown_cell(body))
        elif kind == "md_sol":
            sol.append(new_markdown_cell(body))
        else:
            ex.append(new_code_cell(strip_solutions(body)))
            sol.append(new_code_cell(unmark(body)))
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    outs = {}
    targets = (("exercises", ex),) if path.stem.startswith("99") else (("exercises", ex), ("solutions", sol))
    for folder, cs in targets:
        (ROOT / folder).mkdir(exist_ok=True)
        out = ROOT / folder / (path.stem + ".ipynb")
        nbformat.write(new_notebook(cells=cs, metadata=meta), out)
        outs[folder] = out
    return outs


def execute(path):
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for src in sorted(SRC.glob("*.txt")):
        if only and not any(o in src.stem for o in only):
            continue
        outs = build(src)
        print("built", src.stem)
        if "--execute" in sys.argv and "solutions" in outs:
            execute(outs["solutions"])
            print("  ✅ solution notebook runs clean")
