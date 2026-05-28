"""Consistency test: every `attachment:foo.png` reference in a notebook
resolves to an actual attachment in the same cell, and vice versa.

Catches drift the moment someone hand-edits a notebook and breaks an
attachment reference or leaves an orphan attachment behind. CPU-only.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATTACHMENT_REF_RE = re.compile(r"attachment:([^\s\)\"']+)")


def _notebook_paths() -> list[Path]:
    candidates: list[Path] = []
    for pat in ("notebooks/0[2-5]-*.ipynb", "solutions/0[2-5]-*.solution.ipynb"):
        candidates.extend(sorted(REPO_ROOT.glob(pat)))
    return candidates


@pytest.mark.parametrize("nb_path", _notebook_paths(), ids=lambda p: p.name)
def test_attachment_references_resolve(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        refs = set(ATTACHMENT_REF_RE.findall(source))
        attachments = set((cell.get("attachments") or {}).keys())
        missing = refs - attachments
        orphans = attachments - refs
        assert not missing, (
            f"{nb_path.name} cell {cell_idx}: references {sorted(missing)} "
            f"but no matching attachment is on the cell"
        )
        assert not orphans, (
            f"{nb_path.name} cell {cell_idx}: has attachment(s) {sorted(orphans)} "
            f"with no `attachment:` reference in the cell source"
        )


def test_at_least_one_notebook_has_an_attachment() -> None:
    """Sanity check: every attendee notebook is supposed to have at least
    one diagram attached. If this regresses to zero, something has gone wrong."""
    found = False
    for nb_path in _notebook_paths():
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "markdown" and cell.get("attachments"):
                found = True
                break
        if found:
            break
    assert found, "no notebook in the curriculum has any cell attachments"
