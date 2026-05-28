"""Regression guard for the five hand-authored concept SVGs.

For each SVG under `notebooks/shared/diagrams/`, assert:
  - All expected symbol-level identifiers are present in the SVG text.
  - None of the stripped explanatory phrases have crept back in.

The test reads the raw SVG file; no rendering required. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAGRAMS_DIR = REPO_ROOT / "notebooks" / "shared" / "diagrams"

EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "mask-load-1d.svg": {
        "required": ["tl.load"],
        "forbidden": ["past n", "loaded vector", "ptr =", "mask = offs"],
    },
    "compile-step-4.svg": {
        "required": ["Python", "Triton-IR", "TritonGPU-IR", "LLVM IR", "PTX", "SASS"],
        "forbidden": ["the driver JITs", "cubin is cached", "virtual ISA", "predicate"],
    },
    "pointer-grid-2d.svg": {
        "required": ["offs_m", "offs_n", "stride_m", "stride_n"],
        "forbidden": ["base +", "ptr math is typed", "itemsize"],
    },
    "nb05-kloop.svg": {
        "required": ["tl.dot", "BLOCK_M", "BLOCK_K", "BLOCK_N", "acc"],
        "forbidden": ["for k_start", "in registers", "ACCUMULATE", "TODO"],
    },
    "nb05-store.svg": {
        "required": ["tl.store", "HBM", "c_mask"],
        "forbidden": ["WRITE", "TODO", "no out-of-bounds", "store only"],
    },
    "tile-1d.svg": {
        "required": ["pid", "BLOCK_SIZE", "n ="],
        "forbidden": ["program", "partition", "chunk", "program_id", "arange", "tl.load"],
    },
}


@pytest.mark.parametrize("svg_name", sorted(EXPECTATIONS), ids=lambda n: n)
def test_svg_has_required_identifiers(svg_name: str) -> None:
    text = (DIAGRAMS_DIR / svg_name).read_text(encoding="utf-8")
    for token in EXPECTATIONS[svg_name]["required"]:
        assert token in text, f"{svg_name}: required identifier {token!r} missing"


@pytest.mark.parametrize("svg_name", sorted(EXPECTATIONS), ids=lambda n: n)
def test_svg_does_not_contain_stripped_prose(svg_name: str) -> None:
    text = (DIAGRAMS_DIR / svg_name).read_text(encoding="utf-8")
    for token in EXPECTATIONS[svg_name]["forbidden"]:
        assert token not in text, (
            f"{svg_name}: forbidden phrase {token!r} reappeared "
            f"(belongs in the notebook markdown, not the SVG)"
        )
