# bench/test_compare_vs_torch.py
"""Unit tests for compare_vs_torch. Run without a GPU."""

from __future__ import annotations

import io

import numpy as np
import pytest

from bench import compare_vs_torch as cvt


def test_naive_python_matmul_one_row_matches_numpy_first_row() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 4)).astype(np.float32)
    expected = (a @ b)[0]
    got = cvt.naive_python_matmul_first_row(a, b)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_extrapolated_seconds_scales_by_row_count() -> None:
    # If one row took 0.01s on an (M, N, K) matmul, the full matmul takes M * 0.01s.
    full = cvt.extrapolated_seconds(one_row_seconds=0.01, m=128)
    assert full == pytest.approx(1.28)


def test_format_table_emits_expected_columns() -> None:
    rows = [
        cvt.Row("naive python loop", 0.012, 1.0),
        cvt.Row("numpy.matmul", 18.4, 1500.0),
    ]
    buf = io.StringIO()
    cvt.format_table(rows, file=buf)
    text = buf.getvalue()
    assert "Implementation" in text
    assert "GFLOPs" in text
    assert "Speedup" in text
    assert "naive python loop" in text
    assert "numpy.matmul" in text


def test_row_speedup_relative_to_naive_is_computed_from_gflops() -> None:
    naive = cvt.Row("naive", 0.01, 1.0)
    fast = cvt.Row("fast", 10.0, 0.0)
    # If we know naive is 0.01 GFLOPs and fast is 10 GFLOPs, speedup = 1000x.
    rows = cvt.fill_speedups([naive, fast])
    assert rows[0].speedup_vs_naive == pytest.approx(1.0)
    assert rows[1].speedup_vs_naive == pytest.approx(1000.0)


def test_benchmark_naive_returns_positive_gflops_and_unit_speedup() -> None:
    """benchmark_naive() should return a non-zero GFLOPs and speedup_vs_naive == 1.0."""
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    b = rng.standard_normal((32, 32)).astype(np.float32)
    row = cvt.benchmark_naive(a, b)
    assert row.gflops > 0
    assert row.speedup_vs_naive == pytest.approx(1.0)
    assert row.name == "naive python loop (extrapolated)"
