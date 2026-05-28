# notebooks/shared/test_benchmark_utils.py
"""Tests for benchmark_utils. All run on CPU so the suite is portable."""

from __future__ import annotations

import time

import pytest

from notebooks.shared import benchmark_utils as bu


def test_gflops_for_matmul_returns_expected_value() -> None:
    # A 1024x1024 matmul performs 2 * 1024**3 floating-point ops.
    gflops = bu.gflops_for_matmul(m=1024, n=1024, k=1024, seconds=1.0)
    expected = 2 * 1024 ** 3 / 1e9
    assert gflops == pytest.approx(expected, rel=1e-9)


def test_gflops_scales_inversely_with_time() -> None:
    fast = bu.gflops_for_matmul(m=128, n=128, k=128, seconds=0.5)
    slow = bu.gflops_for_matmul(m=128, n=128, k=128, seconds=1.0)
    assert fast == pytest.approx(slow * 2.0)


def test_median_seconds_returns_median_of_runs() -> None:
    def workload() -> None:
        time.sleep(0.01)

    seconds = bu.median_seconds(workload, runs=5, warmup=1)
    assert 0.005 <= seconds <= 0.2


def test_median_seconds_rejects_zero_runs() -> None:
    with pytest.raises(ValueError):
        bu.median_seconds(lambda: None, runs=0, warmup=0)


def test_gflops_rejects_non_positive_seconds() -> None:
    with pytest.raises(ValueError):
        bu.gflops_for_matmul(m=128, n=128, k=128, seconds=0.0)
    with pytest.raises(ValueError):
        bu.gflops_for_matmul(m=128, n=128, k=128, seconds=-1.0)


def test_median_seconds_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError):
        bu.median_seconds(lambda: None, runs=1, warmup=-1)
