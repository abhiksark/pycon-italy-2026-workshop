# notebooks/shared/benchmark_utils.py
"""Tiny timing helpers shared across the workshop notebooks.

CPU-only fallbacks are deliberate: the tests run anywhere, and the helpers
work the same way whether torch.cuda is available or not. When torch is
installed and CUDA is available, the caller is expected to insert
`torch.cuda.synchronize()` inside their workload callable before the timer
sees the result.
"""

from __future__ import annotations

import time
from statistics import median
from typing import Callable


def gflops_for_matmul(m: int, n: int, k: int, seconds: float) -> float:
    """Returns sustained GFLOPs for an `(m, k) @ (k, n) -> (m, n)` matmul.

    A matmul performs `2 * m * n * k` floating-point operations (one multiply
    and one add per inner-product step).

    Args:
        m: rows of A and rows of the output.
        n: columns of B and columns of the output.
        k: shared inner dimension.
        seconds: wall-clock seconds the matmul took.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds!r}")
    return (2.0 * m * n * k) / (seconds * 1e9)


def median_seconds(
    workload: Callable[[], None],
    *,
    runs: int,
    warmup: int,
) -> float:
    """Times a no-argument callable and returns the median wall-clock seconds.

    The caller is responsible for any device synchronisation inside the
    workload. The first ``warmup`` calls are discarded; the next ``runs``
    are timed and their median is returned.

    Args:
        workload: A no-argument callable to time.
        runs: Number of timed iterations; must be positive.
        warmup: Number of untimed warm-up calls; must be non-negative.

    Returns:
        Median wall-clock duration in seconds across ``runs`` iterations.
    """
    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs!r}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup!r}")

    for _ in range(warmup):
        workload()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        workload()
        samples.append(time.perf_counter() - start)

    return median(samples)
