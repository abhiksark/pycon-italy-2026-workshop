# bench/compare_vs_torch.py
"""Benchmark four matmul implementations at a fixed shape.

This script is the standalone equivalent of the wrap-up cell in
``notebooks/05-tiled-matmul-triton.ipynb``. It prints a 4-row comparison:

  1. naive Python triple-loop (timed on a single row, then extrapolated)
  2. numpy.matmul (full shape, on CPU)
  3. the workshop's Triton tiled matmul (full shape, on GPU)
  4. torch.matmul (full shape, on GPU)

Run: python -m bench.compare_vs_torch [--shape 1024]

When torch+triton are unavailable, the GPU rows are skipped with a message,
the CPU rows still run, and the table prints just those.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from typing import TextIO

import numpy as np


@dataclasses.dataclass
class Row:
    name: str
    gflops: float
    speedup_vs_naive: float


def naive_python_matmul_first_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes one row of `a @ b` using nested Python loops.

    Only one row is computed because doing the full matmul in pure Python
    would take minutes at workshop sizes. We extrapolate.
    """
    _, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError(f"inner dims must match, got {k_a} and {k_b}")
    out = np.zeros(n, dtype=np.float32)
    a0 = a[0]
    for j in range(n):
        acc = 0.0
        for k in range(k_a):
            acc += float(a0[k]) * float(b[k, j])
        out[j] = acc
    return out


def extrapolated_seconds(one_row_seconds: float, m: int) -> float:
    """One row × m rows = the full matmul time (linear extrapolation)."""
    return one_row_seconds * m


def fill_speedups(rows: list[Row]) -> list[Row]:
    """Assigns `speedup_vs_naive` to every row using rows[0] as the reference."""
    if not rows:
        return rows
    naive_gflops = rows[0].gflops
    if naive_gflops <= 0:
        raise ValueError("naive row must have positive GFLOPs")
    return [
        dataclasses.replace(r, speedup_vs_naive=r.gflops / naive_gflops)
        for r in rows
    ]


def format_table(rows: list[Row], file: TextIO = sys.stdout) -> None:
    print(f"{'Implementation':36s} {'GFLOPs':>12s} {'Speedup vs naive':>20s}", file=file)
    print("-" * 70, file=file)
    for r in rows:
        print(f"{r.name:36s} {r.gflops:12.4f} {r.speedup_vs_naive:18.0f}x", file=file)


def benchmark_naive(a: np.ndarray, b: np.ndarray) -> Row:
    """Times one row of the naive Python matmul (median of two after one warmup)."""
    m, _ = a.shape
    _, n = b.shape
    k = a.shape[1]
    # One warmup discards interpreter/cache cold-paths; two timed runs + statistics.median
    # makes the baseline robust to a single CPU preemption.
    naive_python_matmul_first_row(a, b)
    samples = []
    for _ in range(2):
        start = time.perf_counter()
        naive_python_matmul_first_row(a, b)
        samples.append(time.perf_counter() - start)
    one_row = statistics.median(samples)
    full = extrapolated_seconds(one_row, m)
    gflops = (2.0 * m * n * k) / (full * 1e9)
    return Row("naive python loop (extrapolated)", gflops, 1.0)


def benchmark_numpy(a: np.ndarray, b: np.ndarray) -> Row:
    m, _ = a.shape
    _, n = b.shape
    k = a.shape[1]
    # warm up cache
    a @ b
    start = time.perf_counter()
    for _ in range(3):
        a @ b
    elapsed = (time.perf_counter() - start) / 3
    gflops = (2.0 * m * n * k) / (elapsed * 1e9)
    return Row("numpy.matmul (CPU fp32)", gflops, 0.0)


def benchmark_torch_and_triton(m: int, n: int, k: int) -> list[Row]:
    try:
        import torch
        import triton
        from solutions._shared import matmul_kernel
    except ImportError as exc:
        print(f"# skipping GPU rows: {exc}", file=sys.stderr)
        return []
    if not torch.cuda.is_available():
        print("# skipping GPU rows: no CUDA device", file=sys.stderr)
        return []

    A = torch.randn((m, k), device="cuda", dtype=torch.float32)
    B = torch.randn((k, n), device="cuda", dtype=torch.float32)
    C = torch.empty((m, n), device="cuda", dtype=torch.float32)

    BLOCK = 64
    grid = (triton.cdiv(m, BLOCK), triton.cdiv(n, BLOCK))

    def run_triton() -> None:
        matmul_kernel[grid](
            A, B, C, m, n, k,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK,
        )
        torch.cuda.synchronize()

    def run_torch() -> None:
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()

    # warmup
    for _ in range(5):
        run_triton()
        run_torch()

    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        run_triton()
    t_triton = (time.perf_counter() - start) / runs

    start = time.perf_counter()
    for _ in range(runs):
        run_torch()
    t_torch = (time.perf_counter() - start) / runs

    g_triton = (2.0 * m * n * k) / (t_triton * 1e9)
    g_torch = (2.0 * m * n * k) / (t_torch * 1e9)
    return [
        Row("triton tiled matmul (workshop)", g_triton, 0.0),
        Row("torch.matmul (cuBLAS)", g_torch, 0.0),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, default=1024, help="square M=N=K")
    args = parser.parse_args(argv)

    n = args.shape
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, n)).astype(np.float32)
    b_np = rng.standard_normal((n, n)).astype(np.float32)

    rows: list[Row] = []
    rows.append(benchmark_naive(a_np, b_np))
    rows.append(benchmark_numpy(a_np, b_np))
    rows.extend(benchmark_torch_and_triton(n, n, n))
    rows = fill_speedups(rows)

    print(f"\nmatmul shape: {n} x {n} x {n} (M = N = K)\n")
    format_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
