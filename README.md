<!-- README.md -->

# Write Your First High-Performance GPU Kernel in Python!

**PyCon Italy 2026 · Bologna · May 27–30, 2026**
Speaker: [Abhik Sarkar](https://www.abhik.ai) · [Session page](https://2026.pycon.it/profile/bxlreb)

A 120-minute hands-on workshop. You walk in fluent in NumPy and PyTorch. You walk out with your own tiled matrix-multiplication kernel running on a GPU - correctness-tested against `torch.matmul`, benchmarked, and **roughly 100,000× faster than the equivalent naive Python loop**.

No prior CUDA. No prior GPU programming. Just Python.

---

## The arc

Five notebooks. Each one introduces exactly one new GPU primitive on top of the previous.

```
        threads          shared mem         2D + halos       tiling + tl.dot
           │                  │                  │                  │
           ▼                  ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
   │ vector add  │ →  │  reduction  │ →  │ image blur  │ →  │ tiled matmul │
   │ (CuPy demo  │    │  (parallel  │    │ (2D tiles,  │    │ (the capstone│
   │  +  Triton) │    │    sum)     │    │   halos)    │    │   - 6 TODOs) │
   └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
   nb01-02            nb03                nb04                nb05
```

Every attendee notebook ships with the boilerplate written - imports, input data, reference output, correctness check, timing harness. You fill in the kernel body, marked by numbered `# TODO (N/M):` blocks.

The speaker demos one kernel from a blank cell. Attendees write the next.

---

## Prerequisites

**Required**
- Comfortable Python: closures, decorators, list comprehensions.
- NumPy fluency: indexing, broadcasting, dtypes.
- Basic PyTorch: creating tensors, moving to a device, calling `torch.matmul`.
- A Google account (for Colab).

**Helpful, not required**
- Reading-level familiarity with C or C++ (eases the CuPy RawKernel segment).
- A mental model of matrix multiplication.

**Explicitly not required**
- Any prior CUDA, OpenCL, or GPU programming experience.
- Deep-learning theory beyond "what a tensor is."
- Your own GPU. The workshop runs entirely on Colab's free T4.

---

## Before the workshop (15 minutes)

1. Open [`pre-work/00-hello-gpu.ipynb`](pre-work/00-hello-gpu.ipynb) in Google Colab.
2. Set the runtime to **T4 GPU**: _Runtime → Change runtime type → T4 GPU_.
3. _Runtime → Run all_.
4. Save a copy to your Drive (_File → Save a copy in Drive_) so it survives a disconnect.

You're ready when the last cell prints:

```
✓ All checks passed. See you in Bologna.
```

If anything fails, **reply to the pre-work email**. We'll sort it out before the room is full.

---

## On the day

We work through the five attendee notebooks in order. Each `# TODO (N/M):` is one short, well-scoped change to the kernel body - usually 1–3 lines.

| # | Notebook | Format | What's new |
|---|----------|--------|------------|
| 01 | [`notebooks/01-vector-add-cupy-raw.ipynb`](notebooks/01-vector-add-cupy-raw.ipynb) | Speaker demo | What a GPU kernel actually is: CUDA C in a Python string, JIT-compiled and launched. |
| 02 | [`notebooks/02-vector-add-triton.ipynb`](notebooks/02-vector-add-triton.ipynb) | 3 TODOs | Your first Triton kernel. `program_id`, `tl.arange`, masks. |
| 03 | [`notebooks/03-reduction-triton.ipynb`](notebooks/03-reduction-triton.ipynb) | 4 TODOs | Parallel sum. `tl.sum` and the multi-block pattern. |
| 04 | [`notebooks/04-image-blur-triton.ipynb`](notebooks/04-image-blur-triton.ipynb) | 5 TODOs | 2D programs, halos, a visible payoff (blurred image). |
| 05 | [`notebooks/05-tiled-matmul-triton.ipynb`](notebooks/05-tiled-matmul-triton.ipynb) | 6 TODOs | **The capstone.** Tiling + `tl.dot`. Benchmarked against `torch.matmul`. |

Total: 18 TODOs across the four attendee notebooks.

Solutions and the standalone benchmark script land here on the `main` branch when the workshop ends on May 27.

### Schedule (120 min, four quadrants)

The session breaks into four themed quadrants. Each has one verb attendees can hold in their head; the story arc is **WATCH → WRITE → MEASURE → CLIMB**. The mid-workshop roofline detour sits between nb03 and nb04 - this is where attendees first see "% of T4 peak" on their own kernels and learn why nb04/nb05 are about climbing the diagonal.

| Quadrant            | Block                          | Min | What                                                                              |
|---------------------|--------------------------------|-----|-----------------------------------------------------------------------------------|
| **Q1 · WATCH**      | Setup + intro                  |   5 | T4 runtime check, repo open                                                       |
| *(25 min)*          | Part 1 slides                  |   9 | intro + 7-slide GPU model                                                         |
|                     | nb01 vector add (CuPy demo)    |  11 | speaker demo, no TODOs; what a kernel actually is                                 |
| **Q2 · WRITE**      | Part 2 slides                  |   5 | the Triton recipe, TODOs                                                          |
| *(30 min)*          | nb02 vector add (Triton)       |  12 | 3 TODOs; print effective GB/s vs T4 peak                                          |
|                     | nb03 reduction                 |  13 | 4 TODOs; print effective GB/s vs T4 peak                                          |
| **Q3 · MEASURE**    | Part 3 slides                  |  10 | roofline + memory wall                                                            |
| *(28 min)*          | nb04 image blur                |  18 | 5 TODOs; halos = first lever against the memory wall                              |
| **Q4 · CLIMB**      | Part 4 slides                  |   5 | tile, recipe, what good is                                                        |
| *(37 min)*          | nb05 tiled matmul              |  27 | 6 TODOs (capstone); print arithmetic intensity + % of T4 peak FP32                |
|                     | Wrap + Q&A                     |   5 | closing roofline read, take-homes                                                 |
| **Total**           |                                | **120** |                                                                               |

---

## If something goes wrong on the day

- **"No CUDA GPU detected"** - _Runtime → Change runtime type → T4 GPU_, then _Runtime → Restart and run all_.
- **Triton or CuPy import fails** - re-run the bootstrap cell at the top of the notebook; it installs anything missing.
- **Colab disconnects mid-workshop** - your saved copy in Drive is still there; reconnect and pick up where you left off.
- **Kernel compile error** - read the line Triton points at. Nine times in ten it's a missing `mask=` argument or a typo in an offset expression.
- **Correctness check fails** - re-read the TODO comments above the failing block. The reference output and shape are right there in the notebook.

---

## Running locally instead of Colab

Colab is the supported path. Local is best-effort and Linux-only - see [`env-setup.md`](env-setup.md) for prerequisites (NVIDIA driver ≥ 535, CUDA 12, Python ≥ 3.10).

```bash
git clone https://github.com/abhiksark/pycon-italy-2026-workshop.git
cd pycon-italy-2026-workshop
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/02-vector-add-triton.ipynb
```

---

## Repo layout

| Path | What it is |
|------|------------|
| [`pre-work/00-hello-gpu.ipynb`](pre-work/00-hello-gpu.ipynb) | 15-minute env check. Run this **before** the workshop. |
| [`notebooks/01-vector-add-cupy-raw.ipynb`](notebooks/01-vector-add-cupy-raw.ipynb) | Speaker demo. CUDA C in a string, launched from Python via CuPy. |
| [`notebooks/02-vector-add-triton.ipynb`](notebooks/02-vector-add-triton.ipynb) | Your first Triton kernel. Three TODOs. |
| [`notebooks/03-reduction-triton.ipynb`](notebooks/03-reduction-triton.ipynb) | Parallel sum. Core single-block + bonus multi-block. |
| [`notebooks/04-image-blur-triton.ipynb`](notebooks/04-image-blur-triton.ipynb) | 1D blur core + 2D halo bonus. Visible payoff. |
| [`notebooks/05-tiled-matmul-triton.ipynb`](notebooks/05-tiled-matmul-triton.ipynb) | The capstone. Six TODOs, protected slot. |
| [`notebooks/shared/`](notebooks/shared/) | Timing helper and the procedural blur image. |
| [`env-setup.md`](env-setup.md) | Run notebooks locally instead of Colab. |

---

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

The test suite covers the shared timing helper. CPU-only - no GPU required.

---

## License

Workshop materials are MIT-licensed - see [`LICENSE`](LICENSE). The procedural blur image in `notebooks/shared/assets/` is in the public domain.
