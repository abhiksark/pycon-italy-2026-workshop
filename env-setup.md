<!-- env-setup.md -->

# Local environment setup

Colab is the supported runtime for the workshop. This document covers running the notebooks on a local Linux machine with an NVIDIA GPU. macOS and Windows are not supported.

## Prerequisites

- Linux (Ubuntu 22.04 LTS or newer recommended).
- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer).
- NVIDIA driver ≥ 535. Verify with `nvidia-smi`.
- CUDA 12.x runtime libraries available on the system path.
- Python ≥ 3.10.

## Install

```bash
git clone <this-repo>
cd pycon-italy
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[gpu,dev]"
```

## Verify

```bash
python -c "import torch, triton, cupy; print(torch.cuda.is_available())"
```

Expected: `True`. If `False`, the rest of the workshop will not work locally - use Colab instead.

## Run a notebook

```bash
jupyter notebook notebooks/02-vector-add-triton.ipynb
```

## If CuPy fails to import on local

CuPy needs a CUDA toolkit version that matches your driver. If `pip install cupy-cuda12x` fails, try `pip install cupy-cuda11x` against an 11.x driver, or skip CuPy entirely - it is only used in notebook 01, which is a speaker demo. The rest of the workshop runs on Triton + PyTorch alone.
