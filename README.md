# JAX Transformer Optimization

A JAX/XLA transformer optimization toolkit that implements masked self-attention, LayerNorm, residual connections, and a GELU feed-forward network. It compares four execution strategies, packed QKV projections, FP32/BF16 precision, and sequence-length bucketing across 32 configurations.

Sequence-length bucketing achieved **3.46x faster cold-start workload completion** on a synthetic CPU workload, reducing shape-specific compilations from **13 to 3**.

## Run on this machine

From the project directory in PowerShell, using the existing local environment:

```powershell
# Check the execution device
.\.venv\Scripts\python.exe scripts/run.py doctor --backend cpu

# Run the tests
.\.venv\Scripts\python.exe scripts/run.py test

# Run a small benchmark
.\.venv\Scripts\python.exe scripts/run.py smoke --backend cpu

# Run the main comparison or variable-length experiment
.\.venv\Scripts\python.exe scripts/run.py main --backend cpu
.\.venv\Scripts\python.exe scripts/run.py workload --backend cpu
```

For a fresh Windows checkout, install Python 3.12 and create an environment first:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

## NVIDIA GPU execution

The current measurements use CPU execution. NVIDIA GPU execution requires a working Linux or WSL2 environment with GPU access. The GPU setup path is included but has not been validated on this machine.

From the project directory in Linux/WSL2:

```bash
bash scripts/setup_linux.sh
bash scripts/run_linux.sh gpu-test
bash scripts/run_linux.sh main gpu
```

## Code layout

- `src/jax_transformer_lab/`: transformer implementation, execution variants, benchmark CLI, and profiling tools.
- `configs/`: experiment settings, shapes, precision, and bucket boundaries.
- `tests/`: numerical correctness, padding, and CLI checks.
- `scripts/`: environment setup and experiment runners.

Experiments use synthetic inputs and random weights. Generated results, profiles, reports, local environments, and caches stay local and are excluded from version control.
