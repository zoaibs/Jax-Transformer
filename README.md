# JAX Transformer Optimization

A JAX/XLA transformer optimization toolkit that implements masked self-attention, LayerNorm, residual connections, and a GELU feed-forward network. It compares compilation boundaries, packed QKV projections, FP32/BF16 precision, cuDNN fused attention, and sequence-length bucketing on CPU and NVIDIA GPU.

Sequence-length bucketing achieved **3.46x faster cold-start workload completion** on a synthetic CPU workload, reducing shape-specific compilations from **13 to 3**.

On the **RTX 5070 Laptop GPU**, whole-block JIT achieved **2.61x higher queued inference throughput** than component-level JIT across three fresh-process comparisons. This used BF16 model storage, batch size 1, 1024 tokens, hidden width 256, and eight batches queued. Synchronized batch latency decreased from **2.35 ms to 1.80 ms**.

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

The 3.46x result above uses CPU execution. NVIDIA GPU access and the transformer GPU correctness test have also been verified on this machine using Ubuntu on WSL2. GPU performance measurements can be run separately using the commands below.

From the project directory in Ubuntu/WSL2, these commands automatically use `.venv-linux`. Your system `python3` does not need JAX installed, and shell activation is optional:

```bash
bash scripts/setup_linux.sh
python3 scripts/run_gpu.py doctor
python3 scripts/run_gpu.py test
python3 scripts/run_gpu.py smoke
python3 scripts/run_gpu.py full
```

To use plain `python` interactively, run `source .venv-linux/bin/activate` first. From PowerShell, `.\.venv\Scripts\python.exe scripts/run_gpu.py smoke` launches the GPU runner through Ubuntu automatically.

GPU runs measure synchronized latency, queued throughput, numerical error, and separately scoped memory usage. Generated files stay in the ignored `results/` and `reports/` folders. Use `--output results/my_gpu_run` to select another GPU output directory.

If Windows Application Control blocks a JAX DLL, CPU tests can also run in Ubuntu with `bash scripts/run_linux.sh test`.

## Code layout

- `src/jax_transformer_lab/`: transformer implementation, execution variants, benchmark CLI, and profiling tools.
- `configs/`: experiment settings, shapes, precision, and bucket boundaries.
- `tests/`: numerical correctness, padding, and CLI checks.
- `scripts/`: environment setup and experiment runners.

Experiments use synthetic inputs and random weights. Generated results, profiles, reports, local environments, and caches stay local and are excluded from version control.
