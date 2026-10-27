#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Linux GPU path; requires working WSL2 GPU passthrough or native Linux.
# Uses isolated .venv-linux; never reuses the Windows environment.
export UV_CACHE_DIR="$PWD/.cache/uv-linux"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python-linux"
mkdir -p .tools/linux
if [[ ! -x .tools/linux/uv ]]; then
  curl -LsSf --max-time 120 https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz -o .tools/linux/uv.tar.gz
  tar -xzf .tools/linux/uv.tar.gz -C .tools/linux --strip-components=1
fi
.tools/linux/uv python install 3.12.15
if [[ ! -f .venv-linux/pyvenv.cfg ]]; then
  .tools/linux/uv venv --python 3.12.15 .venv-linux
fi
.venv-linux/bin/python -c 'import sys; assert sys.platform == "linux", "A Linux Python environment is required"'
if [[ -f requirements-linux.lock.txt ]]; then
  .tools/linux/uv pip sync --python .venv-linux/bin/python requirements-linux.lock.txt
  .tools/linux/uv pip install --python .venv-linux/bin/python --no-deps -e .
else
  .tools/linux/uv pip install --python .venv-linux/bin/python -e '.[test,gpu]' 'jax[cuda13]==0.8.2'
fi
mkdir -p results
.tools/linux/uv pip freeze --python .venv-linux/bin/python > results/requirements-linux-tested.txt
.venv-linux/bin/python -m jax_transformer_lab.cli doctor --backend gpu --output results/environment_gpu.json
printf '\nReady. Run without activation: bash scripts/jaxlab.sh doctor --backend gpu\n'
printf 'Or activate this shell: source .venv-linux/bin/activate\n'

