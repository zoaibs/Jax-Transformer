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
.tools/linux/uv venv --python 3.12.15 .venv-linux
.tools/linux/uv pip install --python .venv-linux/bin/python -e '.[test,gpu]' 'jax[cuda13]==0.8.2'
mkdir -p results
.tools/linux/uv pip freeze --python .venv-linux/bin/python > results/requirements-linux-tested.txt
.venv-linux/bin/python -m jax_transformer_lab.cli doctor --backend gpu --output results/environment_gpu.json

