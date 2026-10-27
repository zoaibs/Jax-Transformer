#!/usr/bin/env bash
# No activation or system Python needed: always use this project's Linux environment.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_path="$project_dir/.venv-linux/bin/python"
if [[ ! -x "$python_path" ]]; then
  printf 'Linux environment missing. Run: bash scripts/setup_linux.sh\n' >&2
  exit 2
fi
cd "$project_dir"
exec "$python_path" -m jax_transformer_lab.cli "$@"
