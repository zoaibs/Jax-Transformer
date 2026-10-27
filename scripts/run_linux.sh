#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-smoke}"
backend="${2:-gpu}"
python_path="$PWD/.venv-linux/bin/python"
lab() { "$python_path" -m jax_transformer_lab.cli "$@"; }
case "$action" in
  test) JAX_PLATFORMS=cpu "$python_path" -m pytest -q ;;
  gpu-test) JAX_PLATFORMS=cuda "$python_path" -m pytest -q -m gpu ;;
  doctor) lab doctor --backend "$backend" --output "results/environment_${backend}.json" ;;
  report) lab report --results results --output reports ;;
  profile)
    for variant in whole packed components; do
      lab profile --backend "$backend" --variant "$variant" --output "results/profile_${backend}_${variant}"
    done ;;
  full)
    bash scripts/run_linux.sh doctor "$backend"
    for phase in pilot main workload refine confirm; do bash scripts/run_linux.sh "$phase" "$backend"; done
    bash scripts/run_linux.sh profile "$backend"
    bash scripts/run_linux.sh report "$backend" ;;
  refine)
    lab benchmark --config configs/refine.json --backend "$backend" --output "results/workload_refine_${backend}" ;;
  smoke|pilot|main|workload|confirm|cudnn)
    lab benchmark --config "configs/${action}.json" --backend "$backend" --output "results/${action}_${backend}" ;;
  *) printf 'Unknown action: %s\n' "$action" >&2; exit 2 ;;
esac

