"""GPU entry point: selects .venv-linux automatically, even from system python3/Windows."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if sys.platform == "win32":
        return subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", ".venv-linux/bin/python", "scripts/run_gpu.py", *sys.argv[1:]],
            cwd=root,
            check=False,
        ).returncode
    parser = argparse.ArgumentParser(description="Local RTX GPU experiments; shell activation is optional")
    parser.add_argument(
        "action",
        choices=[
            "doctor",
            "test",
            "smoke",
            "pilot",
            "main",
            "memory",
            "confirm",
            "workload",
            "refine",
            "profile",
            "report",
            "full",
        ],
    )
    parser.add_argument("--output", default="results/gpu_local", help="Root for local GPU artifacts")
    args = parser.parse_args()
    python = root / ".venv-linux/bin/python"
    if not python.exists():
        parser.error("Run bash scripts/setup_linux.sh first")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actions = (
        [
            "doctor",
            "test",
            "smoke",
            "pilot",
            "main",
            "profile",
            "memory",
            "confirm",
            "workload",
            "refine",
            "report",
        ]
        if args.action == "full"
        else [args.action]
    )
    base = [str(python), "-m", "jax_transformer_lab.cli"]
    for action in actions:
        if action == "doctor":
            commands = [base + ["doctor", "--backend", "gpu", "--output", str(output / "environment.json")]]
        elif action == "test":
            import os

            env = os.environ.copy() | {"JAX_PLATFORMS": "cuda"}
            subprocess.run(
                [str(python), "-m", "pytest", "-q", "-m", "gpu", "--basetemp", str(output / "test_tmp")],
                cwd=root,
                check=True,
                env=env,
            )
            continue
        elif action == "profile":
            commands = [
                base
                + [
                    "profile",
                    "--backend",
                    "gpu",
                    "--variant",
                    variant,
                    "--dtype",
                    "bf16",
                    "--sequence",
                    "1024",
                    "--hidden",
                    "256",
                    "--heads",
                    "4",
                    "--batch",
                    "1",
                    "--output",
                    str(output / "profiles" / variant),
                ]
                for variant in ("components", "whole", "packed", "cudnn")
            ]
        elif action == "report":
            commands = [
                base + ["gpu-report", "--results", str(output), "--output", "reports/gpu_findings.md"]
            ]
        else:
            config = (
                f"configs/{action}.json" if action in ("workload", "refine") else f"configs/gpu_{action}.json"
            )
            commands = [
                base + ["benchmark", "--backend", "gpu", "--config", config, "--output", str(output / action)]
            ]
        for command in commands:
            print(f"GPU {action}: {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
