"""Cross-platform runner; does not require PowerShell script execution permission."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "doctor",
            "test",
            "smoke",
            "pilot",
            "main",
            "workload",
            "refine",
            "confirm",
            "profile",
            "cudnn",
            "report",
            "full",
        ],
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    actions = (
        ["doctor", "pilot", "main", "workload", "refine", "confirm", "profile", "report"]
        if args.action == "full"
        else [args.action]
    )
    for action in actions:
        base = [sys.executable, "-m", "jax_transformer_lab.cli"]
        if action == "test":
            commands = [[sys.executable, "-m", "pytest", "-q"]]
        elif action == "doctor":
            commands = [
                base
                + [
                    "doctor",
                    "--backend",
                    args.backend,
                    "--output",
                    f"results/environment_{args.backend}.json",
                ]
            ]
        elif action == "report":
            commands = [base + ["report", "--results", "results", "--output", "reports"]]
        elif action == "profile":
            commands = [
                base
                + [
                    "profile",
                    "--backend",
                    args.backend,
                    "--variant",
                    variant,
                    "--output",
                    f"results/profile_{args.backend}_{variant}",
                ]
                for variant in ("whole", "packed", "components")
            ]
        else:
            suffix = "smoke" if action == "smoke" else f"{action}_{args.backend}"
            if action == "refine":
                suffix = f"workload_refine_{args.backend}"
            commands = [
                base
                + [
                    "benchmark",
                    "--config",
                    f"configs/{action}.json",
                    "--backend",
                    args.backend,
                    "--output",
                    f"results/{suffix}",
                ]
            ]
        for command in commands:
            print(f"Running {action}: {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
