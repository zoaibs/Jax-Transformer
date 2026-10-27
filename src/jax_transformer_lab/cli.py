import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible JAX/XLA transformer experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser(
        "doctor", help="Inspect environment and execute a synchronized device computation"
    )
    doctor.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    doctor.add_argument("--output", default="results/environment.json")
    bench = sub.add_parser("benchmark", help="Run config cases in fresh sequential processes")
    bench.add_argument("--config", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--backend", choices=["cpu", "gpu"])
    bench.add_argument("--repeats", type=int)
    work = sub.add_parser("_worker", help=argparse.SUPPRESS)
    work.add_argument("--spec", required=True)
    work.add_argument("--output", required=True)
    profile = sub.add_parser("profile", help="Capture local profiler trace and optimized HLO")
    profile.add_argument("--output", required=True)
    profile.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    profile.add_argument(
        "--variant", choices=["reference", "components", "whole", "packed", "cudnn"], default="packed"
    )
    profile.add_argument("--sequence", type=int, default=256)
    profile.add_argument("--hidden", type=int, default=256)
    profile.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    report = sub.add_parser("report", help="Regenerate figures and technical report from saved results")
    report.add_argument("--results", default="results")
    report.add_argument("--output", default="reports")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .benchmark import configure_backend

        configure_backend(args.backend)
        import jax
        import jax.numpy as jnp

        from .util import environment, write_json

        devices = jax.devices()
        if args.backend == "gpu" and not any(d.platform == "gpu" for d in devices):
            raise RuntimeError("GPU required; JAX did not initialize a GPU backend")
        a = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
        result = jax.block_until_ready(jax.jit(lambda x: x @ x.T)(a))
        data = environment() | {
            "verified_result_sum": float(result.sum()),
            "result_devices": [str(d) for d in result.devices()],
        }
        write_json(args.output, data)
        print(json.dumps(data, indent=2))
    elif args.command == "benchmark":
        from .benchmark import run_suite

        manifest = run_suite(args.config, args.output, args.backend, args.repeats)
        failed = sum(j["status"] != "ok" for j in manifest["jobs"])
        print(f"Completed {len(manifest['jobs'])} jobs; {failed} failed. Results: {args.output}")
        return 1 if failed else 0
    elif args.command == "_worker":
        from .benchmark import worker

        worker(args.spec, args.output)
    elif args.command == "profile":
        from .profiling import capture

        print(
            json.dumps(
                capture(args.output, args.backend, args.variant, args.sequence, args.hidden, args.dtype),
                indent=2,
            )
        )
    elif args.command == "report":
        from .reporting import generate

        print(generate(Path(args.results), Path(args.output)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
