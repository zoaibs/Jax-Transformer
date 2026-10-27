"""Isolated cold processes; synchronized host-observed execution timings in ms."""

import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from .util import ROOT, environment, error_metrics, now, source_fingerprint, stats, write_json


def configure_backend(backend):
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "gpu" and sys.platform == "win32":
        raise RuntimeError(
            "NVIDIA JAX requires Linux/WSL2; native Windows has no supported CUDA backend. "
            "See docs/environment.md. GPU results were not produced."
        )
    os.environ["JAX_PLATFORMS"] = "cuda" if backend == "gpu" else "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def synchronize(value):
    import jax

    return jax.block_until_ready(value)


def timed(fn):
    start = time.perf_counter_ns()
    output = synchronize(fn())
    return output, (time.perf_counter_ns() - start) / 1e6


def allocator_snapshot():
    import jax

    try:
        data = jax.devices()[0].memory_stats()
        return {k: int(v) for k, v in data.items()} if data else None
    except (AttributeError, RuntimeError):
        return None


def executable_memory(compiled):
    if compiled is None:
        return None
    try:
        data = compiled.memory_analysis()
        keys = (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        result = {k: int(getattr(data, k)) for k in keys if hasattr(data, k)}
        result["estimated_buffer_bytes"] = (
            result.get("argument_size_in_bytes", 0)
            + result.get("output_size_in_bytes", 0)
            + result.get("temp_size_in_bytes", 0)
            - result.get("alias_size_in_bytes", 0)
        )
        return result
    except (AttributeError, RuntimeError) as exc:
        return {"unavailable": str(exc)}


def worker_case(spec, output):
    configure_backend(spec["backend"])
    import jax
    import jax.numpy as jnp

    from .model import ModelConfig, device_params, host_params
    from .variants import make_variant
    from .workloads import static_inputs

    config = ModelConfig(**spec["model"])
    device = jax.devices()[0]
    if spec["backend"] == "gpu" and device.platform != "gpu":
        raise RuntimeError("GPU benchmark requested, but selected device is not GPU")
    host = host_params(config, spec.get("seed", 0))
    hx, hm = static_inputs(spec["batch"], spec["sequence"], config.hidden, spec.get("input_seed", 7))

    # Backend initialization and host generation are outside the transfer measurement.
    def transfer():
        return device_params(host, config), jnp.asarray(hx, config.compute_dtype), jax.device_put(hm)

    (params, x, mask), transfer_ms = timed(transfer)
    fn = make_variant(spec["variant"], config)
    memory_before = allocator_snapshot()
    stage = {"trace_ms": None, "lower_ms": None, "compile_ms": None}
    compiled = None
    if spec["variant"] in ("whole", "packed", "cudnn"):
        start = time.perf_counter_ns()
        traced = fn.trace(params, x, mask)
        stage["trace_ms"] = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        lowered = traced.lower()
        stage["lower_ms"] = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        compiled = lowered.compile()
        stage["compile_ms"] = (time.perf_counter_ns() - start) / 1e6
        run = lambda: compiled(params, x, mask)
        result, first_execution_ms = timed(run)
        first_call_ms = sum(stage.values()) + first_execution_ms
        if spec.get("save_hlo"):
            path = Path(output).with_suffix(".optimized_hlo.txt")
            path.write_text(compiled.as_text(), encoding="utf-8")
    else:
        run = lambda: fn(params, x, mask)
        result, first_call_ms = timed(run)
        first_execution_ms = None  # first call includes primitive/component compilation
    memory_after_first = allocator_snapshot()

    # Validate against the whole-block FP32 path using the unquantized master weights.
    # Occurs AFTER the cold measurement and BEFORE warmups.
    fp_config = ModelConfig(**(config.to_dict() | {"dtype": "fp32"}))
    fp_params = device_params(host, fp_config)
    expected = synchronize(make_variant("whole", fp_config)(fp_params, jnp.asarray(hx), mask))
    validation = error_metrics(result, expected, config.dtype)
    if not validation["passed"]:
        raise ValueError(f"Numerical validation failed: {validation}")
    del fp_params, expected
    for _ in range(spec.get("warmup", 5)):
        timed(run)
    samples = []
    group_size = spec.get("group_size", 1)
    # Each call synchronizes: amortizes timer bookkeeping, not Python dispatch/sync overhead.
    for _ in range(spec.get("repeats", 30)):
        start = time.perf_counter_ns()
        for _ in range(group_size):
            result = synchronize(run())
        samples.append((time.perf_counter_ns() - start) / 1e6 / group_size)
    summary = stats(samples)
    tokens = spec["batch"] * spec["sequence"]
    metadata = environment()
    record = {
        "status": "ok",
        "kind": "case",
        "spec": spec,
        "environment": metadata,
        "timing": stage
        | {
            "host_to_device_ms": transfer_ms,
            "first_execution_ms": first_execution_ms,
            "cold_path_sum_ms": first_call_ms,
        },
        "samples_ms": samples,
        "warm": summary,
        "group_size": group_size,
        "tokens_per_second": tokens / (summary["median_ms"] / 1000),
        "validation": validation,
        "executable_memory": executable_memory(compiled),
        "allocator_before": memory_before,
        "allocator_after_first": memory_after_first,
        "allocator_after_warm": allocator_snapshot(),
        "memory_notes": "Executable memory is a static compiler estimate, not process VRAM or runtime peak. "
        "Allocator snapshots are process-wide and include validation allocations after first.",
    }
    write_json(output, record)
    return record


def worker_workload(spec, output):
    configure_backend(spec["backend"])
    import jax
    import jax.numpy as jnp

    from .bucketing import make_batches
    from .model import ModelConfig, device_params, host_params
    from .variants import make_variant
    from .workloads import batch_inputs, request_lengths

    config = ModelConfig(**spec["model"])
    lengths = request_lengths(**spec["distribution"])
    batches = make_batches(lengths, spec["batch_size"], spec.get("buckets"))
    host = host_params(config, spec.get("seed", 0))
    params, params_transfer_ms = timed(lambda: device_params(host, config))
    host_batches = [batch_inputs(b, config.hidden, spec.get("input_seed", 7)) for b in batches]
    fn = make_variant(spec.get("variant", "packed"), config)
    compile_events = []
    executables = {}
    batch_records = []
    # Cold workload includes transfer, per-new-shape staging/compilation, execution and dispatch.
    # Excludes process/JAX initialization, host generation, parameter upload, and validation.
    start = time.perf_counter_ns()
    for index, (batch, (hx, hm)) in enumerate(zip(batches, host_batches, strict=True)):
        (x, mask), transfer_ms = timed(lambda: (jnp.asarray(hx, config.compute_dtype), jax.device_put(hm)))
        key = batch.shape
        if key not in executables:
            t = time.perf_counter_ns()
            traced = fn.trace(params, x, mask)
            t1 = time.perf_counter_ns()
            lowered = traced.lower()
            t2 = time.perf_counter_ns()
            executable = lowered.compile()
            t3 = time.perf_counter_ns()
            executables[key] = executable
            compile_events.append(
                {
                    "batch_index": index,
                    "shape": key,
                    "trace_ms": (t1 - t) / 1e6,
                    "lower_ms": (t2 - t1) / 1e6,
                    "compile_ms": (t3 - t2) / 1e6,
                }
            )
        _, execution_ms = timed(lambda: executables[key](params, x, mask))
        batch_records.append(
            batch.to_dict() | {"transfer_ms": transfer_ms, "cold_execution_ms": execution_ms}
        )
    cold_ms = (time.perf_counter_ns() - start) / 1e6

    # Prepare a device-resident copy of the exact same request trace for warm compute-only timing.
    resident = [
        synchronize((jnp.asarray(hx, config.compute_dtype), jax.device_put(hm))) for hx, hm in host_batches
    ]
    for batch, (x, mask) in zip(batches, resident, strict=True):
        synchronize(executables[batch.shape](params, x, mask))
    warm_workload_samples = []
    warm_batch_samples = [[] for _ in batches]
    warm_e2e_samples = []
    for _ in range(spec.get("repeats", 5)):
        start = time.perf_counter_ns()
        for i, (batch, (x, mask)) in enumerate(zip(batches, resident, strict=True)):
            _, elapsed = timed(lambda: executables[batch.shape](params, x, mask))
            warm_batch_samples[i].append(elapsed)
        warm_workload_samples.append((time.perf_counter_ns() - start) / 1e6)
    for _ in range(spec.get("repeats", 5)):
        start = time.perf_counter_ns()
        for batch, (hx, hm) in zip(batches, host_batches, strict=True):
            x, mask = synchronize((jnp.asarray(hx, config.compute_dtype), jax.device_put(hm)))
            synchronize(executables[batch.shape](params, x, mask))
        warm_e2e_samples.append((time.perf_counter_ns() - start) / 1e6)

    # Validate every distinct shape against exact-length FP32 output AFTER all measured timings.
    fp_config = ModelConfig(**(config.to_dict() | {"dtype": "fp32"}))
    fp_params = device_params(host, fp_config)
    reference = make_variant("whole", fp_config)
    validations = []
    for batch, (hx, hm), (x, mask) in zip(batches, host_batches, resident, strict=True):
        n = max(batch.lengths)
        expected = synchronize(reference(fp_params, jnp.asarray(hx[:, :n]), jnp.asarray(hm[:, :n])))
        actual = synchronize(executables[batch.shape](params, x, mask))
        validations.append(
            error_metrics(
                np.asarray(actual, np.float32)[:, :n][hm[:, :n]],
                np.asarray(expected, np.float32)[hm[:, :n]],
                config.dtype,
            )
        )
    if not all(v["passed"] for v in validations):
        raise ValueError(f"Workload validation failed: {validations}")
    useful = sum(lengths)
    processed = sum(b.processed_tokens for b in batches)
    warm = stats(warm_workload_samples)
    record = {
        "status": "ok",
        "kind": "workload",
        "spec": spec,
        "environment": environment(),
        "lengths": lengths,
        "batches": batch_records,
        "compile_events": compile_events,
        "observed_explicit_compile_calls": len(compile_events),
        "compile_count_definition": "Explicit lower().compile() calls, one per distinct (batch,length). "
        "Does not count internal primitive/library compilations.",
        "distinct_shapes": [list(k) for k in executables],
        "useful_tokens": useful,
        "processed_tokens": processed,
        "padding_overhead_fraction": (processed - useful) / useful,
        "parameter_transfer_ms": params_transfer_ms,
        "cold_workload_ms": cold_ms,
        "warm_compute_samples_ms": warm_workload_samples,
        "warm_compute": warm,
        "warm_e2e_samples_ms": warm_e2e_samples,
        "warm_e2e": stats(warm_e2e_samples),
        "warm_batch_samples_ms": warm_batch_samples,
        "warm_useful_tokens_per_second": useful / (warm["median_ms"] / 1000),
        "cold_useful_tokens_per_second": useful / (cold_ms / 1000),
        "validations": validations,
    }
    write_json(output, record)
    return record


def run_suite(config_path, output_dir, backend=None, repeats=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Immutable measurement-time source snapshot; supports reproduction after later edits.
    fingerprint = source_fingerprint()
    with zipfile.ZipFile(output_dir / f"source_{fingerprint[:12]}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "configs", "scripts", "tests"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT))
        for name in ("pyproject.toml", "requirements-windows.lock.txt"):
            if (ROOT / name).exists():
                archive.write(ROOT / name, name)
    jobs = []
    for model in config.get("models", []):
        for shape in config.get("shapes", []):
            for dtype in config.get("dtypes", ["fp32"]):
                for variant in config.get("variants", ["whole", "packed"]):
                    jobs.append(
                        {
                            "kind": "case",
                            "model": model | {"dtype": dtype},
                            **shape,
                            "variant": variant,
                            "backend": backend or config["backend"],
                            "repeats": repeats or config.get("repeats", 30),
                            "warmup": config.get("warmup", 5),
                            "group_size": config.get("group_size", 1),
                            "seed": config.get("seed", 0),
                            "save_hlo": config.get("save_hlo", False),
                        }
                    )
    for workload in config.get("workloads", []):
        jobs.append({"kind": "workload", "backend": backend or config["backend"], **workload})
    manifest = {"started": now(), "config": config, "source_sha256": source_fingerprint(), "jobs": []}
    write_json(output_dir / "manifest.json", manifest)
    for i, job in enumerate(jobs):
        tag = f"{i:03d}_{job['kind']}"
        if job["kind"] == "case":
            tag += f"_{job['variant']}_{job['model']['dtype']}_b{job['batch']}_s{job['sequence']}"
        else:
            tag += f"_{job.get('name', 'trace')}"
        spec_path, result_path = output_dir / f"{tag}.spec.json", output_dir / f"{tag}.json"
        write_json(spec_path, job)
        command = [
            sys.executable,
            "-m",
            "jax_transformer_lab.cli",
            "_worker",
            "--spec",
            str(spec_path.resolve()),
            "--output",
            str(result_path.resolve()),
        ]
        print(f"[{i + 1}/{len(jobs)}] {tag}", flush=True)
        start = time.monotonic()
        log_path = output_dir / f"{tag}.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config.get("timeout_seconds", 600),
                    env=os.environ.copy(),
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
                log.write("\nWorker timed out; process terminated.\n")
        if not result_path.exists():
            write_json(
                result_path,
                {
                    "status": "error",
                    "kind": job["kind"],
                    "spec": job,
                    "returncode": returncode,
                    "log": str(log_path),
                    "error": log_path.read_text(encoding="utf-8")[-4000:],
                },
            )
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if returncode and record.get("status") == "ok":
            record["status"] = "error"
            record["error"] = f"Worker exit code {returncode}"
            write_json(result_path, record)
        manifest["jobs"].append(
            {
                "result": result_path.name,
                "status": record["status"],
                "elapsed_seconds": time.monotonic() - start,
                "command": command,
            }
        )
        write_json(output_dir / "manifest.json", manifest)
    manifest["finished"] = now()
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def worker(spec_path, output):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    try:
        return worker_case(spec, output) if spec["kind"] == "case" else worker_workload(spec, output)
    except Exception as exc:
        write_json(
            output,
            {
                "status": "error",
                "kind": spec["kind"],
                "spec": spec,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
