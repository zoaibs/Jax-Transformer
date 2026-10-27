"""Capture local traces and optimized HLO without launching any network service."""

import gzip
import json
import re
from pathlib import Path

from .benchmark import configure_backend, executable_memory, synchronize
from .util import environment, write_json


def capture(output, backend="gpu", variant="packed", sequence=256, hidden=256, dtype="fp32"):
    configure_backend(backend)
    import jax
    import jax.numpy as jnp

    from .model import ModelConfig, device_params, host_params
    from .variants import make_variant
    from .workloads import static_inputs

    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(hidden=hidden, heads=4, dtype=dtype)
    params = device_params(host_params(cfg), cfg)
    hx, hm = static_inputs(1, sequence, hidden)
    x, mask = synchronize((jnp.asarray(hx, cfg.compute_dtype), jnp.asarray(hm)))
    fn = make_variant(variant, cfg)
    for _ in range(5):
        synchronize(fn(params, x, mask))
    summary = {
        "environment": environment(),
        "variant": variant,
        "model": cfg.to_dict(),
        "batch": 1,
        "sequence": sequence,
    }
    if hasattr(fn, "lower"):
        lowered = fn.lower(params, x, mask)
        lowered.as_text().encode("utf-8")
        (path / "stablehlo.mlir").write_text(lowered.as_text(), encoding="utf-8")
        compiled = lowered.compile()
        hlo = compiled.as_text()
        (path / "optimized_hlo.txt").write_text(hlo, encoding="utf-8")
        summary["executable_memory"] = executable_memory(compiled)
        # Count instruction definitions, not every occurrence of the words in operands/metadata.
        summary["hlo_instruction_counts"] = {
            op: len(re.findall(r"^\s*(?:ROOT\s+)?%[^\n=]+=[^\n]*?\s" + re.escape(op) + r"\(", hlo, re.M))
            for op in ("fusion", "dot", "custom-call", "reduce", "transpose", "copy")
        }
        summary["hlo_note"] = "Textual optimized-HLO instruction definitions; not GPU kernel counts."
    try:
        with jax.profiler.trace(str(path), create_perfetto_link=False, create_perfetto_trace=True):
            for step in range(8):
                with jax.profiler.StepTraceAnnotation("inference", step_num=step):
                    with jax.profiler.TraceAnnotation(f"{variant}_block"):
                        synchronize(fn(params, x, mask))
        summary["trace_status"] = "captured"
    except Exception as exc:
        summary["trace_status"] = "unavailable"
        summary["trace_error"] = repr(exc)
    summaries = []
    for trace in path.rglob("*.trace.json.gz"):
        try:
            with gzip.open(trace, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
            events = data.get("traceEvents", [])
            processes = [
                e.get("args", {}).get("name", "")
                for e in events
                if e.get("ph") == "M" and e.get("name") == "process_name"
            ]
            durations = [e for e in events if e.get("ph") == "X"]
            totals = {}
            for e in durations:
                name = e.get("name", "unknown")
                item = totals.setdefault(name, {"count": 0, "summed_duration_us": 0.0})
                item["count"] += 1
                item["summed_duration_us"] += e.get("dur", 0)
            summaries.append(
                {
                    "file": str(trace.relative_to(path)),
                    "process_names": processes,
                    "duration_events": len(durations),
                    "top_events_by_summed_duration": sorted(
                        totals.items(), key=lambda t: t[1]["summed_duration_us"], reverse=True
                    )[:20],
                    "note": "Nested/overlapping durations cannot be added into elapsed time.",
                }
            )
        except (OSError, ValueError) as exc:
            summaries.append({"file": str(trace), "error": str(exc)})
    summary["traces"] = summaries
    write_json(path / "summary.json", summary)
    return summary
