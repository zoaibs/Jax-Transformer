"""Device-wide NVIDIA snapshots; these are not inference-only memory peaks."""

import csv
import io

from .util import now, run_text

FIELDS = (
    "name",
    "uuid",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "utilization.gpu",
    "memory.used",
    "memory.total",
)


def nvidia_snapshot():
    result = run_text(["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"])
    record = {"timestamp": now(), "scope": "device-wide snapshot outside timed execution", "raw": result}
    if result.get("returncode") == 0:
        rows = list(csv.reader(io.StringIO(result["stdout"])))
        record["devices"] = [
            dict(zip(FIELDS, (v.strip() for v in row), strict=True))
            for row in rows
            if len(row) == len(FIELDS)
        ]
    return record


def memory_description(snapshot):
    """Expose unavailable counters as null rather than inventing substitutes."""
    snapshot = snapshot or {}
    return {
        "bytes_in_use": snapshot.get("bytes_in_use"),
        "peak_bytes_in_use": snapshot.get("peak_bytes_in_use"),
        "bytes_reserved": snapshot.get("bytes_reserved"),
        "peak_bytes_reserved": snapshot.get("peak_bytes_reserved"),
        "peak_scope": "since backend/process initialization, including compilation and autotuning; not inference-only",
        "raw_counters": snapshot,
    }
