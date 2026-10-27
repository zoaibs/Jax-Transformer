"""Check archived measurements independently of the reporting implementation."""

import json
import math
import statistics
import sys
from pathlib import Path


def audit(root):
    checks = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "finished" not in manifest:
            raise ValueError(f"Unfinished suite: {manifest_path}")
        for job in manifest["jobs"]:
            path = manifest_path.parent / job["result"]
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] != "ok":
                checks.append({"path": str(path), "status": "recorded_failure"})
                continue
            expected_device = record["spec"]["backend"]
            platforms = {d["platform"] for d in record["environment"]["devices"]}
            assert platforms == {expected_device}, (path, platforms)
            if record["kind"] == "case":
                samples = record["samples_ms"]
                assert all(math.isfinite(s) and s > 0 for s in samples)
                median = statistics.median(samples)
                assert math.isclose(record["warm"]["median_ms"], median)
                tokens = record["spec"]["batch"] * record["spec"]["sequence"]
                assert math.isclose(record["tokens_per_second"], tokens / (median / 1000))
                assert record["validation"]["passed"]
                assert record["warm"]["count"] == len(samples)
            else:
                useful = sum(record["lengths"])
                processed = sum(len(b["lengths"]) * b["padded_length"] for b in record["batches"])
                shapes = {(len(b["lengths"]), b["padded_length"]) for b in record["batches"]}
                assert useful == record["useful_tokens"]
                assert processed == record["processed_tokens"]
                assert record["observed_explicit_compile_calls"] == len(shapes)
                assert len(record["compile_events"]) == len(shapes)
                assert all(v["passed"] for v in record["validations"])
                assert math.isclose(record["padding_overhead_fraction"], (processed - useful) / useful)
                assert math.isclose(
                    record["warm_compute"]["median_ms"], statistics.median(record["warm_compute_samples_ms"])
                )
            checks.append({"path": str(path), "status": "passed"})
    if not checks:
        raise ValueError("No completed results to audit")
    output = {
        "checked": len(checks),
        "passed": sum(c["status"] == "passed" for c in checks),
        "recorded_failures": sum(c["status"] == "recorded_failure" for c in checks),
        "checks": checks,
    }
    (root / "audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Audited {output['checked']} records: {output['passed']} passed, "
        f"{output['recorded_failures']} explicitly recorded failures."
    )
    return output


if __name__ == "__main__":
    audit(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results"))
