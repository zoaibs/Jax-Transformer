"""Independently verify archived GPU metrics using only the standard library."""

import argparse
import json
import math
import statistics
from pathlib import Path


def check_stats(samples, summary):
    assert samples and all(math.isfinite(x) and x > 0 for x in samples)
    assert len(samples) == summary["count"]
    assert math.isclose(statistics.median(samples), summary["median_ms"], rel_tol=1e-12)
    ordered = sorted(samples)
    position = 0.95 * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    p95 = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    assert math.isclose(p95, summary["p95_ms"], rel_tol=1e-12)


def audit(root):
    checks, failures, traces = [], [], []
    for path in sorted(root.rglob("manifest.json")):
        if "test_tmp" in path.relative_to(root).parts:
            continue
        manifest = json.loads(path.read_text())
        assert manifest.get("finished"), f"Unfinished suite {path}"
        assert not manifest.get("source_changed_during_suite", False), f"Source drift in {path}"
        assert list(path.parent.glob(f"source_{manifest['source_sha256'][:12]}.zip")), (
            f"Missing source snapshot {path}"
        )
        for job in manifest["jobs"]:
            result_path = path.parent / job["result"]
            r = json.loads(result_path.read_text())
            if r["status"] != "ok":
                failures.append({"path": str(result_path), "error": r.get("error")})
                assert path.parent.name == "baseline_probe", f"Unexpected failed measurement {result_path}"
                continue
            assert r["spec"]["backend"] == "gpu"
            assert {d["platform"] for d in r["environment"]["devices"]} == {"gpu"}
            if r["kind"] == "case":
                check_stats(r["samples_ms"], r["warm"])
                useful = r["spec"]["batch"] * r["spec"]["sequence"]
                assert math.isclose(r["tokens_per_second"], useful / (r["warm"]["median_ms"] / 1000))
                assert r["validation"]["passed"] and r["validation"]["finite"]
                if r.get("queued"):
                    check_stats(r["queued"]["samples_ms_per_batch"], r["queued"]["per_batch"])
            elif r["kind"] == "memory":
                assert "validation" not in r and r["finite"]
                snapshot = r["snapshots"]["after_warm"]
                assert "including compilation" in snapshot["peak_scope"]
                if snapshot["bytes_in_use"] is not None and snapshot["peak_bytes_in_use"] is not None:
                    assert snapshot["peak_bytes_in_use"] >= snapshot["bytes_in_use"]
            else:
                traces.append(r["lengths"])
                useful = sum(r["lengths"])
                processed = sum(len(b["lengths"]) * b["padded_length"] for b in r["batches"])
                shapes = {(len(b["lengths"]), b["padded_length"]) for b in r["batches"]}
                assert useful == r["useful_tokens"] and processed == r["processed_tokens"]
                assert len(shapes) == len(r["compile_events"]) == r["observed_explicit_compile_calls"]
                assert all(v["passed"] for v in r["validations"])
                assert math.isclose(r["padding_overhead_fraction"], (processed - useful) / useful)
                check_stats(r["warm_compute_samples_ms"], r["warm_compute"])
            if r.get("executable_memory") and "estimated_buffer_bytes" in r["executable_memory"]:
                m = r["executable_memory"]
                assert m["estimated_buffer_bytes"] == (
                    m["argument_size_in_bytes"]
                    + m["output_size_in_bytes"]
                    + m["temp_size_in_bytes"]
                    - m["alias_size_in_bytes"]
                )
            checks.append(str(result_path))
    assert checks, "No results found"
    assert not traces or all(trace == traces[0] for trace in traces), "Request traces differ"
    summary = {"verified_records": len(checks), "expected_diagnostic_failures": failures, "files": checks}
    (root / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(checks)} GPU records; {len(failures)} preserved diagnostic failures.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    audit(parser.parse_args().results)
