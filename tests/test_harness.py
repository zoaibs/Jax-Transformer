import json
import os
import subprocess
import sys

import numpy as np
import pytest

from jax_transformer_lab.bucketing import make_batches
from jax_transformer_lab.util import error_metrics, source_fingerprint, stats, write_json
from jax_transformer_lab.workloads import batch_inputs, request_lengths


def test_batches_preserve_trace_and_partial_shape():
    batches = make_batches([5, 6, 3, 9, 4], 2, [4, 8, 16])
    assert [b.shape for b in batches] == [(2, 8), (2, 16), (1, 4)]
    assert [i for b in batches for i in b.indices] == list(range(5))
    assert sum(b.useful_tokens for b in batches) == 27
    assert sum(b.processed_tokens for b in batches) == 52
    assert batches[-1].lengths == (4,)


@pytest.mark.parametrize(
    "lengths,size,buckets",
    [
        ([], 2, None),
        ([0], 2, None),
        ([1], 0, None),
        ([8], 1, [4]),
        ([2], 1, [4, 4]),
        ([2], 1, [8, 4]),
        ([2], 1, []),
    ],
)
def test_invalid_batch_policy(lengths, size, buckets):
    with pytest.raises(ValueError):
        make_batches(lengths, size, buckets)


def test_same_requests_have_same_tokens_regardless_of_padding():
    lengths = request_lengths(count=7, minimum=3, maximum=7)
    exact = make_batches(lengths, 2)
    bucket = make_batches(lengths, 2, [8])
    for a, b in zip(exact, bucket, strict=True):
        ax, am = batch_inputs(a, 16, 7)
        bx, bm = batch_inputs(b, 16, 7)
        np.testing.assert_array_equal(ax[am], bx[bm])


def test_metrics_and_serialization(tmp_path):
    summary = stats([1, 2, 3, 4, 5])
    assert summary["median_ms"] == 3
    assert summary["count"] == 5
    assert summary["p95_ms"] == pytest.approx(4.8)
    write_json(tmp_path / "data.json", summary)
    assert json.loads((tmp_path / "data.json").read_text()) == summary
    for invalid in ([], [float("nan")], [-1]):
        with pytest.raises(ValueError):
            stats(invalid)
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert len(source_fingerprint()) == 64
    assert not error_metrics([float("nan")], [0], "fp32")["passed"]


def test_cli_end_to_end(tmp_path):
    config = {
        "backend": "cpu",
        "models": [{"hidden": 16, "heads": 4}],
        "shapes": [{"batch": 1, "sequence": 8}],
        "variants": ["packed"],
        "repeats": 3,
        "warmup": 1,
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jax_transformer_lab.cli",
            "benchmark",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    path = next(
        p
        for p in (tmp_path / "run").glob("*.json")
        if not p.name.endswith("spec.json") and p.name != "manifest.json"
    )
    record = json.loads(path.read_text())
    assert record["status"] == "ok"
    assert record["environment"]["devices"][0]["platform"] == "cpu"
    assert record["warm"]["count"] == 3
    assert record["timing"]["compile_ms"] > 0
    assert record["validation"]["passed"]


def test_workload_end_to_end(tmp_path):
    config = {
        "backend": "cpu",
        "workloads": [
            {
                "name": "test",
                "model": {"hidden": 16, "heads": 4},
                "distribution": {"count": 5, "minimum": 3, "maximum": 7, "seed": 42},
                "batch_size": 2,
                "buckets": [8],
                "repeats": 2,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jax_transformer_lab.cli",
            "benchmark",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((tmp_path / "run" / "000_workload_test.json").read_text())
    assert record["observed_explicit_compile_calls"] == 2  # final partial batch specializes separately
    assert sum(record["lengths"]) == record["useful_tokens"]
    assert record["processed_tokens"] == 5 * 8
    assert all(v["passed"] for v in record["validations"])
    assert len(record["warm_compute_samples_ms"]) == 2


def test_source_snapshot_rejects_files_changing_during_capture(tmp_path, monkeypatch):
    from jax_transformer_lab import benchmark

    config = tmp_path / "config.json"
    write_json(
        config, {"backend": "cpu", "models": [{"hidden": 16}], "shapes": [{"batch": 1, "sequence": 8}]}
    )
    fingerprints = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(benchmark, "source_fingerprint", lambda: next(fingerprints))
    with pytest.raises(RuntimeError, match="Source changed while capturing"):
        benchmark.run_suite(config, tmp_path / "unstable")


def test_cli_backend_overrides_each_job_kind(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from jax_transformer_lab import benchmark

    item = {
        "backend": "cpu",
        "model": {"hidden": 16, "heads": 4, "dtype": "fp32"},
        "batch": 1,
        "sequence": 8,
        "variant": "whole",
    }
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "backend": "cpu",
            "cases": [item],
            "memory": [item],
            "workloads": [{"name": "override", "backend": "cpu"}],
        },
    )
    selected = []

    def mock_worker(command, **kwargs):
        spec = json.loads(open(command[command.index("--spec") + 1], encoding="utf-8").read())
        selected.append(spec["backend"])
        write_json(command[command.index("--output") + 1], {"status": "ok"})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", mock_worker)
    benchmark.run_suite(config, tmp_path / "run", backend="gpu")
    assert selected == ["gpu", "gpu", "gpu"]
