import pytest

from jax_transformer_lab.gpu_reporting import comparisons, records, summarize_pairs
from jax_transformer_lab.util import write_json


def test_gpu_report_rejects_cpu_measurements(tmp_path):
    write_json(tmp_path / "main/manifest.json", {"jobs": [{"result": "cpu.json"}]})
    write_json(
        tmp_path / "main/cpu.json",
        {
            "status": "ok",
            "kind": "case",
            "spec": {"backend": "cpu"},
            "environment": {"devices": [{"platform": "cpu"}]},
        },
    )
    with pytest.raises(ValueError, match="CPU result"):
        records(tmp_path)


def test_confirmation_pairs_stay_within_the_same_trial():
    rows = []
    for trial, baseline, candidate in [(1, 2.0, 1.0), (2, 9.0, 3.0)]:
        for name, median in [("packed", baseline), ("cudnn", candidate)]:
            rows.append(
                {
                    "spec": {
                        "variant": name,
                        "trial": trial,
                        "model": {"hidden": 256, "heads": 4, "dtype": "bf16"},
                        "batch": 1,
                        "sequence": 1024,
                    },
                    "warm": {"median_ms": median},
                }
            )
    assert [pair[2] for pair in comparisons(rows, "packed", "cudnn")] == [2.0, 3.0]


def test_report_ignores_cli_artifacts_created_by_pytest(tmp_path):
    write_json(tmp_path / "test_tmp/run/manifest.json", {"jobs": [{"result": "fixture.json"}]})
    write_json(
        tmp_path / "test_tmp/run/fixture.json",
        {"status": "ok", "spec": {"backend": "cpu"}, "environment": {"devices": [{"platform": "cpu"}]}},
    )
    assert records(tmp_path) == []


def test_queued_throughput_ratio_and_latency_ratio_use_correct_direction(tmp_path):
    def row(name, latency, queued):
        return {
            "_path": tmp_path / f"{name}.json",
            "spec": {
                "variant": name,
                "batch": 1,
                "sequence": 1024,
                "model": {"hidden": 256, "heads": 4, "dtype": "bf16"},
            },
            "warm": {"median_ms": latency},
            "queued": {"depth": 8, "per_batch": {"median_ms": queued}},
        }

    baseline, candidate = row("components", 2.0, 1.0), row("whole", 1.0, 0.25)
    result = summarize_pairs([(baseline, candidate, 2.0)], tmp_path)
    assert result["speedup"] == 2.0
    assert result["queued_throughput_ratio"] == 4.0
    assert result["candidate_queued_tokens_per_second"] == 4_096_000
