import json

from jax_transformer_lab.reporting import load_records, matched_speedups
from jax_transformer_lab.util import write_json


def case(variant, median, dtype="fp32", sequence=256, backend="cpu", hidden=256):
    return {
        "status": "ok",
        "kind": "case",
        "spec": {
            "variant": variant,
            "backend": backend,
            "model": {"hidden": hidden, "heads": 4, "dtype": dtype},
            "batch": 1,
            "sequence": sequence,
        },
        "warm": {"median_ms": median},
    }


def test_comparisons_require_matching_shape_dtype_backend_and_model():
    rows = [
        case("whole", 10),
        case("packed", 5),
        case("packed", 1, dtype="bf16"),
        case("packed", 1, sequence=512),
        case("packed", 1, backend="gpu"),
        case("packed", 1, hidden=512),
    ]
    pairs = matched_speedups(rows)
    assert len(pairs) == 1
    assert pairs[0][2] == 2


def test_manifest_excludes_stale_results_and_specs(tmp_path):
    suite = tmp_path / "main_cpu"
    write_json(suite / "manifest.json", {"jobs": [{"result": "current.json"}]})
    write_json(suite / "current.json", case("whole", 10))
    write_json(suite / "stale.json", case("packed", 1))
    write_json(suite / "current.spec.json", case("packed", 1))
    (suite / "broken.json").write_text("not json")
    records = load_records(tmp_path)
    assert [r["_path"].name for r in records] == ["current.json"]
    assert json.loads((suite / "stale.json").read_text())["status"] == "ok"  # no deletion
