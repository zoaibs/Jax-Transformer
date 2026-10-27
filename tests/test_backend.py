import json
import subprocess
import sys

import pytest

from jax_transformer_lab.benchmark import configure_backend
from jax_transformer_lab.util import write_json


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported backend"):
        configure_backend("silent-fallback")


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows CUDA rejection path")
def test_gpu_worker_records_failure_instead_of_cpu_result(tmp_path):
    spec = {
        "kind": "case",
        "backend": "gpu",
        "variant": "packed",
        "model": {"hidden": 16, "heads": 4},
        "batch": 1,
        "sequence": 8,
    }
    spec_path, output = tmp_path / "spec.json", tmp_path / "result.json"
    write_json(spec_path, spec)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jax_transformer_lab.cli",
            "_worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0
    data = json.loads(output.read_text())
    assert data["status"] == "error"
    assert "native Windows" in data["error"]
    assert "warm" not in data
