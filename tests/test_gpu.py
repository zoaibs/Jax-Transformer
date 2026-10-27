"""Run with JAX_PLATFORMS=cuda pytest -m gpu. Never requires a global Python install."""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_transformer_lab.bucketing import make_batches
from jax_transformer_lab.model import ModelConfig, device_params, host_params
from jax_transformer_lab.util import error_metrics, write_json
from jax_transformer_lab.variants import make_variant
from jax_transformer_lab.workloads import batch_inputs

pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("Requires a JAX GPU backend")


@pytest.mark.parametrize(
    "variant,dtype",
    [("components", "fp32"), ("whole", "fp32"), ("packed", "fp32"), ("packed", "bf16"), ("cudnn", "bf16")],
)
def test_gpu_variants_and_padding(gpu, variant, dtype):
    cfg = ModelConfig(hidden=128, heads=2, dtype=dtype)
    fp = ModelConfig(hidden=128, heads=2)
    host = host_params(cfg)
    batch = make_batches([47, 19], 2, [64])[0]
    hx, hm = batch_inputs(batch, cfg.hidden, 7)
    params = device_params(host, cfg)
    fn = make_variant(variant, cfg)
    x, mask = jnp.asarray(hx, cfg.compute_dtype), jnp.asarray(hm)
    actual = fn(params, x, mask)
    assert all(d.platform == "gpu" for d in actual.devices())
    expected = make_variant("whole", fp)(device_params(host, fp), jnp.asarray(hx), mask)
    metrics = error_metrics(actual, expected, dtype)
    assert metrics["passed"], metrics
    assert metrics["normalized_rmse"] < (0.02 if dtype == "bf16" else 0.001)
    # A constant feature vector would normalize to zero and would not actually
    # exercise isolation through LayerNorm. Use nonconstant padded content.
    garbage = jnp.asarray(np.random.default_rng(99).standard_normal(hx.shape) * 25, cfg.compute_dtype)
    poisoned = jnp.where(mask[..., None], x, garbage)
    changed = np.asarray(fn(params, poisoned, mask), np.float32)
    np.testing.assert_allclose(np.asarray(actual, np.float32)[hm], changed[hm], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(changed[~hm], 0)


def test_cudnn_nonprefix_and_empty_masks(gpu):
    cfg = ModelConfig(hidden=128, heads=2, dtype="bf16")
    params = device_params(host_params(cfg), cfg)
    rng = np.random.default_rng(3)
    hx = rng.standard_normal((2, 64, 128)).astype(np.float32)
    hm = np.zeros((2, 64), bool)
    hm[0, ::2] = True  # Arbitrary holes, rather than only prefix padding.
    fn = make_variant("cudnn", cfg)
    actual = np.asarray(fn(params, jnp.asarray(hx, jnp.bfloat16), jnp.asarray(hm)), np.float32)
    expected = make_variant("packed", cfg)(params, jnp.asarray(hx, jnp.bfloat16), jnp.asarray(hm))
    assert error_metrics(actual, expected, "bf16")["passed"]
    np.testing.assert_array_equal(actual[1], 0)
    assert np.isfinite(actual).all()


def test_gpu_cli_memory_and_smoke(gpu, tmp_path):
    case = {
        "model": {"hidden": 128, "heads": 2, "dtype": "bf16"},
        "batch": 1,
        "sequence": 64,
        "variant": "cudnn",
    }
    spec = {
        "backend": "gpu",
        "cases": [case],
        "memory": [case],
        "repeats": 4,
        "warmup": 2,
        "warmup_seconds": 0.05,
        "group_size": 1,
        "queue_depth": 4,
        "queue_repeats": 3,
    }
    path = tmp_path / "config.json"
    write_json(path, spec)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jax_transformer_lab.cli",
            "benchmark",
            "--config",
            str(path),
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = [
        json.loads(p.read_text())
        for p in (tmp_path / "run").glob("*.json")
        if not p.name.endswith("spec.json") and p.name != "manifest.json"
    ]
    case_record = next(r for r in records if r["kind"] == "case")
    memory = next(r for r in records if r["kind"] == "memory")
    assert case_record["environment"]["devices"][0]["platform"] == "gpu"
    assert case_record["sample_unit"] == "individual synchronized batch call"
    assert len(case_record["queued"]["samples_ms_per_batch"]) == 3
    assert case_record["queued"]["validation"]["passed"]
    assert len(case_record["queued"]["validation"]["outputs"]) == 4
    assert "including compilation" in memory["snapshots"]["after_warm"]["peak_scope"]
    assert memory["finite"]
    assert "validation" not in memory
