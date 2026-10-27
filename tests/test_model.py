import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_transformer_lab.bucketing import make_batches
from jax_transformer_lab.model import ModelConfig, attention, device_params, host_params, numpy_reference
from jax_transformer_lab.util import error_metrics
from jax_transformer_lab.variants import make_variant
from jax_transformer_lab.workloads import batch_inputs, static_inputs


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hidden=7, heads=2),
        dict(hidden=0),
        dict(heads=0),
        dict(layers=0),
        dict(expansion=0),
        dict(dtype="fp16"),
        dict(epsilon=0),
        dict(epsilon=float("nan")),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("variant", ["reference", "components", "whole", "packed"])
def test_matches_independent_numpy(variant):
    cfg = ModelConfig(hidden=16, heads=4, layers=2)
    host = host_params(cfg)
    batch = make_batches([7, 4], 2, [8])[0]
    hx, mask = batch_inputs(batch, cfg.hidden, 11)
    expected = numpy_reference(host, hx, mask, cfg)
    actual = make_variant(variant, cfg)(device_params(host, cfg), jnp.asarray(hx), jnp.asarray(mask))
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)
    assert np.isfinite(actual).all()
    assert np.count_nonzero(np.asarray(actual)[~mask]) == 0


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("variant", ["reference", "components", "whole", "packed"])
def test_padding_isolation_and_bucket_equivalence(dtype, variant):
    cfg = ModelConfig(hidden=16, heads=4, dtype=dtype)
    params = device_params(host_params(cfg), cfg)
    fn = make_variant(variant, cfg)
    batch = make_batches([5, 3], 2, [8])[0]
    hx, hm = batch_inputs(batch, cfg.hidden, 7)
    x, mask = jnp.asarray(hx, cfg.compute_dtype), jnp.asarray(hm)
    actual = np.asarray(fn(params, x, mask), np.float32)
    exact = np.asarray(fn(params, x[:, :5], mask[:, :5]), np.float32)
    # Altering padded token content cannot influence valid queries.
    poisoned = jnp.where(mask[..., None], x, jnp.asarray(50, cfg.compute_dtype))
    modified = np.asarray(fn(params, poisoned, mask), np.float32)
    np.testing.assert_allclose(actual[hm], modified[hm], atol=1e-5)
    tolerance = 0.04 if dtype == "bf16" else 2e-5
    np.testing.assert_allclose(actual[:, :5][hm[:, :5]], exact[hm[:, :5]], atol=tolerance, rtol=tolerance)


def test_attention_mask_math_and_empty_rows():
    q = jnp.zeros((2, 3, 1, 2))
    k = jnp.zeros_like(q)
    v = jnp.asarray(
        [[[[2.0, 4.0]], [[6.0, 8.0]], [[100.0, 200.0]]], [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]
    )
    mask = jnp.asarray([[True, True, False], [False, False, False]])
    result = np.asarray(attention(q, k, v, mask))
    np.testing.assert_allclose(result[0], np.tile([4, 6], (3, 1)))
    np.testing.assert_array_equal(result[1], 0)
    assert np.isfinite(result).all()


def test_empty_mask_block_is_finite_and_zero():
    cfg = ModelConfig(hidden=16, heads=4)
    params = device_params(host_params(cfg), cfg)
    x = jnp.ones((1, 4, 16))
    result = make_variant("packed", cfg)(params, x, jnp.zeros((1, 4), bool))
    np.testing.assert_array_equal(result, 0)


def test_bf16_error_against_unquantized_fp32():
    fp = ModelConfig(hidden=32, heads=4)
    bf = ModelConfig(hidden=32, heads=4, dtype="bf16")
    hx, hm = static_inputs(2, 16, 32)
    host = host_params(fp)
    expected = make_variant("whole", fp)(device_params(host, fp), jnp.asarray(hx), jnp.asarray(hm))
    actual = make_variant("packed", bf)(
        device_params(host, bf), jnp.asarray(hx, jnp.bfloat16), jnp.asarray(hm)
    )
    error = error_metrics(actual, expected, "bf16")
    assert error["passed"]
    assert error["normalized_rmse"] < 0.02


def test_seed_and_precision_master_weights():
    fp = host_params(ModelConfig(hidden=16))
    bf = host_params(ModelConfig(hidden=16, dtype="bf16"))
    for a, b in zip(jax.tree.leaves(fp), jax.tree.leaves(bf), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.gpu
def test_gpu_execution():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("No JAX GPU backend in selected environment")
    cfg = ModelConfig(hidden=32)
    hx, mask = static_inputs(1, 16, 32)
    out = make_variant("packed", cfg)(
        device_params(host_params(cfg), cfg), jnp.asarray(hx), jnp.asarray(mask)
    )
    assert all(d.platform == "gpu" for d in out.devices())
    assert np.isfinite(out).all()
