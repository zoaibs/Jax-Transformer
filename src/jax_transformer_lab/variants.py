"""Compilation boundaries and a packed-QKV optimization."""

from functools import partial

import jax
import jax.numpy as jnp

from .model import attention, dense, forward, layer_norm, mlp, project

VARIANTS = ("reference", "components", "whole", "packed", "cudnn")


def make_variant(name, config):
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")
    if name == "cudnn" and config.dtype != "bf16":
        raise ValueError("The cuDNN attention variant requires bf16; use whole or packed for fp32")
    if name == "reference":
        return partial(forward, config=config)
    if name in ("whole", "packed", "cudnn"):
        return jax.jit(
            partial(
                forward,
                config=config,
                fused=name != "whole",
                attention_backend="cudnn" if name == "cudnn" else None,
            )
        )

    norm = jax.jit(partial(layer_norm, config=config))
    projection = jax.jit(partial(project, config=config))
    attend, linear, feed_forward = jax.jit(attention), jax.jit(dense), jax.jit(mlp)

    @jax.jit
    def residual(x, z, mask):
        return jnp.where(
            mask[..., None], (x.astype(jnp.float32) + z.astype(jnp.float32)).astype(config.compute_dtype), 0
        )

    def composed(params, x, mask):
        for p in params:
            z = norm(x, p["ln1_g"], p["ln1_b"])
            q, k, v = projection(z, p)
            z = linear(attend(q, k, v, mask), p["wo"], p["bo"])
            x = residual(x, z, mask)
            z = feed_forward(norm(x, p["ln2_g"], p["ln2_b"]), p)
            x = residual(x, z, mask)
        return x

    return composed
