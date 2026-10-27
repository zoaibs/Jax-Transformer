"""Pre-LN transformer encoder. True masks denote valid tokens; no dropout."""

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 256
    heads: int = 4
    expansion: int = 4
    layers: int = 1
    dtype: str = "fp32"
    epsilon: float = 1e-5

    def __post_init__(self):
        if min(self.hidden, self.heads, self.expansion, self.layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.dtype not in ("fp32", "bf16"):
            raise ValueError("dtype must be fp32 or bf16")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    @property
    def compute_dtype(self):
        return jnp.float32 if self.dtype == "fp32" else jnp.bfloat16

    def to_dict(self):
        return asdict(self)


def host_params(config: ModelConfig, seed: int = 0):
    """Host FP32 master weights are identical across precision/implementation variants."""
    rng = np.random.default_rng(seed)
    d, f = config.hidden, config.hidden * config.expansion
    blocks = []
    for _ in range(config.layers):

        def weight(shape):
            return (rng.standard_normal(shape) / math.sqrt(shape[0])).astype(np.float32)

        blocks.append(
            {
                "wqkv": weight((d, 3 * d)),
                "bqkv": np.zeros(3 * d, np.float32),
                "wo": weight((d, d)),
                "bo": np.zeros(d, np.float32),
                "w1": weight((d, f)),
                "b1": np.zeros(f, np.float32),
                "w2": weight((f, d)),
                "b2": np.zeros(d, np.float32),
                "ln1_g": np.ones(d, np.float32),
                "ln1_b": np.zeros(d, np.float32),
                "ln2_g": np.ones(d, np.float32),
                "ln2_b": np.zeros(d, np.float32),
            }
        )
    return tuple(blocks)


def device_params(host, config):
    # Norm parameters remain FP32; GEMM weights and biases use compute precision.
    return tuple(
        {
            k: jnp.asarray(v, dtype=jnp.float32 if k.startswith("ln") else config.compute_dtype)
            for k, v in p.items()
        }
        for p in host
    )


def layer_norm(x, gamma, beta, config):
    y = x.astype(jnp.float32)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    return ((y - mean) * jax.lax.rsqrt(variance + config.epsilon) * gamma + beta).astype(config.compute_dtype)


def dense(x, w, b):
    # Explicit FP32 accumulation, default operand precision (no TF32 flags added).
    return (
        jnp.matmul(x, w, precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32)
        + b.astype(jnp.float32)
    ).astype(x.dtype)


def project(x, p, config, fused=False):
    if fused:
        q, k, v = jnp.split(dense(x, p["wqkv"], p["bqkv"]), 3, axis=-1)
    else:
        weights, biases = jnp.split(p["wqkv"], 3, axis=-1), jnp.split(p["bqkv"], 3)
        q, k, v = [dense(x, w, b) for w, b in zip(weights, biases, strict=True)]
    shape = (*x.shape[:2], config.heads, config.hidden // config.heads)
    return q.reshape(shape), k.reshape(shape), v.reshape(shape)


def attention(q, k, v, mask):
    # FP32 logits, normalization, and value accumulation. Storage/output follows q.
    logits = jnp.einsum(
        "bthd,bshd->bhts", q.astype(jnp.float32), k.astype(jnp.float32), precision=jax.lax.Precision.DEFAULT
    ) / math.sqrt(q.shape[-1])
    valid = mask[:, None, None, :]
    logits = jnp.where(valid, logits, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(logits, axis=-1)
    # Fully padded rows produce zero attention, not a uniform average.
    probabilities = jnp.where(valid, probabilities, 0.0)
    out = jnp.einsum(
        "bhts,bshd->bthd", probabilities, v.astype(jnp.float32), precision=jax.lax.Precision.DEFAULT
    )
    return out.reshape(q.shape[0], q.shape[1], -1).astype(q.dtype)


def mlp(x, p):
    z = dense(x, p["w1"], p["b1"])
    # GELU evaluated in FP32 with the standard tanh approximation.
    z = jax.nn.gelu(z.astype(jnp.float32), approximate=True).astype(x.dtype)
    return dense(z, p["w2"], p["b2"])


def block(p, x, mask, config, fused=False, attention_backend=None):
    with jax.named_scope("self_attention"):
        z = layer_norm(x, p["ln1_g"], p["ln1_b"], config)
        q, k, v = project(z, p, config, fused)
        if attention_backend:
            z = jax.nn.dot_product_attention(
                q, k, v, mask=mask[:, None, None, :], implementation=attention_backend
            ).reshape(x.shape)
        else:
            z = attention(q, k, v, mask)
        z = dense(z, p["wo"], p["bo"])
        x = (x.astype(jnp.float32) + z.astype(jnp.float32)).astype(config.compute_dtype)
    with jax.named_scope("feed_forward"):
        z = mlp(layer_norm(x, p["ln2_g"], p["ln2_b"], config), p)
        x = (x.astype(jnp.float32) + z.astype(jnp.float32)).astype(config.compute_dtype)
    return jnp.where(mask[..., None], x, 0)


def forward(params, x, mask, config, fused=False, attention_backend=None):
    for p in params:
        x = block(p, x, mask, config, fused, attention_backend)
    return x


def numpy_reference(params, x, mask, config):
    """Independent FP64 NumPy oracle, including softmax/GELU expressed directly."""
    x = np.asarray(x, np.float64)
    mask = np.asarray(mask, bool)

    def norm(a, g, b):
        delta = a - a.mean(axis=-1, keepdims=True)
        return delta / np.sqrt((delta * delta).mean(axis=-1, keepdims=True) + config.epsilon) * g + b

    for p in params:
        z = norm(x, p["ln1_g"], p["ln1_b"])
        qkv = z @ p["wqkv"].astype(np.float64) + p["bqkv"]
        q, k, v = [
            a.reshape(*x.shape[:2], config.heads, -1).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, -1)
        ]
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
        scores = np.where(mask[:, None, None, :], scores, -1e30)
        probs = np.exp(scores - scores.max(-1, keepdims=True)) * mask[:, None, None, :]
        denom = probs.sum(-1, keepdims=True)
        probs = np.divide(probs, denom, out=np.zeros_like(probs), where=denom != 0)
        z = (probs @ v).transpose(0, 2, 1, 3).reshape(x.shape)
        x = x + z @ p["wo"] + p["bo"]
        z = norm(x, p["ln2_g"], p["ln2_b"])
        z = z @ p["w1"] + p["b1"]
        z = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        x = np.where(mask[..., None], x + z @ p["w2"] + p["b2"], 0)
    return x
