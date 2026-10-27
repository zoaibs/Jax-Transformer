import numpy as np

from .bucketing import make_batches


def request_lengths(count=25, minimum=65, maximum=512, seed=42):
    if count < 1 or minimum < 1 or maximum < minimum:
        raise ValueError("Invalid request distribution")
    # Uniform-integer trace intentionally exposes specialization; not a production trace.
    return np.random.default_rng(seed).integers(minimum, maximum + 1, size=count).tolist()


def batch_inputs(batch, hidden, seed):
    x = np.zeros((*batch.shape, hidden), dtype=np.float32)
    mask = np.zeros(batch.shape, dtype=bool)
    for row, (index, length) in enumerate(zip(batch.indices, batch.lengths, strict=True)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        x[row, :length] = rng.standard_normal((length, hidden)).astype(np.float32)
        mask[row, :length] = True
    return x, mask


def static_inputs(batch, sequence, hidden, seed=7):
    return batch_inputs(make_batches([sequence] * batch, batch)[0], hidden, seed)
