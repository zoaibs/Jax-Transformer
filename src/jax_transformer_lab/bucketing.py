"""Request-order batching: bucketing pads lengths but never reorders requests."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Batch:
    indices: tuple[int, ...]
    lengths: tuple[int, ...]
    padded_length: int

    @property
    def useful_tokens(self):
        return sum(self.lengths)

    @property
    def processed_tokens(self):
        return len(self.lengths) * self.padded_length

    @property
    def shape(self):
        return len(self.lengths), self.padded_length

    def to_dict(self):
        return asdict(self) | {"useful_tokens": self.useful_tokens, "processed_tokens": self.processed_tokens}


def make_batches(lengths, batch_size, buckets=None):
    if batch_size < 1 or not lengths or min(lengths) < 1:
        raise ValueError("Nonempty positive lengths and a positive batch size are required")
    if buckets is not None and (not buckets or list(buckets) != sorted(set(buckets)) or min(buckets) < 1):
        raise ValueError("buckets must be strictly increasing positive integers")
    batches = []
    for start in range(0, len(lengths), batch_size):
        actual = tuple(int(n) for n in lengths[start : start + batch_size])
        maximum = max(actual)
        if buckets is not None:
            padded = next((n for n in buckets if n >= maximum), None)
            if padded is None:
                raise ValueError(f"No bucket fits length {maximum}")
        else:
            padded = maximum
        batches.append(Batch(tuple(range(start, start + len(actual))), actual, padded))
    return batches
