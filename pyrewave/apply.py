from jax.numpy import concatenate
from .batch.apply import apply as batch_apply


def apply(*args, **kwargs):
    batches = list(batch_apply(*args, **kwargs))
    return concatenate(concatenate(batches))
