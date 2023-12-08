from jax.numpy import iinfo
from .core.scale import scale as scale_function


def scale(temporary_type, target_type, input):
    x = input.astype(temporary_type)
    n = x.max()
    m = iinfo(target_type).max
    y = scale_function(m, n, x)
    return y.astype(target_type)
