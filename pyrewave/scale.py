from jax.numpy import iinfo
from .core.scale import scale as scale_function


def scale(target_type, x):
    m = iinfo(target_type).max
    n = x.max()
    y = scale_function(m, n, x)
    return y.astype(target_type)
