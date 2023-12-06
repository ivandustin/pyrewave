from functools import partial
from jax.numpy import linspace, exp, pi, newaxis
from jax.numpy.fft import fftfreq
from jax import jit, vmap
from pyrewave.core.apply import apply as core_apply


def apply(batch_size, width, multiplier, sample_rate, points):
    frequencies = fftfreq(width, 1 / sample_rate)
    space = (width / sample_rate) / (width * multiplier)
    time = linspace(space, width / sample_rate, int(width * multiplier))[:, newaxis]
    exponential = exp(2j * pi * frequencies * time)
    function = jit(vmap(partial(core_apply, exponential)))
    for i in range(0, points.size, batch_size * width):
        batch = points[i : i + batch_size * width]
        batch = batch.reshape((batch_size, width))
        result = function(batch)
        yield result
