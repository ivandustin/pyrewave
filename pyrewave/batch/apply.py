from functools import partial
from jax.numpy import linspace, exp, pi, newaxis
from jax.numpy.fft import fftfreq
from jax import jit, vmap
from pyrewave.core.apply import apply as core_apply


def apply(batch_width, batch_length, multiplier, sample_rate, points):
    frequencies = fftfreq(batch_width, 1 / sample_rate)
    space = (batch_width / sample_rate) / (batch_width * multiplier)
    time = linspace(space, batch_width / sample_rate, int(batch_width * multiplier))
    exponential = exp(2j * pi * frequencies * time[:, newaxis])
    function = jit(vmap(partial(core_apply, exponential)))
    for i in range(0, points.size, batch_length * batch_width):
        batch = points[i : i + batch_length * batch_width]
        batch = batch.reshape((batch_length, batch_width))
        result = function(batch)
        yield result
