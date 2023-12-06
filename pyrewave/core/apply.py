from jax.numpy.fft import fft
from jax.numpy import sum


def apply(exponential, points):
    return sum(fft(points) * exponential, axis=1)
