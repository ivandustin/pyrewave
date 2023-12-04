from jax.numpy import fft, sum, exp, pi


def function(time, frequencies, points):
    return sum(fft(points) * exp(2j * pi * frequencies * time), axis=1)
