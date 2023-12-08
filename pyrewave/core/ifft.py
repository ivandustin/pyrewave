from jax.numpy import sum, pi, exp


def ifft(frequencies, fourier, time):
    return sum(fourier * exp(2j * pi * frequencies * time))
