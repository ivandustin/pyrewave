from jax.numpy import concatenate
from jax.numpy.fft import fft
from pyrewave.ifft import ifft


def resample(batch_size, new_sample_rate, sample_rate, points):
    fourier = fft(points)
    batches = list(ifft(batch_size, new_sample_rate, sample_rate, fourier))
    return concatenate(batches)
