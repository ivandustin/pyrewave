from functools import partial
from jax.numpy.fft import fftfreq
from jax.numpy import arange
from jax import jit, vmap
from pyrewave.ifft import ifft


def ifft(batch_size, new_sample_rate, sample_rate, fourier):
    frequencies = fftfreq(fourier.size, 1 / sample_rate)
    time = arange(0, fourier.size / sample_rate, 1 / new_sample_rate)
    function = jit(vmap(partial(ifft, frequencies, fourier)))
    for i in range(0, time.size, batch_size):
        value = function(time[i : i + batch_size])
        yield value
