import numpy as np
import dask.array as da

from .gpu import have_cuda
if have_cuda:
    import cupy as cp

def fft_roll(arr, shift):
    """
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    """
    if isinstance(arr, da.array):
        xp = da
    elif isinstance(arr, cp.array):
        xp = cp
    else:
        xp = np

    n = arr.shape[-1]
    if not hasattr(shift, 'shape'):
        shift = xp.array(shift)
    shift = shift[..., xp.newaxis]
    phase = -2j*xp.pi*shift*xp.fft.rfftfreq(n)
    return xp.fft.irfft(xp.fft.rfft(arr)*xp.exp(phase), n)

def fft_interp(arr, x):
    """
    Interpolate the values in `arr` at the locations `x`, in bins.
    As with `fft_roll()`, this works by using the amplitudes and frequencies
    associated with the DFT of `arr` to define a continuous function.
    """
    if isinstance(arr, da.array):
        xp = da
    elif isinstance(arr, cp.array):
        xp = cp
    else:
        xp = np

    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = xp.array(x)
    phase = 2j*xp.pi*x[..., xp.newaxis]*xp.fft.fftfreq(n)
    return xp.mean(xp.fft.fft(arr)*xp.exp(phase), axis=-1)[()]

def lerp(arr, x):
    """
    Linearly interpolate the values in `arr` at the locations `x`, in bins.
    For locations `x` outside the original array, extrapolate the function
    periodically.
    """
    if isinstance(arr, da.array):
        xp = da
    elif isinstance(arr, cp.array):
        xp = cp
    else:
        xp = np

    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = xp.array(x)
    floor = xp.floor(x)
    t = x - floor
    pre_idx = floor.astype(xp.int64) % n
    post_idx = xp.ceil(x).astype(xp.int64) % n
    pre_val = da.take(arr, pre_idx)
    post_val = da.take(arr, post_idx)
    interp_val = (1-t)*pre_val + t*post_val

    return interp_val[()]
