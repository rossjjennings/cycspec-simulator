import numpy as np
import dask.array as da

def fft_roll(arr, shift):
    """
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    """
    n = arr.shape[-1]
    if not hasattr(shift, 'shape'):
        shift = np.array(shift)
    shift = shift[..., np.newaxis]
    phase = -2j*np.pi*shift*np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(arr)*np.exp(phase), n)

def fft_interp(arr, x):
    """
    Interpolate the values in `arr` at the locations `x`, in bins.
    As with `fft_roll()`, this works by using the amplitudes and frequencies
    associated with the DFT of `arr` to define a continuous function.
    """
    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = np.array(x)
    phase = 2j*np.pi*x[..., np.newaxis]*np.fft.fftfreq(n)
    return np.mean(np.fft.fft(arr)*np.exp(phase), axis=-1)[()]

def lerp(arr, x):
    """
    Linearly interpolate the values in `arr` at the locations `x`, in bins.
    For locations `x` outside the original array, extrapolate the function
    periodically.
    """
    n = arr.shape[-1]
    if not hasattr(x, 'shape'):
        x = np.array(x)
    floor = np.floor(x)
    t = x - floor
    pre_idx = floor.astype(np.int64) % n
    post_idx = np.ceil(x).astype(np.int64) % n
    pre_val = da.take(arr, pre_idx)
    post_val = da.take(arr, post_idx)
    interp_val = (1-t)*pre_val + t*post_val

    return interp_val[()]
