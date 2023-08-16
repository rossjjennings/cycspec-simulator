import numpy as np

def fft_roll(a, shift):
    """
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    """
    try:
        shift = shift[..., np.newaxis]
    except (TypeError, IndexError): pass
    phase = -2j*np.pi*shift*np.fft.rfftfreq(a.shape[-1])
    return np.fft.irfft(np.fft.rfft(a)*np.exp(phase))
