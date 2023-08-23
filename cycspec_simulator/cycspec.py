import numpy as np

def outer_correlate(x, y, nlag):
    """
    Produce an array containing all valid lagged products of x and y up to `nlag` lags.
    For 1D arrays, this is equivalent to, but faster than, Ryan's correlate() function.
    """
    ncorr = x.size - nlag + 1
    out = np.empty((ncorr, nlag), dtype=np.complex128)
    for lag in range(nlag):
        corr = x[lag:ncorr + lag]*y[:ncorr].conjugate()
        out[:,lag] = corr

    return out

def cycfold4(data, nchan, nbin, phase_predictor):
    """
    Produce a periodic spectrum estimate using the algorithm from Ryan's `cycfold4.py`.
    """
    nlag = nchan//2 + 1
    corr_xx = outer_correlate(data.X, data.X, nlag)
    corr_yy = outer_correlate(data.Y, data.Y, nlag)
    wigner_xx = np.fft.fftshift(np.fft.hfft(corr_xx), axes=-1)
    wigner_yy = np.fft.fftshift(np.fft.hfft(corr_yy), axes=-1)

    periodic_spectrum_xx = np.zeros((nchan, nbin))
    periodic_spectrum_yy = np.zeros((nchan, nbin))
    samples = np.zeros(nbin)
    phase = phase_predictor(data.t) % 1
    phase_bin = np.int64(phase[:-(nlag-1)]*nbin)
    for ibin in range(nbin):
        periodic_spectrum_xx[:, ibin] += np.sum(wigner_xx[phase_bin==ibin, :])
        periodic_spectrum_yy[:, ibin] += np.sum(wigner_yy[phase_bin==ibin, :])
        samples[ibin] += np.sum(phase_bin==ibin)
    periodic_spectrum_xx /= samples
    periodic_spectrum_yy /= samples
    return periodic_spectrum_xx, periodic_spectrum_yy
