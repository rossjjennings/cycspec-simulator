import numpy as np
import numba as nb

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

def pspec_ryan4(data, nchan, nbin, phase_predictor):
    """
    Produce a periodic spectrum estimate using the algorithm from Ryan's `cycfold4.py`.
    """
    nlag = nchan//2 + 1
    corr_xx = outer_correlate(data.X, data.X, nlag)
    corr_yy = outer_correlate(data.Y, data.Y, nlag)
    wigner_xx = np.fft.fftshift(np.fft.hfft(corr_xx), axes=-1)
    wigner_yy = np.fft.fftshift(np.fft.hfft(corr_yy), axes=-1)

    pspec_xx = np.zeros((nchan, nbin))
    pspec_yy = np.zeros((nchan, nbin))
    samples = np.zeros(nbin)
    phase = phase_predictor.phase(data.t) % 1
    phase_bin = np.int64(phase[:-(nlag-1)]*nbin)
    for ibin in range(nbin):
        pspec_xx[:, ibin] += np.sum(wigner_xx[phase_bin==ibin, :], axis=0)
        pspec_yy[:, ibin] += np.sum(wigner_yy[phase_bin==ibin, :], axis=0)
        samples[ibin] += np.sum(phase_bin==ibin)
    pspec_xx /= samples
    pspec_yy /= samples
    return pspec_xx, pspec_yy

def pspec_corrfirst(data, nchan, nbin, phase_predictor):
    """
    Produce a periodic spectrum estimate by first folding the cyclic correlation
    functions, and taking the Fourier transform once at the end.
    """
    nlag = nchan//2 + 1
    outer_corr_xx = outer_correlate(data.X, data.X, nlag)
    outer_corr_yy = outer_correlate(data.Y, data.Y, nlag)

    folded_corr_xx = np.zeros((nlag, nbin), dtype=np.complex128)
    folded_corr_yy = np.zeros((nlag, nbin), dtype=np.complex128)
    samples = np.zeros(nbin, dtype=np.int64)
    phase = phase_predictor.phase(data.t) % 1
    phase_bin = np.int64(phase[:-(nlag-1)]*nbin)
    for ibin in range(nbin):
        folded_corr_xx[:, ibin] += np.sum(outer_corr_xx[phase_bin==ibin, :], axis=0)
        folded_corr_yy[:, ibin] += np.sum(outer_corr_yy[phase_bin==ibin, :], axis=0)
        samples[ibin] += np.sum(phase_bin==ibin)
    pspec_xx = np.fft.fftshift(np.fft.hfft(folded_corr_xx, axis=0), axes=0)
    pspec_yy = np.fft.fftshift(np.fft.hfft(folded_corr_yy, axis=0), axes=0)
    pspec_xx /= samples
    pspec_yy /= samples
    return pspec_xx, pspec_yy

@nb.njit
def cycfold_numba(data, nchan, nbin, phase_predictor):
    nlag = nchan//2 + 1
    ncorr = data.X.size - nlag + 1
    corr_xx = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_yy = np.zeros((nlag, nbin), dtype=np.complex128)
    samples = np.zeros(nbin, dtype=np.int64)
    for icorr in range(ncorr):
        phase = phase_predictor.phase(data.t[icorr]) % 1
        phase_bin = np.int64(phase*nbin)
        samples[phase_bin] += 1
        for ilag in range(nlag):
            corr_xx[ilag, phase_bin] += data.X[icorr + ilag]*data.X[icorr].conjugate()
            corr_yy[ilag, phase_bin] += data.Y[icorr + ilag]*data.Y[icorr].conjugate()
    corr_xx /= samples
    corr_yy /= samples
    return corr_xx, corr_yy

def pspec_numba(data, nchan, nbin, phase_predictor):
    corr_xx, corr_yy = cycfold_numba(data, nchan, nbin, phase_predictor)
    pspec_xx = np.fft.fftshift(np.fft.hfft(corr_xx, axis=0), axes=0)
    pspec_yy = np.fft.fftshift(np.fft.hfft(corr_yy, axis=0), axes=0)
    return pspec_xx, pspec_yy
