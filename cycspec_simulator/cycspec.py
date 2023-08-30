import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from .interpolation import fft_roll
from .polarization import validate_stokes, coherence_to_stokes
from .plot_helpers import symmetrize_limits

class PeriodicSpectrum:
    def __init__(self, freq, I, Q=None, U=None, V=None):
        """
        Create a new peiodic spectrum from frequency, I, Q, U, and V arrays.
        If one of Q, U, or V is present, all must be present with the same shape.
        """
        self.freq = freq

        self.full_stokes, self.shape = validate_stokes(I, Q, U, V)
        self.I = I
        if self.full_stokes:
            self.Q = Q
            self.U = U
            self.V = V

        self.nbin = self.shape[-1]
        self.phase = np.linspace(0, 1, self.nbin, endpoint=False)

    def plot(self, ax=None, what='I', shift=0.0, sym_lim=False, vmin=None, vmax=None,
             **kwargs):
        """
        Plot the periodic spectrum.

        Parameters
        ----------
        ax: Axes on which to plot periodic spectrum. If `None`,
            a new Figure and Axes will be created.
        what: Which Stokes parameter to plot: 'I', 'Q', 'U', or 'V'.
              Ignored if spectrum only has total intensity data.
        shift: Rotation (in cycles) to apply before plotting.

        Additional keyword arguments are passed on to ax.pcolormesh().
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        arr = getattr(self, what)
        arr = fft_roll(arr, shift*self.nbin)
        if sym_lim:
            vmin, vmax = symmetrize_limits(arr, vmin, vmax)
        pc = ax.pcolormesh(self.phase, self.freq/1e6, arr, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xlabel('Phase (cycles)')
        ax.set_ylabel('Frequency (MHz)')

        return pc

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
    corr_AA = outer_correlate(data.A, data.A, nlag)
    corr_AB = outer_correlate(data.A, data.B, nlag)
    corr_BA = outer_correlate(data.B, data.A, nlag)
    corr_BB = outer_correlate(data.B, data.B, nlag)
    wigner_AA = np.fft.fftshift(np.fft.hfft(corr_AA), axes=-1)
    wigner_BB = np.fft.fftshift(np.fft.hfft(corr_BB), axes=-1)
    wigner_CR = np.fft.fftshift(np.fft.hfft((corr_AB + corr_BA)/2), axes=-1)
    wigner_CI = np.fft.fftshift(np.fft.hfft((corr_AB - corr_BA)/2j), axes=-1)

    pspec_AA = np.zeros((nchan, nbin))
    pspec_BB = np.zeros((nchan, nbin))
    pspec_CR = np.zeros((nchan, nbin))
    pspec_CI = np.zeros((nchan, nbin))
    samples = np.zeros(nbin)
    phase = phase_predictor.phase(data.t) % 1
    phase_bin = np.int64(phase[:-(nlag-1)]*nbin)
    for ibin in range(nbin):
        pspec_AA[:, ibin] += np.sum(wigner_AA[phase_bin==ibin, :], axis=0)
        pspec_BB[:, ibin] += np.sum(wigner_BB[phase_bin==ibin, :], axis=0)
        pspec_CR[:, ibin] += np.sum(wigner_CR[phase_bin==ibin, :], axis=0)
        pspec_CI[:, ibin] += np.sum(wigner_CI[phase_bin==ibin, :], axis=0)
        samples[ibin] += np.sum(phase_bin==ibin)
    pspec_AA /= samples
    pspec_BB /= samples
    pspec_CR /= samples
    pspec_CI /= samples

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    freq = np.fft.fftshift(np.fft.fftfreq(nchan, 1/data.bandwidth))
    pspec = PeriodicSpectrum(freq, I, Q, U, V)
    return pspec

def pspec_corrfirst(data, nchan, nbin, phase_predictor):
    """
    Produce a periodic spectrum estimate by first folding the cyclic correlation
    functions, and taking the Fourier transform once at the end.
    """
    nlag = nchan//2 + 1
    outer_corr_AA = outer_correlate(data.A, data.A, nlag)
    outer_corr_AB = outer_correlate(data.A, data.B, nlag)
    outer_corr_BA = outer_correlate(data.B, data.A, nlag)
    outer_corr_BB = outer_correlate(data.B, data.B, nlag)

    folded_corr_AA = np.zeros((nlag, nbin), dtype=np.complex128)
    folded_corr_AB = np.zeros((nlag, nbin), dtype=np.complex128)
    folded_corr_BA = np.zeros((nlag, nbin), dtype=np.complex128)
    folded_corr_BB = np.zeros((nlag, nbin), dtype=np.complex128)
    samples = np.zeros(nbin, dtype=np.int64)
    phase = phase_predictor.phase(data.t) % 1
    phase_bin = np.int64(phase[:-(nlag-1)]*nbin)
    for ibin in range(nbin):
        folded_corr_AA[:, ibin] += np.sum(outer_corr_AA[phase_bin==ibin, :], axis=0)
        folded_corr_AB[:, ibin] += np.sum(outer_corr_AB[phase_bin==ibin, :], axis=0)
        folded_corr_BA[:, ibin] += np.sum(outer_corr_BA[phase_bin==ibin, :], axis=0)
        folded_corr_BB[:, ibin] += np.sum(outer_corr_BB[phase_bin==ibin, :], axis=0)
        samples[ibin] += np.sum(phase_bin==ibin)
    pspec_AA = np.fft.fftshift(np.fft.hfft(folded_corr_AA, axis=0), axes=0)
    pspec_BB = np.fft.fftshift(np.fft.hfft(folded_corr_BB, axis=0), axes=0)
    folded_corr_CR = (folded_corr_AB + folded_corr_BA)/2
    folded_corr_CI = (folded_corr_AB - folded_corr_BA)/2j
    pspec_CR = np.fft.fftshift(np.fft.hfft(folded_corr_CR, axis=0), axes=0)
    pspec_CI = np.fft.fftshift(np.fft.hfft(folded_corr_CI, axis=0), axes=0)
    pspec_AA /= samples
    pspec_BB /= samples
    pspec_CR /= samples
    pspec_CI /= samples

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    freq = np.fft.fftshift(np.fft.fftfreq(nchan, 1/data.bandwidth))
    pspec = PeriodicSpectrum(freq, I, Q, U, V)
    return pspec

@nb.njit
def cycfold_numba(data, nchan, nbin, phase_predictor, use_midpt=True, round_to_nearest=True):
    nlag = nchan//2 + 1
    ncorr = data.A.size - nlag + 1
    corr_AA = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_AB = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_BA = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_BB = np.zeros((nlag, nbin), dtype=np.complex128)
    samples = np.zeros((nlag, nbin), dtype=np.int64)
    for icorr in range(ncorr):
        for ilag in range(nlag):
            if use_midpt:
                t = (data.t[icorr] + data.t[icorr + ilag])/2
            else:
                t = data.t[icorr]
            phase = phase_predictor.phase(t) % 1
            if round_to_nearest:
                phase_bin = np.int64(np.round(phase*nbin)) % nbin
            else:
                phase_bin = np.int64(phase*nbin)
            samples[ilag, phase_bin] += 1
            corr_AA[ilag, phase_bin] += data.A[icorr + ilag]*data.A[icorr].conjugate()
            corr_AB[ilag, phase_bin] += data.A[icorr + ilag]*data.B[icorr].conjugate()
            corr_BA[ilag, phase_bin] += data.B[icorr + ilag]*data.A[icorr].conjugate()
            corr_BB[ilag, phase_bin] += data.B[icorr + ilag]*data.B[icorr].conjugate()
    corr_AA /= samples
    corr_AB /= samples
    corr_BA /= samples
    corr_BB /= samples
    return corr_AA, corr_AB, corr_BA, corr_BB

def pspec_numba(data, nchan, nbin, phase_predictor, use_midpt=True, round_to_nearest=True):
    corr_AA, corr_AB, corr_BA, corr_BB = cycfold_numba(data, nchan, nbin, phase_predictor,
                                                       use_midpt, round_to_nearest)
    corr_CR = (corr_AB + corr_BA)/2
    corr_CI = (corr_AB - corr_BA)/2j
    pspec_AA = np.fft.fftshift(np.fft.hfft(corr_AA, axis=0), axes=0)
    pspec_BB = np.fft.fftshift(np.fft.hfft(corr_BB, axis=0), axes=0)
    pspec_CR = np.fft.fftshift(np.fft.hfft(corr_CR, axis=0), axes=0)
    pspec_CI = np.fft.fftshift(np.fft.hfft(corr_CI, axis=0), axes=0)

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    freq = np.fft.fftshift(np.fft.fftfreq(nchan, 1/data.bandwidth))
    pspec = PeriodicSpectrum(freq, I, Q, U, V)
    return pspec
