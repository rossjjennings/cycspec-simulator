import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from .interpolation import fft_roll
from .polarization import validate_stokes, coherence_to_stokes
from .plot_helpers import symmetrize_limits
from .time import Time

class PeriodicSpectrum:
    def __init__(self, freq, feed_poln, start_time, I, Q=None, U=None, V=None):
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
        pc = ax.pcolormesh(self.phase - shift, self.freq/1e6, arr, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xlabel('Phase (cycles)')
        ax.set_ylabel('Frequency (MHz)')

        return pc

@nb.njit
def _cycfold_cpu(phi, A, B, nchan, nbin, use_midpt=True, round_to_nearest=True):
    nlag = nchan//2 + 1
    ncorr = A.size - nlag + 1
    corr_AA = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_AB = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_BA = np.zeros((nlag, nbin), dtype=np.complex128)
    corr_BB = np.zeros((nlag, nbin), dtype=np.complex128)
    samples = np.zeros((nlag, nbin), dtype=np.int64)
    for icorr in range(ncorr):
        for ilag in range(nlag):
            if use_midpt:
                phase = phi[2*icorr + ilag] % 1
            else:
                phase = phi[2*icorr] % 1
            if round_to_nearest:
                phase_bin = np.int64(np.round(phase*nbin)) % nbin
            else:
                phase_bin = np.int64(phase*nbin)
            samples[ilag, phase_bin] += 1
            corr_AA[ilag, phase_bin] += A[icorr + ilag] * A[icorr].conjugate()
            corr_AB[ilag, phase_bin] += A[icorr + ilag] * B[icorr].conjugate()
            corr_BA[ilag, phase_bin] += B[icorr + ilag] * A[icorr].conjugate()
            corr_BB[ilag, phase_bin] += B[icorr + ilag] * B[icorr].conjugate()
    corr_AA /= samples
    corr_AB /= samples
    corr_BA /= samples
    corr_BB /= samples
    return corr_AA, corr_AB, corr_BA, corr_BB

def cycfold_cpu(data, ncyc, nbin, phase_predictor, use_midpt=True, round_to_nearest=True):
    offset = np.empty(2*data.t.offset.size - 1)
    offset[::2] = data.t.offset
    offset[1::2] = (data.t.offset[1:] + data.t.offset[:-1])/2
    t = Time(data.t.mjd, data.t.second, offset)
    phi = phase_predictor.phase(t)
    pspec_shape = (data.nchan*ncyc, nbin)
    pspec_AA = np.empty(pspec_shape, dtype=np.float64)
    pspec_BB = np.empty(pspec_shape, dtype=np.float64)
    pspec_CR = np.empty(pspec_shape, dtype=np.float64)
    pspec_CI = np.empty(pspec_shape, dtype=np.float64)
    freq = np.empty(data.nchan*ncyc, dtype=np.float64)
    for ichan in range(data.nchan):
        corr_AA, corr_AB, corr_BA, corr_BB = _cycfold_cpu(
            phi, data.A[ichan], data.B[ichan], ncyc, nbin, use_midpt, round_to_nearest
        )
        corr_CR = (corr_AB + corr_BA)/2
        corr_CI = (corr_AB - corr_BA)/2j
        pspec_AA[ichan*ncyc:(ichan+1)*ncyc] = np.fft.fftshift(np.fft.hfft(corr_AA, axis=0), axes=0)
        pspec_BB[ichan*ncyc:(ichan+1)*ncyc] = np.fft.fftshift(np.fft.hfft(corr_BB, axis=0), axes=0)
        pspec_CR[ichan*ncyc:(ichan+1)*ncyc] = np.fft.fftshift(np.fft.hfft(corr_CR, axis=0), axes=0)
        pspec_CI[ichan*ncyc:(ichan+1)*ncyc] = np.fft.fftshift(np.fft.hfft(corr_CI, axis=0), axes=0)
        chan_midfreq = data.obsfreq + (ichan + 1/2 - data.nchan/2)*data.chan_bw
        freq[ichan*ncyc:(ichan+1)*ncyc] = (
            chan_midfreq + np.fft.fftshift(np.fft.fftfreq(ncyc, 1/data.chan_bw))
        )

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    pspec = PeriodicSpectrum(freq, data.feed_poln, data.start_time, I, Q, U, V)
    return pspec
