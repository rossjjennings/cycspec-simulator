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

class CPUTimer:
    """
    Context manager for timing CPU code.
    """
    def __init__(self):
        self.elapsed = None # elapsed time in ms

    def __enter__(self):
        self.start_time_ns = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.end_time_ns = time.perf_counter_ns()
        self.elapsed = (self.end_time_ns - self.start_time_ns)/1e6

class NumbaThreads:
    """
    Context manager for setting and restoring the number of
    threads launched by Numba for parallel functions.
    """
    def __init__(self, n_threads):
        self.n_threads = n_threads

    def __enter__(self):
        self.n_threads_old = nb.get_num_threads()
        nb.set_num_threads(self.n_threads)

    def __exit__(self, type, value, traceback):
        nb.set_num_threads(self.n_threads_old)

@nb.njit(parallel=True)
def _cycfold_cpu(A, B, nlag, nbin, binplan):
    nchan = A.shape[0]
    ncorr = A.shape[1] - nlag + 1
    corr_AA = np.zeros((nchan, nlag, nbin), dtype=A.dtype)
    corr_AB = np.zeros((nchan, nlag, nbin), dtype=A.dtype)
    corr_BA = np.zeros((nchan, nlag, nbin), dtype=A.dtype)
    corr_BB = np.zeros((nchan, nlag, nbin), dtype=A.dtype)
    samples = np.zeros((nchan, nlag, nbin), dtype=np.int64)
    for ichan in range(nchan):
        for ilag in nb.prange(nlag):
            for icorr in range(ncorr):
                phase_bin = binplan[2*icorr + ilag]
                samples[ichan, ilag, phase_bin] += 1
                corr_AA[ichan, ilag, phase_bin] += (
                    A[ichan, icorr + ilag] * A[ichan, icorr].conjugate()
                )
                corr_AB[ichan, ilag, phase_bin] += (
                    A[ichan, icorr + ilag] * B[ichan, icorr].conjugate()
                )
                corr_BA[ichan, ilag, phase_bin] += (
                    B[ichan, icorr + ilag] * A[ichan, icorr].conjugate()
                )
                corr_BB[ichan, ilag, phase_bin] += (
                    B[ichan, icorr + ilag] * B[ichan, icorr].conjugate()
                )
    corr_AA /= samples
    corr_AB /= samples
    corr_BA /= samples
    corr_BB /= samples
    return corr_AA, corr_AB, corr_BA, corr_BB, samples

def cycfold_cpu(data, ncyc, nbin, phase_predictor, n_threads=nb.config.NUMBA_NUM_THREADS):
    nlag = ncyc//2 + 1
    offset = np.empty(2*data.t.offset.size - 1)
    offset[::2] = data.t.offset
    offset[1::2] = (data.t.offset[1:] + data.t.offset[:-1])/2
    t = Time(data.t.mjd, data.t.second, offset)
    phase = phase_predictor.phase(t)
    binplan = np.int64(np.round((phase % 1)*nbin)) % nbin
    with NumbaThreads(n_threads):
        corr_AA, corr_AB, corr_BA, corr_BB, samples = _cycfold_cpu(
            data.A, data.B, nlag, nbin, binplan
        )
    corr_CR = (corr_AB + corr_BA)/2
    corr_CI = (corr_AB - corr_BA)/2j
    pspec_AA = np.fft.fftshift(np.fft.hfft(corr_AA, axis=1), axes=1)
    pspec_AA = pspec_AA.reshape(data.nchan*ncyc, nbin)
    pspec_BB = np.fft.fftshift(np.fft.hfft(corr_BB, axis=1), axes=1)
    pspec_BB = pspec_BB.reshape(data.nchan*ncyc, nbin)
    pspec_CR = np.fft.fftshift(np.fft.hfft(corr_CR, axis=1), axes=1)
    pspec_CR = pspec_CR.reshape(data.nchan*ncyc, nbin)
    pspec_CI = np.fft.fftshift(np.fft.hfft(corr_CI, axis=1), axes=1)
    pspec_CI = pspec_CI.reshape(data.nchan*ncyc, nbin)
    bandwidth = data.nchan*data.chan_bw
    nfreq = data.nchan*ncyc
    freq = data.obsfreq + np.linspace(-bandwidth/2, bandwidth/2, nfreq, endpoint=False)

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    pspec = PeriodicSpectrum(freq, data.feed_poln, data.start_time, I, Q, U, V)
    return pspec
