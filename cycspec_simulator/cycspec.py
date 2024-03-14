import numpy as np
import numba as nb
import dask
import dask.array as da
import matplotlib.pyplot as plt
import time

from .interpolation import fft_roll
from .polarization import validate_stokes, coherence_to_stokes
from .plot_helpers import symmetrize_limits
from .time import Time

class PeriodicSpectrum:
    def __init__(self, freq, start_time, I, Q=None, U=None, V=None):
        """
        Create a new peiodic spectrum from frequency, I, Q, U, and V arrays.
        If one of Q, U, or V is present, all must be present with the same shape.
        """
        self.freq = freq
        self.start_time = start_time

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

signatures = [
    nb.types.Tuple((
        nb.complex64[:,:],
        nb.complex64[:,:],
        nb.complex64[:,:],
        nb.complex64[:,:],
        nb.int64[:,:]
    ))(
        nb.complex64[:],
        nb.complex64[:],
        nb.int64,
        nb.int64,
        nb.int64[:],
        nb.boolean,
    ),
    nb.types.Tuple((
        nb.complex128[:,:],
        nb.complex128[:,:],
        nb.complex128[:,:],
        nb.complex128[:,:],
        nb.int64[:,:]
    ))(
        nb.complex128[:],
        nb.complex128[:],
        nb.int64,
        nb.int64,
        nb.int64[:],
        nb.boolean,
    ),
]

@nb.njit(signatures, parallel=True)
def corrfold_cpu(A, B, nlag, nbin, binplan, include_end=False):
    """
    Compute the cyclic autocorrelation function from sampled data.
    This function is intended to be used internally by cycfold_cpu().

    Parameters
    ----------
    A, B: Baseband samples in each of two polarizations (each of length n)
    nlag: Number of lags to use for the correlation
    nbin: Number of phase bins in which to accumulate
    binplan: Array giving the phase bin corresponding to each half-sample time
          (length 2*n, where n is the number of samples)
    include_end: Whether to calculate products where the first sample is among
          the last nlag - ilag - 1 samples. In such cases, there are fewer than
          nlag choices for the second sample. Setting include_end=True means that
          slightly more samples will contribute to lower lags.
    """
    nchan = A.shape[0]
    corr_AA = np.zeros((nlag, nbin), dtype=A.dtype)
    corr_AB = np.zeros((nlag, nbin), dtype=A.dtype)
    corr_BA = np.zeros((nlag, nbin), dtype=A.dtype)
    corr_BB = np.zeros((nlag, nbin), dtype=A.dtype)
    samples = np.zeros((nlag, nbin), dtype=np.int64)
    for ilag in nb.prange(nlag):
        if include_end:
            ncorr = A.size - ilag
        else:
            ncorr = A.size - nlag + 1

        for icorr in range(ncorr):
            phase_bin = binplan[2*icorr + ilag]
            samples[ilag, phase_bin] += 1
            corr_AA[ilag, phase_bin] += (
                A[icorr + ilag] * A[icorr].conjugate()
            )
            corr_AB[ilag, phase_bin] += (
                A[icorr + ilag] * B[icorr].conjugate()
            )
            corr_BA[ilag, phase_bin] += (
                B[icorr + ilag] * A[icorr].conjugate()
            )
            corr_BB[ilag, phase_bin] += (
                B[icorr + ilag] * B[icorr].conjugate()
            )
    corr_AA /= samples
    corr_AB /= samples
    corr_BA /= samples
    corr_BB /= samples
    return corr_AA, corr_AB, corr_BA, corr_BB, samples

def cycfold_cpu(data, ncyc, nbin, phase_predictor, include_end=False,
                n_threads=nb.config.NUMBA_NUM_THREADS):
    """
    Compute the periodic spectrum from sampled data.

    Parameters
    ----------
    data: BasebandData object containing the data to use
    ncyc: Number of "cyclic channels" per input channel
    nbin: Number of phase bins in which to accumulate
    phase_predictor: Predictor object to use in computing phases. Should have
          a phase() method which can be called with a Time object to yield the
          corresponding array of phases.
    include_end: Passed along to corrfold_cpu(), see there for details.
    n_threads: Number of CPU threads to use. Defaults to the total number of
          available CPUs, as detected by Numba.
    """
    nlag = ncyc//2 + 1
    t_span = data.n_samples/np.abs(data.bandwidth)

    if data.delayed:
        chunks, = data.A.chunks
        def linspace(*args, **kwargs):
            return da.linspace(*args, chunks=2*chunks[0], **kwargs)
    else:
        linspace = np.linspace
    offset = linspace(0, t_span, 2*data.n_samples, endpoint=False)
    t = Time(
        data.start_time.mjd,
        data.start_time.second,
        data.start_time.offset + offset,
    )

    phase = phase_predictor.phase(t)
    binplan = (np.round((phase % 1)*nbin)).astype(np.int64) % nbin

    if data.delayed:
        corr_AA, corr_AB, corr_BA, corr_BB, samples = [], [], [], [], []
        A = da.overlap.overlap(data.A, depth={0: (0, nlag - 1)}, boundary=None)
        B = da.overlap.overlap(data.B, depth={0: (0, nlag - 1)}, boundary=None)
        binplan = da.overlap.overlap(binplan, depth={0: (0, 2*nlag - 2)}, boundary=None)
        for A_blk, B_blk, plan_blk in zip(A.blocks, B.blocks, binplan.blocks):
            corrfold = dask.delayed(corrfold_cpu, nout=5)
            AA_blk, AB_blk, BA_blk, BB_blk, samples_blk = corrfold(
                A_blk, B_blk, nlag, nbin, plan_blk, include_end
            )
            corr_AA.append(da.from_delayed(AA_blk, (nlag, nbin), dtype=data.A.dtype))
            corr_AB.append(da.from_delayed(AB_blk, (nlag, nbin), dtype=data.A.dtype))
            corr_BA.append(da.from_delayed(BA_blk, (nlag, nbin), dtype=data.A.dtype))
            corr_BB.append(da.from_delayed(BB_blk, (nlag, nbin), dtype=data.A.dtype))
            samples.append(da.from_delayed(samples_blk, (nlag, nbin), dtype=np.int64))
        corr_AA = sum(corr_AA).compute()
        corr_AB = sum(corr_AB).compute()
        corr_BA = sum(corr_BA).compute()
        corr_BB = sum(corr_BB).compute()
        samples = sum(samples).compute()
        print(f"Total products accumulated: {4*np.sum(samples)}")
    else:
        timer = CPUTimer()
        with timer, NumbaThreads(n_threads):
            corr_AA, corr_AB, corr_BA, corr_BB, samples = corrfold_cpu(
                data.A, data.B, nlag, nbin, binplan, include_end
            )
        print(f"Elapsed time: {timer.elapsed:g} ms")
        print(f"Total products accumulated: {4*np.sum(samples)}")
        throughput = 4*np.sum(samples)/(timer.elapsed/1000)
        print(f"Throughput: {throughput:g} products/sec.")
    corr_CR = (corr_AB + corr_BA)/2
    corr_CI = (corr_AB - corr_BA)/np.array(2j).astype(data.A.dtype)
    pspec_AA = np.fft.fftshift(np.fft.hfft(corr_AA, axis=0), axes=0)
    pspec_BB = np.fft.fftshift(np.fft.hfft(corr_BB, axis=0), axes=0)
    pspec_CR = np.fft.fftshift(np.fft.hfft(corr_CR, axis=0), axes=0)
    pspec_CI = np.fft.fftshift(np.fft.hfft(corr_CI, axis=0), axes=0)
    bandwidth = data.bandwidth
    freq = data.obsfreq + np.linspace(-bandwidth/2, bandwidth/2, ncyc, endpoint=False)

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    pspec = PeriodicSpectrum(freq, data.start_time, I, Q, U, V)
    return pspec
