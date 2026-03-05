import numpy as np
import numba as nb
import dask
import dask.array as da
import matplotlib.pyplot as plt
import time
from loguru import logger

from .interpolation import fft_roll
from .polarization import validate_stokes, coherence_to_stokes
from .plot_helpers import symmetrize_limits
from .time import Time

class PeriodicSpectrum:
    def __init__(self, freq, start_time, I, Q=None, U=None, V=None, samples=None, elapsed=None):
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

        self.samples = samples
        self.elapsed = elapsed

        self.nbin = self.shape[-1]
        self.phase = np.linspace(0, 1, self.nbin, endpoint=False)

    @property
    def delayed(self):
        return isinstance(self.I, da.Array)

    def compute(self, n_workers=None):
        if self.delayed:
            I, Q, U, V, samples, elapsed = dask.compute(
                self.I, self.Q, self.U, self.V, self.samples, self.elapsed,
                num_workers=n_workers,
            )
            if hasattr(I, 'get'):
                # transfer CuPy arrays to CPU
                I = I.get()
                Q = Q.get()
                U = U.get()
                V = V.get()
            logger.info(f"Total products accumulated: {4*np.sum(samples)}")
            logger.info(f"Elapsed time: {elapsed:g} ms")
            throughput = 4*np.sum(samples)/(elapsed/1000)
            logger.info(f"Throughput: {throughput:g} products/sec.")
            return PeriodicSpectrum(self.freq, self.start_time, I, Q, U, V, samples, elapsed)
        else:
            return self

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
        if self.delayed:
            spec = self.compute()
        else:
            spec = self

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        arr = getattr(spec, what)
        arr = fft_roll(arr, shift*spec.nbin)
        if sym_lim:
            vmin, vmax = symmetrize_limits(arr, vmin, vmax)
        pc = ax.pcolormesh(spec.phase - shift, spec.freq/1e6, arr, vmin=vmin, vmax=vmax, **kwargs)
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
def corrfold_numba(A, B, nlag, nbin, binplan, include_end=False):
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

def corrfold_cpu(A, B, nlag, nbin, binplan, include_end=False,
                 n_threads=nb.config.NUMBA_NUM_THREADS):
    """
    Wrap the compiled Numba function to clean up output and measure performance.

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
    timer = CPUTimer()
    with timer, NumbaThreads(n_threads):
        AA, AB, BA, BB, samples = corrfold_numba(
            A, B, nlag, nbin, binplan, include_end
        )
    CR = (AB + BA)/2
    CI = (AB - BA)/np.array(2j).astype(A.dtype)
    elapsed = np.array(timer.elapsed, dtype=np.float64)

    return AA, BB, CR, CI, samples, elapsed

def cycfold_cpu(
    data,
    ncyc,
    nbin,
    phase_predictor,
    include_end=False,
    n_threads=nb.config.NUMBA_NUM_THREADS,
    n_workers=None,
    compute=True,
):
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
    compute: If data are delayed, invoke Dask to compute the output.
          Otherwise, has no effect.
    """
    complex_dtype = data.A.dtype
    logger.debug(f"Input dtype: {complex_dtype}")
    nlag = ncyc//2 + 1

    # construct the bin plan
    t_span = data.n_samples/np.abs(data.bandwidth)
    if data.delayed:
        chunks, = data.A.chunks
        def linspace(*args, **kwargs):
            return da.linspace(*args, chunks=[2*chunk for chunk in chunks], **kwargs)
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
        AA, BB, CR, CI, samples, elapsed = [], [], [], [], [], []
        A = da.overlap.overlap(data.A, depth={0: (0, nlag - 1)}, boundary=None)
        B = da.overlap.overlap(data.B, depth={0: (0, nlag - 1)}, boundary=None)
        binplan = da.overlap.overlap(binplan, depth={0: (0, 2*nlag - 2)}, boundary=None)
        for A_blk, B_blk, plan_blk in zip(A.blocks, B.blocks, binplan.blocks):
            corrfold = dask.delayed(corrfold_cpu, nout=6)
            AA_blk, BB_blk, CR_blk, CI_blk, samples_blk, elapsed_blk = corrfold(
                A_blk, B_blk, nlag, nbin, plan_blk, include_end
            )
            AA.append(da.from_delayed(AA_blk, (nlag, nbin), dtype=data.A.dtype))
            BB.append(da.from_delayed(BB_blk, (nlag, nbin), dtype=data.A.dtype))
            CR.append(da.from_delayed(CR_blk, (nlag, nbin), dtype=data.A.dtype))
            CI.append(da.from_delayed(CI_blk, (nlag, nbin), dtype=data.A.dtype))
            samples.append(da.from_delayed(samples_blk, (nlag, nbin), dtype=np.int64))
            elapsed.append(da.from_delayed(elapsed_blk, (), dtype=np.float64))
        AA = da.mean(da.stack(AA), axis=0)
        BB = da.mean(da.stack(BB), axis=0)
        CR = da.mean(da.stack(CR), axis=0)
        CI = da.mean(da.stack(CI), axis=0)
        samples = da.sum(da.stack(samples), axis=0)
        elapsed = da.sum(da.stack(elapsed), axis=0)
        if compute:
            AA, BB, CR, CI, samples, elapsed = dask.compute(
                AA, BB, CR, CI, samples, elapsed, num_workers=n_workers,
            )
            logger.info(f"Total products accumulated: {4*np.sum(samples)}")
            logger.info(f"Elapsed time in numba: {elapsed:g} ms")
            throughput = 4*np.sum(samples)/(elapsed/1000)
            logger.info(f"Throughput: {throughput:g} products/sec.")
    else:
        AA, BB, CR, CI, samples, elapsed = corrfold_cpu(
            data.A, data.B, nlag, nbin, binplan, include_end
        )
        logger.info(f"Total products accumulated: {4*np.sum(samples)}")
        logger.info(f"Elapsed time: {elapsed:g} ms")
        throughput = 4*np.sum(samples)/(elapsed/1000)
        logger.info(f"Throughput: {throughput:g} products/sec.")
    pspec_AA = np.fft.fftshift(np.fft.hfft(AA, axis=0), axes=0)
    pspec_BB = np.fft.fftshift(np.fft.hfft(BB, axis=0), axes=0)
    pspec_CR = np.fft.fftshift(np.fft.hfft(CR, axis=0), axes=0)
    pspec_CI = np.fft.fftshift(np.fft.hfft(CI, axis=0), axes=0)
    bandwidth = data.bandwidth
    freq = data.obsfreq + np.linspace(-bandwidth/2, bandwidth/2, ncyc, endpoint=False)

    I, Q, U, V = coherence_to_stokes(
        pspec_AA,
        pspec_BB,
        pspec_CR,
        pspec_CI,
        data.feed_poln,
    )
    pspec = PeriodicSpectrum(freq, data.start_time, I, Q, U, V, samples, elapsed)
    return pspec
