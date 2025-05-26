import numpy as np
import numba as nb
from numba import cuda
import dask
import dask.array as da
import cupy
from loguru import logger

from .polarization import coherence_to_stokes
from .time import Time
from .cycspec import PeriodicSpectrum

class CUDATimer:
    """
    A context manager for timing CUDA code using Events.
    Borrowed from Carlos Costa, "CUDA by Numba Examples",
    https://towardsdatascience.com/cuda-by-numba-examples-7652412af1ee
    """
    def __init__(self, stream):
        self.stream = stream
        self.elapsed = None # elapsed time in ms

    def __enter__(self):
        self.event_begin = cuda.event()
        self.event_end = cuda.event()
        self.event_begin.record(stream=self.stream)
        return self

    def __exit__(self, type, value, traceback):
        self.event_end.record(stream=self.stream)
        self.event_end.wait(stream=self.stream)
        self.event_end.synchronize()
        self.elapsed = self.event_begin.elapsed_time(self.event_end)

signatures = [
    (
        nb.complex64[::1],
        nb.complex64[::1],
        nb.int64,
        nb.int64[::1],
        nb.int32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.boolean,
    ),
    (
        nb.complex128[::1],
        nb.complex128[::1],
        nb.int64,
        nb.int64[::1],
        nb.int32[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.float64[::1],
        nb.boolean,
    ),
]

@cuda.jit(signatures)
def corrfold_kernel(A, B, nbin, binplan, n_samples,
                    AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag,
                    include_end=False):
    """
    Compute the cyclic autocorrelation function from sampled data, using CUDA.
    This CUDA kernel is intended to be used internally by cycfold_gpu().

    Parameters
    ----------
    A, B: Baseband samples in each of two polarizations (each of length n)
    nbin: Number of phase bins in which to accumulate
    binplan: Array giving the phase bin corresponding to each half-sample time
          (length 2*n - 1, where n is the number of samples)
    n_samples: Output array which will be used to hold the number of samples
          accumulated into each phase bin for each lag.
    AA_real, etc.: Output arrays which will be used to hold each of the polarization
          components of the result, for both real and imaginary part.
    include_end: Whether to calculate products where the first sample is among
          the last nlag - ilag - 1 samples. In such cases, there are fewer than
          nlag choices for the second sample. Setting include_end=True means that
          slightly more samples will contribute to lower lags.
    """
    ilag = cuda.blockIdx.x
    nlag = cuda.gridDim.x
    ithread = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    if include_end:
        ncorr = A.size - ilag
    else:
        ncorr = A.size - nlag + 1

    for icorr in range(ithread, ncorr, nthreads):
        ibin = binplan[2*icorr + ilag]
        ibuf = ilag*nbin + ibin
        cuda.atomic.add(n_samples, ibuf, 1)
        product_AA = (A[icorr + ilag] * A[icorr].conjugate())
        cuda.atomic.add(AA_real, ibuf, product_AA.real)
        cuda.atomic.add(AA_imag, ibuf, product_AA.imag)
        product_AB = (A[icorr + ilag] * B[icorr].conjugate())
        cuda.atomic.add(AB_real, ibuf, product_AB.real)
        cuda.atomic.add(AB_imag, ibuf, product_AB.imag)
        product_BA = (B[icorr + ilag] * A[icorr].conjugate())
        cuda.atomic.add(BA_real, ibuf, product_BA.real)
        cuda.atomic.add(BA_imag, ibuf, product_BA.imag)
        product_BB = (B[icorr + ilag] * B[icorr].conjugate())
        cuda.atomic.add(BB_real, ibuf, product_BB.real)
        cuda.atomic.add(BB_imag, ibuf, product_BB.imag)

def corrfold_gpu(A, B, nlag, nbin, binplan, stream, include_end=False):
    """
    Wrap the CUDA kernel into something more directly analogous to corrfold_cpu.
    Copies data to GPU, allocates GPU memory, invokes the kernel, and cleans up output.
    Dask doesn't understand out parameters and so can't invoke the kernel directly,
    but it can invoke this function.

    Parameters
    ----------
    A, B: Baseband samples in each of two polarizations (each of length n)
    nlag: Number of lags to use for the correlation
    nbin: Number of phase bins in which to accumulate
    binplan: Array giving the phase bin corresponding to each half-sample time
          (length 2*n, where n is the number of samples)
    stream: CUDA stream to use
    include_end: Whether to calculate products where the first sample is among
          the last nlag - ilag - 1 samples. In such cases, there are fewer than
          nlag choices for the second sample. Setting include_end=True means that
          slightly more samples will contribute to lower lags.
    """
    complex_dtype = A.dtype
    real_dtype = A.real.dtype
    A_gpu = cupy.array(A)
    B_gpu = cupy.array(B)
    binplan = cupy.array(binplan)
    samples = cupy.zeros(nlag*nbin, dtype=np.int32)
    AA_real = cupy.zeros(nlag*nbin, dtype=real_dtype)
    AA_imag = cupy.zeros(nlag*nbin, dtype=real_dtype)
    AB_real = cupy.zeros(nlag*nbin, dtype=real_dtype)
    AB_imag = cupy.zeros(nlag*nbin, dtype=real_dtype)
    BA_real = cupy.zeros(nlag*nbin, dtype=real_dtype)
    BA_imag = cupy.zeros(nlag*nbin, dtype=real_dtype)
    BB_real = cupy.zeros(nlag*nbin, dtype=real_dtype)
    BB_imag = cupy.zeros(nlag*nbin, dtype=real_dtype)

    # Number of threads per CUDA thread block.
    # Turing has 1024 threads per SM, Ampere has 1536. 512 is the gcd of these,
    # so should make it possible to achieve full occupancy on either.
    nthreads_block = 512

    with CUDATimer(stream) as cudatimer:
        corrfold_kernel[nlag, nthreads_block, stream](
            A_gpu, B_gpu, nbin, binplan, samples,
            AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag,
            include_end
        )
    elapsed = np.array(cudatimer.elapsed, dtype=np.float64)

    i = cupy.array(1j, dtype=complex_dtype)
    AA = (AA_real + i*AA_imag)/samples
    AA = AA.reshape(nlag, nbin)
    BB = (BB_real + i*BB_imag)/samples
    BB = BB.reshape(nlag, nbin)
    CR = (AB_real + BA_real + i*(AB_imag + BA_imag))/(2*samples)
    CR = CR.reshape(nlag, nbin)
    CI = (AB_real - BA_real + i*(AB_imag - BA_imag))/(2*i*samples)
    CI = CI.reshape(nlag, nbin)
    samples = samples.reshape(nlag, nbin)

    return AA, BB, CR, CI, samples, elapsed

def cycfold_gpu(data, ncyc, nbin, phase_predictor, include_end=False, n_workers=None):
    """
    Compute the periodic spectrum from sampled data, using CUDA.

    Parameters
    ----------
    data: BasebandData object containing the data to use
    ncyc: Number of "cyclic channels" per input channel
    nbin: Number of phase bins in which to accumulate
    phase_predictor: Predictor object to use in computing phases. Should have
          a phase() method which can be called with a Time object to yield the
          corresponding array of phases.
    include_end: Passed along to corrfold_gpu(), see there for details.
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

    stream = cuda.stream()
    cuda.profile_start()
    if data.delayed:
        A = da.overlap.overlap(data.A, depth={0: (0, nlag - 1)}, boundary=None)
        B = da.overlap.overlap(data.B, depth={0: (0, nlag - 1)}, boundary=None)
        binplan = da.overlap.overlap(binplan, depth={0: (0, 2*nlag - 2)}, boundary=None)
        AA, BB, CR, CI, samples, elapsed = [], [], [], [], [], []
        for A_blk, B_blk, plan_blk in zip(A.blocks, B.blocks, binplan.blocks):
            corrfold = dask.delayed(corrfold_gpu, nout=6)
            AA_blk, BB_blk, CR_blk, CI_blk, samples_blk, elapsed_blk = corrfold(
                A_blk, B_blk, nlag, nbin, plan_blk, stream, include_end
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
        AA, BB, CR, CI, samples, elapsed = dask.compute(
            AA, BB, CR, CI, samples, elapsed, num_workers=n_workers
        )
        logger.info(f"Total products accumulated: {4*np.sum(samples)}")
        logger.info(f"Elapsed time in kernel: {elapsed:g} ms")
        throughput = 4*np.sum(samples)/(elapsed/1000)
        logger.info(f"Throughput: {throughput:g} products/sec.")
    else:
        AA, BB, CR, CI, samples, elapsed = corrfold_gpu(
            data.A, data.B, nlag, nbin, binplan, stream, include_end
        )
        logger.info(f"Elapsed time: {elapsed:g} ms")
        logger.info(f"Total products accumulated: {4*np.sum(samples)}")
        throughput = 4*np.sum(samples)/(elapsed/1000)
        logger.info(f"Throughput: {throughput:g} products/sec.")
    cuda.profile_stop()

    pspec_AA = np.fft.fftshift(np.fft.hfft(AA.get(), axis=0), axes=0)
    pspec_AA = pspec_AA.reshape(ncyc, nbin)
    pspec_BB = np.fft.fftshift(np.fft.hfft(BB.get(), axis=0), axes=0)
    pspec_BB = pspec_BB.reshape(ncyc, nbin)
    pspec_CR = np.fft.fftshift(np.fft.hfft(CR.get(), axis=0), axes=0)
    pspec_CR = pspec_CR.reshape(ncyc, nbin)
    pspec_CI = np.fft.fftshift(np.fft.hfft(CI.get(), axis=0), axes=0)
    pspec_CI = pspec_CI.reshape(ncyc, nbin)
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
