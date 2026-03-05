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
from .gpu import get_current_device

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

@cuda.jit()
def corrfold_kernel(A, B, nbin, binplan, n_samples, AA, AB, BA, BB, include_end=False):
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
    AA, AB, BA, BB: Output arrays which will be used to hold each of the polarization
          components of the result, for both real and imaginary part.
          These should have shape (nbin, nlag, 2), where the last axis (of length 2)
          will hold the real and imaginary part at index 0 and 1, respectively.
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
        ncorr = A.shape[0] - ilag
    else:
        ncorr = A.shape[0] - nlag + 1

    # grid-stride loop
    for icorr in range(ithread, ncorr, nthreads):
        ibin = binplan[2*icorr + ilag]
        cuda.atomic.add(n_samples, (ilag, ibin), 1)
        product_AA = (A[icorr + ilag] * A[icorr].conjugate())
        cuda.atomic.add(AA, (ilag, ibin, 0), product_AA.real)
        cuda.atomic.add(AA, (ilag, ibin, 1), product_AA.imag)
        product_AB = (A[icorr + ilag] * B[icorr].conjugate())
        cuda.atomic.add(AB, (ilag, ibin, 0), product_AB.real)
        cuda.atomic.add(AB, (ilag, ibin, 1), product_AB.imag)
        product_BA = (B[icorr + ilag] * A[icorr].conjugate())
        cuda.atomic.add(BA, (ilag, ibin, 0), product_BA.real)
        cuda.atomic.add(BA, (ilag, ibin, 1), product_BA.imag)
        product_BB = (B[icorr + ilag] * B[icorr].conjugate())
        cuda.atomic.add(BB, (ilag, ibin, 0), product_BB.real)
        cuda.atomic.add(BB, (ilag, ibin, 1), product_BB.imag)

@cuda.jit(device=True)
def warpagg_add_pair(arr, indices, val1, val2):
    # create mask of all peer threads with higher laneid
    ravel_index = indices[0]*arr.shape[0] + arr.shape[1]
    mask = cuda.match_any_sync(cuda.activemask(), ravel_index) & cuda.activemask()
    upper = mask & (~((2<<cuda.laneid) - 1))

    # we will be copying values from src_lane using shfl_sync()
    src_lane = cuda.ffs(upper) - 1
    active = upper != 0
    for i in range(5): # 2**5 = 32
        active = active and 0 <= src_lane < 32
        # all calls to shfl_sync() must be unconditional to avoid weird behavior
        new_val1 = cuda.shfl_sync(mask, val1, src_lane)
        new_val2 = cuda.shfl_sync(mask, val2, src_lane)
        new_src_lane = cuda.shfl_sync(mask, src_lane, src_lane)
        if active:
            val1 += new_val1
            val2 += new_val2
            src_lane = new_src_lane

    # only the leader thread does the final atomic adds
    if cuda.laneid == cuda.ffs(mask) - 1:
        cuda.atomic.add(arr, (*indices, 0), val1)
        cuda.atomic.add(arr, (*indices, 1), val2)

@cuda.jit(device=True)
def warpagg_add_count(arr, indices):
    # create mask of all peer threads with higher laneid
    ravel_index = indices[0]*arr.shape[0] + arr.shape[1]
    mask = cuda.match_any_sync(cuda.activemask(), ravel_index) & cuda.activemask()

    # no need to do a complex warp reduction when `popc()` will do
    count = cuda.popc(mask)
    # leader thread adds the count
    if cuda.laneid == cuda.ffs(mask) - 1:
        cuda.atomic.add(arr, indices, count)

@cuda.jit()
def corrfold_kernel_warpagg(A, B, nbin, binplan, n_samples, AA, AB, BA, BB, include_end=False):
    """
    Compute the cyclic autocorrelation function from sampled data, using CUDA.
    This CUDA kernel is intended to be used internally by cycfold_gpu().
    This version is optimized using warp aggregation.

    Parameters
    ----------
    A, B: Baseband samples in each of two polarizations (each of length n)
    nbin: Number of phase bins in which to accumulate
    binplan: Array giving the phase bin corresponding to each half-sample time
          (length 2*n - 1, where n is the number of samples)
    n_samples: Output array which will be used to hold the number of samples
          accumulated into each phase bin for each lag.
    AA, AB, BA, BB: Output arrays which will be used to hold each of the polarization
          components of the result, for both real and imaginary part.
          These should have shape (nbin, nlag, 2), where the last axis (of length 2)
          will hold the real and imaginary part at index 0 and 1, respectively.
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
        ncorr = A.shape[0] - ilag
    else:
        ncorr = A.shape[0] - nlag + 1

    # grid-stride loop
    for icorr in range(ithread, ncorr, nthreads):
        ibin = binplan[2*icorr + ilag]
        warpagg_add_count(n_samples, (ilag, ibin))
        product_AA = (A[icorr + ilag] * A[icorr].conjugate())
        warpagg_add_pair(AA, (ilag, ibin), product_AA.real, product_AA.imag)
        product_AB = (A[icorr + ilag] * B[icorr].conjugate())
        warpagg_add_pair(AB, (ilag, ibin), product_AB.real, product_AB.imag)
        product_BA = (B[icorr + ilag] * A[icorr].conjugate())
        warpagg_add_pair(BA, (ilag, ibin), product_BA.real, product_BA.imag)
        product_BB = (B[icorr + ilag] * B[icorr].conjugate())
        warpagg_add_pair(BB, (ilag, ibin), product_BB.real, product_BB.imag)

def corrfold_gpu(A, B, nlag, nbin, binplan, stream, include_end=False, use_warpagg=True):
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
    use_warpagg: Use the optimized kernel with warp aggregated atomic adds.
    """
    complex_dtype = A.dtype
    real_dtype = A.real.dtype
    A_gpu = cupy.array(A)
    B_gpu = cupy.array(B)
    binplan = cupy.array(binplan)
    samples = cupy.zeros((nlag, nbin), dtype=np.int32)
    AA = cupy.zeros((nlag, nbin, 2), dtype=real_dtype)
    AB = cupy.zeros((nlag, nbin, 2), dtype=real_dtype)
    BA = cupy.zeros((nlag, nbin, 2), dtype=real_dtype)
    BB = cupy.zeros((nlag, nbin, 2), dtype=real_dtype)

    # Determine block size to use based on device attributes
    device = get_current_device()
    blocksize = np.gcd(device.max_threads_per_block, device.max_threads_per_multiprocessor)
    logger.debug(f"Using blocksize {blocksize}")

    # Determine number of blocks, targeting 32 per multiprocessor
    in_samples, = A.shape
    nblocks_target = device.multiprocessor_count * 32
    nblocks_span = int(np.ceil(in_samples/blocksize))
    nblocks = min(int(np.ceil(nblocks_target/nlag)), nblocks_span)
    logger.debug(f"Number of lags: {nlag}")
    logger.debug(f"Number of blocks per lag for full occupancy: {nblocks}")

    with CUDATimer(stream) as cudatimer:
        if use_warpagg:
            corrfold_kernel_warpagg[nlag, blocksize, stream](
                A_gpu, B_gpu, nbin, binplan, samples, AA, AB, BA, BB, include_end,
            )
        else:
            corrfold_kernel[nlag, blocksize, stream](
                A_gpu, B_gpu, nbin, binplan, samples, AA, AB, BA, BB, include_end,
            )
    elapsed = np.array(cudatimer.elapsed, dtype=np.float64)

    i = cupy.array(1j, dtype=complex_dtype)
    AA = (AA[..., 0] + i*AA[..., 1])/samples
    BB = (BB[..., 0] + i*BB[..., 1])/samples
    CR = (AB[..., 0] + BA[..., 0] + i*(AB[..., 1] + BA[..., 1]))/(2*samples)
    CI = (AB[..., 0] - BA[..., 1] + i*(AB[..., 1] - BA[..., 1]))/(2*i*samples)
    samples = samples.reshape(nlag, nbin)

    return AA, BB, CR, CI, samples, elapsed

def cycfold_gpu(data, ncyc, nbin, phase_predictor, include_end=False, n_workers=None, use_warpagg=False):
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
    use_warpagg: Use the optimized kernel with warp aggregated atomic adds.
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
        logger.debug("Computing using Dask delayed")
        A = da.overlap.overlap(data.A, depth={0: (0, nlag - 1)}, boundary=None)
        B = da.overlap.overlap(data.B, depth={0: (0, nlag - 1)}, boundary=None)
        binplan = da.overlap.overlap(binplan, depth={0: (0, 2*nlag - 2)}, boundary=None)
        AA, BB, CR, CI, samples, elapsed = [], [], [], [], [], []
        for A_blk, B_blk, plan_blk in zip(A.blocks, B.blocks, binplan.blocks):
            corrfold = dask.delayed(corrfold_gpu, nout=6)
            AA_blk, BB_blk, CR_blk, CI_blk, samples_blk, elapsed_blk = corrfold(
                A_blk, B_blk, nlag, nbin, plan_blk, stream, include_end, use_warpagg,
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
        logger.debug("Computing directly, not delayed")
        AA, BB, CR, CI, samples, elapsed = corrfold_gpu(
            data.A, data.B, nlag, nbin, binplan, stream, include_end, use_warpagg,
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
