import numpy as np
import cupy
import numba as nb
from numba import cuda

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

@cuda.jit((nb.complex64[:,:], nb.complex64[:,:], nb.int64, nb.int64[:], nb.int32[:],
           nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
           nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]))
def _cycfold_gpu(A, B, nbin, binplan, n_samples,
                 AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag):
    ilag = cuda.blockIdx.x
    nlag = cuda.gridDim.x
    ichan = cuda.blockIdx.y
    ithread = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    for icorr in range(ithread, A.shape[1] - ilag, nthreads):
        ibin = binplan[2*icorr + ilag]
        ibuf = ichan*nlag*nbin + ilag*nbin + ibin
        cuda.atomic.add(n_samples, ibuf, 1)
        product_AA = (A[ichan, icorr + ilag] * A[ichan, icorr].conjugate())
        cuda.atomic.add(AA_real, ibuf, product_AA.real)
        cuda.atomic.add(AA_imag, ibuf, product_AA.imag)
        product_AB = (A[ichan, icorr + ilag] * B[ichan, icorr].conjugate())
        cuda.atomic.add(AB_real, ibuf, product_AB.real)
        cuda.atomic.add(AB_imag, ibuf, product_AB.imag)
        product_BA = (B[ichan, icorr + ilag] * A[ichan, icorr].conjugate())
        cuda.atomic.add(BA_real, ibuf, product_BA.real)
        cuda.atomic.add(BA_imag, ibuf, product_BA.imag)
        product_BB = (B[ichan, icorr + ilag] * B[ichan, icorr].conjugate())
        cuda.atomic.add(BB_real, ibuf, product_BB.real)
        cuda.atomic.add(BB_imag, ibuf, product_BB.imag)

def cycfold_gpu(data, ncyc, nbin, phase_predictor):
    nlag = ncyc//2 + 1
    A_gpu = cupy.array(data.A)
    B_gpu = cupy.array(data.B)

    # construct the bin plan
    offset = np.empty(2*data.t.offset.size - 1)
    offset[::2] = data.t.offset
    offset[1::2] = (data.t.offset[1:] + data.t.offset[:-1])/2
    t = Time(data.t.mjd, data.t.second, offset)
    phase = phase_predictor.phase(t)
    binplan = np.int64(np.round((phase % 1)*nbin)) % nbin
    binplan = cupy.array(binplan)

    n_samples = cupy.zeros(data.nchan*nlag*nbin, dtype=np.int32)
    AA_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AA_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AB_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AB_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BA_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BA_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BB_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BB_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    stream = cuda.stream()
    with CUDATimer(stream) as cudatimer:
        cuda.profile_start()
        _cycfold_gpu[(nlag, data.nchan), 512, stream](
            A_gpu, B_gpu, nbin, binplan, n_samples,
            AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag
        )
        cuda.profile_stop()
    print(f"Elapsed time: {cudatimer.elapsed:g} ms")
    print(f"Total products accumulated: {4*np.sum(n_samples)}")
    throughput = 4*np.sum(n_samples)/(cudatimer.elapsed/1000)
    print(f"Throughput: {throughput:g} products/sec.")

    AA = (AA_real + 1j*AA_imag)/n_samples
    AA = AA.reshape(data.nchan, nlag, nbin)
    BB = (BB_real + 1j*BB_imag)/n_samples
    BB = BB.reshape(data.nchan, nlag, nbin)
    CR = (AB_real + BA_real + 1j*(AB_imag + BA_imag))/(2*n_samples)
    CR = CR.reshape(data.nchan, nlag, nbin)
    CI = (AB_real - BA_real + 1j*(AB_imag - BA_imag))/(2j*n_samples)
    CI = CI.reshape(data.nchan, nlag, nbin)
    pspec_AA = np.fft.fftshift(np.fft.hfft(AA.get(), axis=1), axes=1)
    pspec_AA = pspec_AA.reshape(data.nchan*ncyc, nbin)
    pspec_BB = np.fft.fftshift(np.fft.hfft(BB.get(), axis=1), axes=1)
    pspec_BB = pspec_BB.reshape(data.nchan*ncyc, nbin)
    pspec_CR = np.fft.fftshift(np.fft.hfft(CR.get(), axis=1), axes=1)
    pspec_CR = pspec_CR.reshape(data.nchan*ncyc, nbin)
    pspec_CI = np.fft.fftshift(np.fft.hfft(CI.get(), axis=1), axes=1)
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

def cycfold_gpu_sharedmem(data, ncyc, nbin, phase_predictor):
    nlag = ncyc//2 + 1
    threads_per_block = 512
    shared_binplan_size = 2*threads_per_block + nlag - 2
    shared_samples_size = threads_per_block + nlag - 1

    @cuda.jit
    def _cycfold_gpu_sharedmem(A, B, nbin, binplan, n_samples,
                               AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag):
        iblock = cuda.blockIdx.x
        nblocks = cuda.gridDim.x
        ichan = cuda.blockIdx.y
        ithread = cuda.threadIdx.x
        nthreads = cuda.blockDim.x
        igrid = iblock*nthreads + ithread

        binplan_shared = cuda.shared.array((shared_binplan_size,), dtype=nb.int32)
        binplan_shared[ithread] = binplan[2*iblock*nthreads + ithread]
        binplan_shared[nthreads + ithread] = binplan[(2*iblock + 1)*nthreads + ithread]
        if ithread < nlag - 2 and (2*iblock + 2)*nthreads + ithread < binplan.size:
            binplan_shared[2*nthreads + ithread] = binplan[(2*iblock + 2)*nthreads + ithread]

        A_shared = cuda.shared.array((shared_samples_size,), dtype=nb.complex64)
        B_shared = cuda.shared.array((shared_samples_size,), dtype=nb.complex64)
        A_shared[ithread] = A[ichan, igrid]
        B_shared[ithread] = B[ichan, igrid]
        if ithread < nlag - 1 and igrid + nthreads < A.shape[1]:
            A_shared[ithread + nthreads] = A[ichan, igrid + nthreads]
            B_shared[ithread + nthreads] = B[ichan, igrid + nthreads]
        cuda.syncthreads()

        for ilag in range(nlag):
            ibin = binplan_shared[2*ithread + ilag]
            ibuf = ichan*nlag*nbin + ilag*nbin + ibin
            if igrid + ilag < A.shape[1]:
                cuda.atomic.add(n_samples, ibuf, 1)
                product_AA = (A_shared[ithread + ilag] * A_shared[ithread].conjugate())
                cuda.atomic.add(AA_real, ibuf, product_AA.real)
                cuda.atomic.add(AA_imag, ibuf, product_AA.imag)
                product_AB = (A_shared[ithread + ilag] * B_shared[ithread].conjugate())
                cuda.atomic.add(AB_real, ibuf, product_AB.real)
                cuda.atomic.add(AB_imag, ibuf, product_AB.imag)
                product_BA = (B_shared[ithread + ilag] * A_shared[ithread].conjugate())
                cuda.atomic.add(BA_real, ibuf, product_BA.real)
                cuda.atomic.add(BA_imag, ibuf, product_BA.imag)
                product_BB = (B_shared[ithread + ilag] * B_shared[ithread].conjugate())
                cuda.atomic.add(BB_real, ibuf, product_BB.real)
                cuda.atomic.add(BB_imag, ibuf, product_BB.imag)

    A_gpu = cupy.array(data.A)
    B_gpu = cupy.array(data.B)

    # construct the bin plan
    offset = np.empty(2*data.t.offset.size - 1)
    offset[::2] = data.t.offset
    offset[1::2] = (data.t.offset[1:] + data.t.offset[:-1])/2
    t = Time(data.t.mjd, data.t.second, offset)
    phase = phase_predictor.phase(t)
    binplan = np.int64(np.round((phase % 1)*nbin)) % nbin
    binplan = cupy.array(binplan)

    nblocks = int(np.ceil(data.A.shape[1]/threads_per_block))
    n_samples = cupy.zeros(data.nchan*nlag*nbin, dtype=np.int32)
    AA_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AA_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AB_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    AB_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BA_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BA_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BB_real = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    BB_imag = cupy.zeros(data.nchan*nlag*nbin, dtype=np.float32)
    stream = cuda.stream()
    with CUDATimer(stream) as cudatimer:
        cuda.profile_start()
        _cycfold_gpu_sharedmem[(nblocks, data.nchan), threads_per_block, stream](
            A_gpu, B_gpu, nbin, binplan, n_samples,
            AA_real, AA_imag, AB_real, AB_imag, BA_real, BA_imag, BB_real, BB_imag
        )
        cuda.profile_stop()
    print(f"Elapsed time: {cudatimer.elapsed:g} ms")
    print(f"Total products accumulated: {4*np.sum(n_samples)}")
    throughput = 4*np.sum(n_samples)/(cudatimer.elapsed/1000)
    print(f"Throughput: {throughput:g} products/sec.")

    AA = (AA_real + 1j*AA_imag)/n_samples
    AA = AA.reshape(data.nchan, nlag, nbin)
    BB = (BB_real + 1j*BB_imag)/n_samples
    BB = BB.reshape(data.nchan, nlag, nbin)
    CR = (AB_real + BA_real + 1j*(AB_imag + BA_imag))/(2*n_samples)
    CR = CR.reshape(data.nchan, nlag, nbin)
    CI = (AB_real - BA_real + 1j*(AB_imag - BA_imag))/(2j*n_samples)
    CI = CI.reshape(data.nchan, nlag, nbin)
    pspec_AA = np.fft.fftshift(np.fft.hfft(AA.get(), axis=1), axes=1)
    pspec_AA = pspec_AA.reshape(data.nchan*ncyc, nbin)
    pspec_BB = np.fft.fftshift(np.fft.hfft(BB.get(), axis=1), axes=1)
    pspec_BB = pspec_BB.reshape(data.nchan*ncyc, nbin)
    pspec_CR = np.fft.fftshift(np.fft.hfft(CR.get(), axis=1), axes=1)
    pspec_CR = pspec_CR.reshape(data.nchan*ncyc, nbin)
    pspec_CI = np.fft.fftshift(np.fft.hfft(CI.get(), axis=1), axes=1)
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
