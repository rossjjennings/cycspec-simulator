import numpy as np
import dask.array as da
import scipy.fft

from .linear_filter import LinearFilter
from .baseband import BasebandData
from .gpu import have_cuda
if have_cuda:
    import cupy as cp
    import cupyx.scipy.fft

DM_CONSTANT = 1e16/2.41 # Hz**2 s cm**3 pc**-1

class DispersionFilter(LinearFilter):
    def __init__(self, dm, bandwidth, obsfreq=0):
        """
        Create a dispersion filter with a given DM.

        Parameters
        ----------
        dm: dispersion measure (DM, pc cm**-3) to use.
        bandwidth: bandwidth of data the filter is to be applied to
        obsfreq: observing frequency of data the filter is to be applied to
        """
        self.dm = dm
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq

    @property
    def ndm(self):
        hifreq = (self.obsfreq + np.abs(self.bandwidth)/2)
        lofreq = (self.obsfreq - np.abs(self.bandwidth)/2)
        tdm = DM_CONSTANT*self.dm*(1/lofreq**2 - 1/hifreq**2)
        return 2*int(np.ceil(tdm*self.bandwidth/2)) # round to even

    @property
    def nlag_neg(self):
        return self.ndm//2

    @property
    def nlag_pos(self):
        return self.ndm//2

    def filter_function(self, nsamples):
        f = np.fft.fftfreq(nsamples, d=1/self.bandwidth)
        a = DM_CONSTANT*self.dm/self.obsfreq
        x = f/self.obsfreq
        return np.exp(2j*np.pi*a*x**2/(1 + x))

    def apply_block(self, block, device=False):
        H = self.filter_function(block.shape[-1]).astype(block.dtype)
        if device:
            fft = cupyx.scipy.fft
            H = cp.array(H)
        else:
            fft = scipy.fft
        # need scipy fft to avoid dtype promotion
        return fft.ifft(H*fft.fft(block))

    def apply(self, data):
        """
        Apply this dispersion filter to baseband data.
        The returned BasebandData object will be shorter by `self.ndm` samples.
        """
        if data.obsfreq != self.obsfreq or data.bandwidth != self.bandwidth:
            raise ValueError(f"Data observing frequency ({data.obsfreq} Hz) "
                             f"and channel bandwidth ({data.bandwidth} Hz) "
                             "do not match this dispersion filter")

        ndm = self.ndm

        if data.delayed:
            A = da.map_overlap(
                self.apply_block,
                data.A,
                device=data.device,
                depth=ndm//2,
                boundary=None,
                meta=data.A._meta,
            )
            B = da.map_overlap(
                self.apply_block,
                data.B,
                device=data.device,
                depth=ndm//2,
                boundary=None,
                meta=data.B._meta,
            )
            A = A[ndm//2:-ndm//2]
            B = B[ndm//2:-ndm//2]
            t = data.t[ndm//2].compute()
        else:
            new_size = data.A.size - ndm
            A = self.apply_block(data.A)[ndm//2:-ndm//2]
            B = self.apply_block(data.B)[ndm//2:-ndm//2]
            t = data.t[ndm//2]
        return BasebandData(
            A, B, t,
            data.feed_poln,
            data.bandwidth,
            data.obsfreq,
        )
