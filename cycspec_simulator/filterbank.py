import numpy as np
import numba as nb
from scipy import signal

from .baseband import BasebandData
from .interpolation import lerp
from .cycspec import PeriodicSpectrum, cycfold_cpu
from .cuda import have_cuda, cuda_failure
if have_cuda:
    from .cycspec_gpu import cycfold_gpu

def pfb(x, nchan, ntap, window="hamming", fs=1.0):
    """
    Channelize a time series using a polyphase filterbank

    Parameters
    ----------
    x : ndarray
       The input time series.
    nchan : int
       The number of channels to form.
    ntap : int
       The number of PFB taps to use.
    window : str
       The windowing function to use for the PFB coefficients.
    fs : float
       The sampling frequency of the input data.

    Returns
    -------
    x_pfb : ndarray
       The channelized data

    Notes
    -----
    If the input data are real-valued then only positive frequencies
    are returned.
    """
    h = signal.firwin(ntap*nchan,cutoff=1.0/nchan, window="rectangular")
    h *= signal.get_window(window, ntap*nchan)
    nwin = x.shape[0]//ntap//nchan
    x = x[:nwin*ntap*nchan].reshape((nwin*ntap, nchan)).T
    h = h.reshape((ntap, nchan)).T
    xs = np.zeros((nchan, ntap*(nwin-1)+1), dtype=x.dtype)
    for ii in range(ntap*(nwin-1)+1):
        xw = h*x[:, ii:ii+ntap]
        xs[:, ii] = xw.sum(axis=1)
    xs = xs.T
    xpfb = np.fft.fft(xs, nchan, axis=1)
    xpfb *= np.sqrt(nchan)

    return xpfb

def channelize(data, nchan, ntap=24, window='hamming'):
    """
    Channelize a BasebandData object using a polyphase filterbank.

    Parameters
    ----------
    data: BasebandData object
    nchan: Number of channels into which to split the data
    ntap: Number of polyphase fiterbank taps
    window: Window function used for polyphase filterbank
            (string interpreted by `scipy.signal.get_window()`)

    Returns
    -------
    channelized_data: ChannelizedData object
    """
    A_pfb = pfb(data.A, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth)
    B_pfb = pfb(data.B, nchan=nchan, ntap=ntap, window=window, fs=data.bandwidth)
    freqs = np.fft.fftshift(np.fft.fftfreq(nchan, d=1/data.bandwidth))
    freqs += data.obsfreq
    A_pfb = np.fft.fftshift(A_pfb.T, axes=0)
    B_pfb = np.fft.fftshift(B_pfb.T, axes=0)
    return ChannelizedData(
        A_pfb, B_pfb, start_time=data.start_time,
        feed_poln=data.feed_poln,
        chan_bw=data.bandwidth/nchan, freqs=freqs,
    )


class ChannelizedModel:
    """
    A model representing data which has been channelized using a polyphase filterbank.
    """
    def __init__(self, baseband_model, nchan, ntap=24, window='hamming'):
        """
        Create a channelized model.

        Parameters
        ----------
        baseband_model: The underlying baseband data model
        nchan: Number of channels into which to split the data
        ntap: Number of polyphase fiterbank taps
        window: Window function used for polyphase filterbank
                (string interpreted by `scipy.signal.get_window()`)
        """
        self.baseband_model = baseband_model
        self.nchan = nchan
        self.ntap = ntap
        self.window = window

    @property
    def chan_bw(self):
        return self.baseband_model.bandwidth/self.nchan

    def sample(self, n_samples, t_start=None, interp=lerp, dtype=np.float32):
        """
        Simulate a specified number of samples in each channel.

        Parameters
        ----------

        n_samples: The number of samples to use.
        t_start: Time of the first sample, as a Time object. If `None`,
                 the predictor epoch will be used.
        interp: Interpolation function to use. Passed to `BasebandModel.sample()`.
        dtype: Numpy dtype to use for samples.
        """
        data = self.baseband_model.sample(n_samples, t_start, interp, dtype)
        return channelize(data, self.nchan, self.ntap, self.window)

class ChannelizedData:
    """
    Channelized data produced by a polyphase filterbank.
    """
    def __init__(self, A, B, start_time, feed_poln, chan_bw, freqs):
        self.A = A
        self.B = B
        self.start_time = start_time
        self.feed_poln = feed_poln
        self.chan_bw = chan_bw
        self.freqs = freqs

    @property
    def nchan(self):
        return self.A.shape[0]

    @property
    def n_samples(self):
        return self.A.shape[-1]

    @property
    def bandwidth(self):
        return self.nchan*self.chan_bw

    @property
    def tspan(self):
        n_samples = self.A.shape[-1]
        sample_freq = np.abs(self.chan_bw)
        return n_samples/sample_freq

    def extract_channel(self, ichan):
        return BasebandData(
            self.A[ichan], self.B[ichan], start_time=self.start_time,
            feed_poln=self.feed_poln, bandwidth=self.chan_bw, obsfreq=self.freqs[ichan],
        )

    def __getitem__(self, ichan):
        return self.extract_channel(ichan)

    def cycfold(self, ncyc, nbin, predictor, use_cuda=have_cuda,
                   n_threads=nb.config.NUMBA_NUM_THREADS):
        """
        Compute the periodic spectrum from channelized data.

        Parameters
        ----------
        ncyc: Number of cyclic channels per filterbank channel
        nbin: Number of phase bins in which to accumulate
        predictor: `PhasePredictor` object to use in computing phases.
        use_cuda: Whether to use CUDA acceleration. Defaults to True if a CUDA
                  device is detected by CuPy. Otherwise defaults to False.
        n_threads: Number of CPU threads to use. Defaults to the total number of
                   CPUs, as detected by Numba. Has no effect if use_gpu is True.
        """
        if use_cuda and have_cuda:
            cycfold = cycfold_gpu
            cycfold_kwargs = {}
        elif use_cuda:
            err = ValueError("use_cuda was specified, but no CUDA device was found")
            raise err from cuda_failure
        else:
            cycfold = cycfold_cpu
            cycfold_kwargs = {'n_threads': n_threads}

        freq = np.zeros(ncyc*self.nchan)
        I = np.zeros((ncyc*self.nchan, nbin))
        Q = np.zeros((ncyc*self.nchan, nbin))
        U = np.zeros((ncyc*self.nchan, nbin))
        V = np.zeros((ncyc*self.nchan, nbin))
        chan_data = self.extract_channel(0)
        pspec = cycfold(chan_data, ncyc, nbin, predictor, **cycfold_kwargs)
        bot = slice(0, ncyc - ncyc//2)
        top = slice(-ncyc//2, None)
        freq[top] = pspec.freq[:ncyc//2] + self.bandwidth
        freq[bot] = pspec.freq[ncyc//2:]
        I[top] = pspec.I[:ncyc//2]
        I[bot] = pspec.I[ncyc//2:]
        Q[top] = pspec.Q[:ncyc//2]
        Q[bot] = pspec.Q[ncyc//2:]
        U[top] = pspec.U[:ncyc//2]
        U[bot] = pspec.U[ncyc//2:]
        V[top] = pspec.V[:ncyc//2]
        V[bot] = pspec.V[ncyc//2:]
        for ichan in range(1, self.nchan):
            chan_data = self.extract_channel(ichan)
            pspec = cycfold(chan_data, ncyc, nbin, predictor, **cycfold_kwargs)
            sl = slice((ichan-1)*ncyc + ncyc//2, ichan*ncyc + ncyc//2)
            freq[sl] = pspec.freq
            I[sl] = pspec.I
            Q[sl] = pspec.Q
            U[sl] = pspec.U
            V[sl] = pspec.V
        return PeriodicSpectrum(freq, self.start_time, I, Q, U, V)
