import numpy as np
from scipy import signal

from .baseband import BasebandData
from .interpolation import lerp

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

    def extract_channel(self, ichan):
        return BasebandData(
            self.A[ichan], self.B[ichan], start_time=self.start_time,
            feed_poln=self.feed_poln, bandwidth=self.chan_bw, obsfreq=self.freqs[ichan],
        )

    def __getitem__(self, ichan):
        return self.extract_channel(ichan)
