import numpy as np
import dask.array as da
from scipy import signal
import matplotlib.pyplot as plt

from .channelized import ChannelizedData

class PolyphaseFilterbank:
    def __init__(self, nchan, ntap, bandwidth, obsfreq=0, window="hamming", shift=True):
        """
        Create a polyphase filterbank.

        Parameters
        ----------
        nchan: Number of frequency channels in which to split the data.
        ntap: Number of polyphase filter taps to use.
        bandwidth: Bandwidth of input data, in Hz.
        obsfreq: Center frequency in Hz.
        window: Window function to use.
        """
        self.nchan = nchan
        self.ntap = ntap
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq
        self.window = window
        self.shift = shift
        t = np.linspace(-self.ntap/2, self.ntap/2, self.ntap*self.nchan, endpoint=False)
        h = np.ones(t.shape, dtype=np.result_type(t, 1j))
        h *= np.sinc(t)
        if shift:
            if nchan % 2 == 0:
                h *= np.exp(-1j*self.nchan*np.pi*t)
            else:
                h *= np.exp(-1j*(self.nchan-1)*np.pi*t)
        h *= signal.get_window(self.window, self.ntap*self.nchan)
        self.filter_coeffs = h.reshape(ntap, nchan)

    def freq_response(self, n_freq):
        """
        Get the frequency response of this polyphase filterbank.

        Parameters
        ----------
        n_freq: Number of frequencies at which to calculate the response.
        """
        T = self.nchan/self.bandwidth
        t = np.linspace(-T/2*self.ntap, T/2*self.ntap, self.ntap*self.nchan, endpoint=False)
        f = np.linspace(-self.bandwidth/2, self.bandwidth/2, n_freq)
        x = np.exp(-2j*np.pi*f[:, np.newaxis]*t).reshape(-1, self.ntap, self.nchan)
        x_fold = np.sum(x*self.filter_coeffs, axis=1)
        x_pfb = np.fft.fft(x_fold, axis=1)
        spec = np.abs(x_pfb/self.nchan)
        f += self.obsfreq

        return f, spec

    def plot_freq_response(self, n_freq=2049, ax=None, hl_last=False, show=True):
        """
        Calculate and plot the frequency response of this polyphase filterbank.

        Parameters
        ----------
        n_freq: Number of frequencies at which to calculate the response.
        ax: Matplotlib `Axes` object to use for plotting. If `None`, will be created.
        hl_last: Whether to highlight the last channel in the plot (in addition to the first).
        show: If `True`, call `plt.show()`. Set to `False` if this is part of a larger plot.
        """
        f, spec = self.freq_response(n_freq)
        if ax is None:
            fig, ax = plt.subplots()

        freq_magnitude = int(np.clip(np.log10(self.bandwidth)//3, -10, 10))
        freq_prefix = metric_prefixes[freq_magnitude]
        freq_factor = 10**(3*freq_magnitude)
        artists = []
        if hl_last:
            non_hl_range = range(1, spec.shape[1]-1)
        else:
            non_hl_range = range(1, spec.shape[1])
        for i in non_hl_range:
            artists.extend(ax.plot(f/freq_factor, spec[:, i], color='C7', alpha=0.2))
        artists.extend(ax.plot(f/freq_factor, spec[:, 0], color='C0', label='$k=0$'))
        if hl_last:
            artists.extend(
                ax.plot(f/freq_factor, spec[:, -1], color='C1', label=f'$k={spec.shape[1]-1}$')
            )
        ax.set_xlabel(f"Frequency ({freq_prefix}Hz)")
        ax.set_ylabel("Response")
        ax.set_title(
            f"Polyphase filterbank ({self.nchan} channels, {self.ntap} taps), "
            f"{self.window.capitalize()} window"
        )
        if self.shift and hl_last:
            loc = 'upper center'
        else:
            loc = 'upper right'
        ax.legend(loc=loc)
        if show:
            plt.tight_layout()
            plt.show()

    def channelize(self, data, rechunk=True):
        """
        Channelize baseband data using this polyphase filterbank.

        Parameters
        ----------
        data: BasebandData object
        nchan: Number of channels into which to split the data
        ntap: Number of polyphase fiterbank taps
        window: Window function used for polyphase filterbank
                (string interpreted by `scipy.signal.get_window()`)
        rechunk: Whether to re-chunk the input arrays so that the
                chunk size is a multiple of `nchan`. Without this,
                some samples will be skipped at block boundaries,
                leading to a slight drift.

        Returns
        -------
        channelized_data: ChannelizedData object
        """
        bw = data.bandwidth
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nchan, d=1/data.bandwidth))
        freqs += data.obsfreq
        if data.obsfreq != self.obsfreq or data.bandwidth != self.bandwidth:
            raise ValueError(f"Data observing frequency ({data.obsfreq} Hz) "
                             f"and channel bandwidth ({data.bandwidth} Hz) "
                             "do not match this filterbank")
        ns_chan = data.A.shape[0]//self.nchan
        start_time = data.t[self.nchan*self.ntap//2]
        h = self.filter_coeffs

        def process_block(block):
            out = np.zeros((block.shape[0]-self.ntap+1, self.nchan), dtype=block.dtype)
            for ichan in range(self.nchan):
                out[:,ichan] = signal.correlate(block[:,ichan], h[:,ichan], mode="valid")
            out = np.fft.fft(out, axis=1)
            return out

        if data.delayed:
            if rechunk and any(chunk % self.nchan for chunk in data.chunks[0]):
                new_chunk_size = int(np.ceil(max(data.A.chunks[0])/self.nchan))*self.nchan
                A = data.A.rechunk(new_chunk_size)
                B = data.B.rechunk(new_chunk_size)
            else:
                A = data.A
                B = data.B
            A = A[:ns_chan*self.nchan].reshape((ns_chan, self.nchan))
            B = B[:ns_chan*self.nchan].reshape((ns_chan, self.nchan))

            depth = self.nchan*(self.ntap - 1)
            A = da.map_overlap(
                process_block, A, depth={0: (depth, 0)}, dtype=data.A.dtype,
            )
            B = da.map_overlap(
                process_block, B, depth={0: (depth, 0)}, dtype=data.B.dtype,
            )
            start_time = start_time.compute()
        else:
            A = process_block(data.A)
            B = process_block(data.B)
        return ChannelizedData(
            A.T, B.T,
            start_time=start_time,
            feed_poln=data.feed_poln,
            chan_bw=data.bandwidth/self.nchan, freqs=freqs,
        )

metric_prefixes = {
    0: "",
    1: "k", 2: "M", 3: "G", 4: "T", 5: "P", 6: "E", 7: "Z", 8: "Y", 9: "R", 10: "Q",
    -1: "m", -2: "μ", -3: "n", -4: "p", -5: "f", -6: "a", -7: "z", -8: "y", -9: "r", -10: "q",
}
