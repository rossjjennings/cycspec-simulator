import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from scipy.signal import convolve

from .baseband import BasebandData
from .linear_filter import LinearFilter

class ExponentialScatteringModel:
    def __init__(self, scattering_time, bandwidth, obsfreq=0, cutoff=15):
        """
        Create an exponential scattering model.

        Parameters
        ----------
        scattering_time: The scattering time, in seconds.
        bandwidth: Bandwidth of the data to which this model will be applied.
                 This determines the time resolution of the impulse response.
        cutoff: Point at which the impulse response function will be cut off,
                as a multiple of the scattering time.
        """
        self.scattering_time = scattering_time
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq
        self.cutoff = cutoff

    def realize(self, rng=None):
        """
        Create a realization of this scattering model (a ScatteringFilter).

        Parameters
        ----------
        rng: `RandomNumberGenerator` object used to generate the realization.
        """
        if rng is None:
            rng = np.random.default_rng()

        # generate pattern across full bandwidth
        dt = 1/self.bandwidth
        n_samples = np.int64(self.cutoff*self.scattering_time*self.bandwidth)
        time = np.linspace(0, n_samples*dt, n_samples, endpoint=False)
        envelope = np.exp(-time/self.scattering_time)*dt/self.scattering_time
        noise = (rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples))/2
        impulse_response = np.sqrt(envelope)*noise

        return ScatteringFilter(
            impulse_response,
            self.bandwidth,
            self.obsfreq,
        )

class ScatteringFilter(LinearFilter):
    def __init__(self, impulse_response, bandwidth, obsfreq=0):
        """
        Create a scintillation pattern from an impulse response function.

        Parameters
        ----------
        impulse_response: Impulse response function, sampled at the given bandwidth.
        bandwidth: sampling frequency of the provided impulse response data.
        obsfreq: observing frequency of data the filter is to be applied to.
        """
        self.impulse_response = impulse_response
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq
        n_samples = impulse_response.size
        tspan = impulse_response.size/self.bandwidth
        self.time = np.linspace(0, tspan, n_samples, endpoint=False)
        self.filter_function = np.fft.fft(impulse_response, axis=-1)
        self.filter_function = np.fft.fftshift(self.filter_function, axes=-1)
        self.freq = np.fft.fftfreq(n_samples, d=1/self.bandwidth)
        self.freq = np.fft.fftshift(self.freq)
        self.freq += self.obsfreq

    @property
    def n_samples(self):
        return self.impulse_response.size

    @property
    def nlag_neg(self):
        return 0

    @property
    def nlag_pos(self):
        return self.impulse_response.size - 1

    def plot_impulse_response(self, ax=None, **kwargs):
        """
        Plot the impulse response associated with this scintillation pattern.

        Parameters
        ----------
        ax: Axis on which to plot the IRF. With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        time = self.time
        filter_function = np.fft.ifftshift(self.filter_function.flatten())
        impulse_response = np.fft.ifft(filter_function)
        artists = []
        artists.extend(ax.plot(time/1e-6, impulse_response.real, label="Real"))
        artists.extend(ax.plot(time/1e-6, impulse_response.imag, label="Imag"))
        ax.legend()
        ax.set_xlabel("Time (\N{MICRO SIGN}s)")
        ax.set_ylabel("Impulse response")
        return artists

    def plot_filter_function(self, ax=None, **kwargs):
        """
        Plot the filter function associated with this scintillation pattern.

        Parameters
        ----------
        ax:    Axis on which to plot the filter function.
               With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        freq = self.freq.flatten()
        filter_function = self.filter_function.flatten()
        artists = []
        artists.extend(ax.plot(freq/1e6, filter_function.real, label="Real", **kwargs))
        artists.extend(ax.plot(freq/1e6, filter_function.imag, label="Imag", **kwargs))
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered E field")
        return artists

    def plot_scattered_intensity(self, ax=None, **kwargs):
        """
        Plot the scattered intensity associated with this scintillation pattern.

        Parameters
        ----------
        ax:    Axis on which to plot the scattered intensity.
               With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        freq = self.freq.flatten()
        filter_function = self.filter_function.flatten()
        artists = ax.plot(freq/1e6, np.abs(filter_function)**2, **kwargs)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered intensity")
        return artists

    def apply(self, data, force=False):
        """
        Apply this scintillation pattern to baseband data.
        The returned BasebandData object will be shorter by a number of samples
        equal to one less than `self.n_samples`.
        """
        if not force and (data.obsfreq != self.obsfreq or data.bandwidth != self.bandwidth):
            raise ValueError(f"Data observing frequency ({data.obsfreq} Hz) "
                             f"and channel bandwidth ({data.bandwidth} Hz) "
                             "do not match this scintillation pattern")

        # Avoid unnecessary dtype promotion
        irf = self.impulse_response.astype(data.A.dtype)
        nlag_irf = self.impulse_response.size - 1

        if data.delayed:
            A = da.overlap.overlap(data.A, depth={0: (nlag_irf, 0)}, boundary=None)
            B = da.overlap.overlap(data.B, depth={0: (nlag_irf, 0)}, boundary=None)
            chunks = ([chunk - nlag_irf for chunk in A.chunks[0]],)
            A = da.map_blocks(convolve, A, irf, mode='valid', chunks=chunks)
            B = da.map_blocks(convolve, B, irf, mode='valid', chunks=chunks)
            t = data.t[nlag_irf].compute()
        else:
            new_size = data.A.size - nlag_irf
            A = convolve(data.A, irf, mode='valid')
            B = convolve(data.B, irf, mode='valid')
            t = data.t[nlag_irf]
        return BasebandData(
            A, B, t,
            data.feed_poln,
            data.bandwidth,
            data.obsfreq,
        )
