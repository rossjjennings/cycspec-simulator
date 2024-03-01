import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from .baseband import BasebandData
from .linear_filter import LinearFilter

class ExponentialScatteringModel:
    def __init__(self, scattering_time, bandwidth, obsfreq=0, cutoff=15, rng=None):
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
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def realize(self):
        """
        Create a realization of this scattering model (a ScatteringFilter).
        """
        # generate pattern across full bandwidth
        dt = 1/self.bandwidth
        n_samples = np.int64(self.cutoff*self.scattering_time*self.bandwidth)
        time = np.linspace(0, n_samples*dt, n_samples, endpoint=False)
        envelope = np.exp(-time/self.scattering_time)*dt/self.scattering_time
        noise = (self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples))/2
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
        """
        self.impulse_response = impulse_response
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq
        self.n_samples = impulse_response.size
        tspan = self.n_samples/self.bandwidth
        self.time = np.linspace(0, tspan, self.n_samples, endpoint=False)
        self.filter_function = np.fft.fft(impulse_response, axis=-1)
        self.filter_function = np.fft.fftshift(self.filter_function, axes=-1)
        self.freq = np.fft.fftfreq(self.n_samples, d=1/self.bandwidth)
        self.freq = np.fft.fftshift(self.freq)
        self.freq += self.obsfreq

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
        new_size = data.A.size - self.impulse_response.size + 1
        A_new = np.empty(new_size, data.A.dtype)
        B_new = np.empty(new_size, data.B.dtype)
        A_new = convolve(data.A, self.impulse_response, mode='valid')
        B_new = convolve(data.B, self.impulse_response, mode='valid')
        return BasebandData(
            A_new,
            B_new,
            data.t[self.impulse_response.size - 1],
            data.feed_poln,
            data.bandwidth,
            data.obsfreq,
        )
