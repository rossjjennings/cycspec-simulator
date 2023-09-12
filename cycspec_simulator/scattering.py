import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from .baseband import BasebandData

class ExponentialScatteringModel:
    def __init__(self, scattering_time, bandwidth, cutoff=15):
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
        self.cutoff = cutoff

    def realize(self):
        """
        Create a realization of this scattering model (a ScintillationPattern).
        """
        dt = 1/self.bandwidth
        n_samples = np.int64(self.cutoff*self.scattering_time/dt)
        time = np.linspace(0, n_samples*dt, n_samples, endpoint=False)
        envelope = np.exp(-time/self.scattering_time)*dt/self.scattering_time
        noise = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))/2
        impulse_response = np.sqrt(envelope)*noise
        return ScintillationPattern(self.bandwidth, impulse_response)

class ScintillationPattern:
    def __init__(self, bandwidth, impulse_response):
        """
        Create a scintillation pattern from an impulse response function.

        Parameters
        ----------
        bandwidth: i.e., sampling frequency of the provided impulse response data.
        impulse_response: Impulse response function, sampled at the given bandwidth.
        """
        self.bandwidth = bandwidth
        dt = 1/bandwidth
        self.impulse_response = impulse_response
        self.n_samples = impulse_response.size
        self.time = np.linspace(0, self.n_samples*dt, self.n_samples, endpoint=False)
        self.filter_function = np.fft.fftshift(np.fft.fft(impulse_response))
        self.freq = np.fft.fftshift(np.fft.fftfreq(self.n_samples, d=dt))

    def plot_impulse_response(self, ax=None, **kwargs):
        """
        Plot the impulse response associated with this scintillation pattern.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        artists = []
        artists.extend(ax.plot(self.time/1e-6, self.impulse_response.real, label="Real"))
        artists.extend(ax.plot(self.time/1e-6, self.impulse_response.imag, label="Imag"))
        ax.legend()
        ax.set_xlabel("Time (\N{MICRO SIGN}s)")
        ax.set_ylabel("Impulse response")
        return artists

    def plot_filter_function(self, ax=None, **kwargs):
        """
        Plot the filter function associated with this scintillation pattern.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        artists = []
        artists.extend(ax.plot(self.freq/1e6, self.filter_function.real, label="Real"))
        artists.extend(ax.plot(self.freq/1e6, self.filter_function.imag, label="Imag"))
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered E field")
        return artists

    def plot_scattered_intensity(self, ax=None, **kwargs):
        """
        Plot the filter function associated with this scintillation pattern.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        artists = ax.plot(self.freq/1e6, np.abs(self.filter_function)**2)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered intensity")
        return artists

    def scatter(self, data):
        """
        Apply this scintillation pattern to baseband data.
        The returned BasebandData object will be shorter by a number of samples
        equal to one less than `self.n_samples`.
        """
        return BasebandData(
            convolve(data.A, self.impulse_response, mode='valid'),
            convolve(data.B, self.impulse_response, mode='valid'),
            data.t[self.n_samples - 1],
            data.feed_poln,
            data.bandwidth,
            data.obsfreq,
        )
