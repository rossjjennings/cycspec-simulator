import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from .baseband import BasebandData

class ExponentialScatteringModel:
    def __init__(self, scattering_time, chan_bw, obsfreq=0, nchan=1, cutoff=15, rng=None):
        """
        Create an exponential scattering model.

        Parameters
        ----------
        scattering_time: The scattering time, in seconds.
        chan_bw: Channel bandwidth of the data to which this model will be applied.
                 This determines the time resolution of the impulse response.
        nchan: Number of channels in the data to which this model will be applied.
        cutoff: Point at which the impulse response function will be cut off,
                as a multiple of the scattering time.
        """
        self.scattering_time = scattering_time
        self.chan_bw = chan_bw
        self.obsfreq = obsfreq
        self.nchan = nchan
        self.cutoff = cutoff
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def realize(self):
        """
        Create a realization of this scattering model (a ScintillationPattern).
        """
        # generate pattern across full bandwidth
        dt = 1/(self.nchan*self.chan_bw)
        n_samples = self.nchan*np.int64(self.cutoff*self.scattering_time*self.chan_bw)
        time = np.linspace(0, n_samples*dt, n_samples, endpoint=False)
        envelope = np.exp(-time/self.scattering_time)*dt/self.scattering_time
        noise = (self.rng.normal(n_samples) + 1j*self.rng.normal(n_samples))/2
        impulse_response = np.sqrt(envelope)*noise

        # split into channels
        filter_function = np.fft.fftshift(np.fft.fft(impulse_response))
        filter_function = filter_function.reshape(self.nchan, -1)
        impulse_response = np.fft.ifft(np.fft.ifftshift(filter_function, axes=-1), axis=-1)

        return ScintillationPattern(
            impulse_response,
            self.chan_bw,
            self.obsfreq,
            self.nchan,
        )

class ScintillationPattern:
    def __init__(self, impulse_response, chan_bw, obsfreq=0, nchan=1):
        """
        Create a scintillation pattern from an impulse response function.

        Parameters
        ----------
        impulse_response: Impulse response function, sampled at the given bandwidth.
        chan_bw: sampling frequency of the provided impulse response data.
        nchan: number of channels in which the impulse response is given.
        """
        self.impulse_response = impulse_response
        self.chan_bw = chan_bw
        self.obsfreq = obsfreq
        self.nchan = nchan
        self.n_samples = impulse_response.size
        tspan = self.n_samples/self.chan_bw
        self.time = np.linspace(0, tspan, self.n_samples, endpoint=False)
        self.filter_function = np.fft.fft(impulse_response, axis=-1)
        self.filter_function = np.fft.fftshift(self.filter_function, axes=-1)
        self.freq = np.fft.fftfreq(self.n_samples, d=1/(self.nchan*self.chan_bw))
        self.freq = np.fft.fftshift(self.freq)
        self.freq += self.obsfreq
        self.freq = self.freq.reshape(self.nchan, -1)

    def plot_impulse_response(self, ichan=None, ax=None, **kwargs):
        """
        Plot the impulse response associated with this scintillation pattern.

        Parameters
        ----------
        ichan: Which channel to plot the IRF for. With `None`, plot the overall IRF.
        ax: Axis on which to plot the IRF. With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if ichan is None:
            time = self.time
            filter_function = np.fft.ifftshift(self.filter_function.flatten())
            impulse_response = np.fft.ifft(filter_function)
        else:
            time = self.time[::self.nchan]
            impulse_response = self.impulse_response[ichan]
        artists = []
        artists.extend(ax.plot(time/1e-6, impulse_response.real, label="Real"))
        artists.extend(ax.plot(time/1e-6, impulse_response.imag, label="Imag"))
        ax.legend()
        ax.set_xlabel("Time (\N{MICRO SIGN}s)")
        if ichan is None:
            ax.set_ylabel("Impulse response")
        else:
            ax.set_ylabel(f"Impulse response (channel {ichan})")
        return artists

    def plot_filter_function(self, ichan=None, ax=None, **kwargs):
        """
        Plot the filter function associated with this scintillation pattern.

        Parameters
        ----------
        ichan: Which channel to plot the filter function for.
               With `None`, plot the filter function across the full band.
        ax:    Axis on which to plot the filter function.
               With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if ichan is None:
            freq = self.freq.flatten()
            filter_function = self.filter_function.flatten()
        else:
            freq = self.freq[ichan]
            filter_function = self.filter_function[ichan]
        artists = []
        artists.extend(ax.plot(freq/1e6, filter_function.real, label="Real", **kwargs))
        artists.extend(ax.plot(freq/1e6, filter_function.imag, label="Imag", **kwargs))
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered E field")
        return artists

    def plot_scattered_intensity(self, ichan=None, ax=None, **kwargs):
        """
        Plot the scattered intensity associated with this scintillation pattern.

        Parameters
        ----------
        ichan: Which channel to plot the scattered intensity for.
               With `None`, plot the scattered intensity across the full band.
        ax:    Axis on which to plot the scattered intensity.
               With `None`, create a new Figure and Axis.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if ichan is None:
            freq = self.freq.flatten()
            filter_function = self.filter_function.flatten()
        else:
            freq = self.freq[ichan]
            filter_function = self.filter_function[ichan]
        artists = ax.plot(freq/1e6, np.abs(filter_function)**2, **kwargs)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Scattered intensity")
        return artists

    def scatter(self, data, force=False):
        """
        Apply this scintillation pattern to baseband data.
        The returned BasebandData object will be shorter by a number of samples
        equal to one less than `self.n_samples`.
        """
        if not force and (data.obsfreq != self.obsfreq or data.chan_bw != self.chan_bw):
            raise ValueError(f"Data observing frequency ({data.obsfreq} Hz) "
                             f"and channel bandwidth ({data.chan_bw} Hz) "
                             "do not match this scintillation pattern")
        new_shape = (data.A.shape[0], data.A.shape[1] - self.impulse_response.shape[-1] + 1)
        A_new = np.empty(new_shape, data.A.dtype)
        B_new = np.empty(new_shape, data.B.dtype)
        for ichan in range(data.nchan):
            A_new[ichan] = convolve(data.A[ichan], self.impulse_response[ichan], mode='valid')
            B_new[ichan] = convolve(data.B[ichan], self.impulse_response[ichan], mode='valid')
        return BasebandData(
            A_new,
            B_new,
            data.t[self.impulse_response.shape[-1] - 1],
            data.feed_poln,
            data.chan_bw,
            data.obsfreq,
        )
