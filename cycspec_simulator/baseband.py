import numpy as np
import numba as nb
from .interpolation import fft_interp, lerp
from .time import Time

class BasebandModel:
    def __init__(self, template, bandwidth, predictor, obsfreq=0,
                 noise_level=0, feed_poln='LIN', rng=None):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        bandwidth: Bandwidth of simulated data (same units as `pulse_freq`).
        predictor: Pulse phase predictor.
        obsfreq: Observing frequency (used in plotting and headers only, same
                 units
                 as `bandwidth`).
        noise_level: Noise variance in intensity units.
        feed_poln: Feed polarization ('LIN' for linear or 'CIRC' for circular).
        rng: Random number generator. Expected to be a `np.random.Generator`.
             If `None`, an instance of `np.random.default_rng()` will be created.
             Only the `normal()` method will ever be called.
        """
        self.template = template
        self.bandwidth = bandwidth
        self.predictor = predictor
        self.obsfreq = obsfreq
        self.noise_level = noise_level
        self.feed_poln = feed_poln.upper()
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def sample(self, n_samples, t_start=None, interp=lerp):
        """
        Simulate a given number of samples from the modeled baseband time series.

        Parameters
        ----------
        n_samples: The number of samples to use.
        t_start: Time of the first sample, as a Time object. If `None`,
                 the predictor epoch will be used.
        interp: Interpolation function to use. Should take two parameters,
                a template array and an array of sample points at which to
                evaluate the interpolated function (extended periodically).
                `fft_interp` and `lerp` (the default) both work.
        """
        if t_start is None:
            t_start = self.predictor.epoch

        t_span = n_samples/self.bandwidth
        t = Time(
            t_start.mjd,
            t_start.second,
            t_start.offset + np.linspace(0, t_span, n_samples, endpoint=False),
        )
        phase = self.predictor.phase(t)
        binno = phase*self.template.nbin
        I = interp(self.template.I, binno)
        noise1 = (
            (self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples))
            /np.sqrt(2)
        )
        noise2 = (
            (self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples))
            /np.sqrt(2)
        )
        noise3 = (
            (self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples))
            /np.sqrt(2)
        )
        if self.template.full_stokes:
            Q = interp(self.template.Q, binno)
            U = interp(self.template.U, binno)
            V = interp(self.template.V, binno)
            if self.feed_poln == 'LIN':
                X = np.sqrt((I + Q)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                Y = (U - 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                Y /= np.sqrt(2*(I + Q))
                Y += np.sqrt(self.noise_level)*noise3
                return BasebandData(t, X, Y, 'LIN', self.bandwidth, self.obsfreq)
            elif self.feed_poln == 'CIRC':
                L = np.sqrt((I + V)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                R = (Q - 1j*U)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                R /= np.sqrt(2*(I + V))
                R += np.sqrt(self.noise_level)*noise3
                return BasebandData(t, L, R, 'CIRC', self.bandwidth, self.obsfreq)
            else:
                raise ValueError(f"Invalid polarization type '{self.feed_poln}'.")
        else:
            A = np.sqrt(I/2)*noise1 + np.sqrt(self.noise_level)*noise3
            B = np.sqrt(I/2)*noise2 + np.sqrt(self.noise_level)*noise3
            return BasebandData(t, A, B, self.feed_poln, self.bandwidth, self.obsfreq)
        return X, Y

    def sample_time(self, duration, phase_start=0, interp=lerp):
        """
        Simulate a given span of time from the modeled baseband time series.

        Parameters
        ----------
        duration: Time for which to sample.
        phase_start: Phase of the first sample (in cycles).
        interp: Interpolation function to use. Should take two parameters,
                a template array and an array of sample points at which to
                evaluate the interpolated function (extended periodically).
                `fft_interp` and `lerp` (the default) both work.
        """
        n_samples = np.int64(duration*self.bandwidth)
        return sample(n_samples, phase_start, interp)


class BasebandData:
    def __init__(self, t, A, B, feed_poln, bandwidth, obsfreq):
        self.t = t
        self.A = A
        self.B = B
        self.feed_poln = feed_poln.upper()
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq
