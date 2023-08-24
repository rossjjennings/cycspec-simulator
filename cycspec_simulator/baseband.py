import numpy as np
import numba as nb
from numba.experimental import jitclass
from .interpolation import fft_interp, lerp
from .baseband_data import BasebandData

class BasebandModel:
    def __init__(self, template, bandwidth, pulse_freq,
                 noise_level=0, feed_poln='LIN', rng=None):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        bandwidth: Bandwidth of simulated data (same units as `pulse_freq`).
        pulse_freq: Pulse period (same units as `bandwidth`).
        noise_level: Noise variance in intensity units.
        feed_poln: Feed polarization ('LIN' for linear or 'CIRC' for circular).
        rng: Random number generator. Expected to be a `np.random.Generator`.
             If `None`, an instance of `np.random.default_rng()` will be created.
             Only the `normal()` method will ever be called.
        """
        self.template = template
        self.bandwidth = bandwidth
        self.pulse_freq = pulse_freq
        self.noise_level = noise_level
        self.feed_poln = feed_poln.upper()
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def sample(self, n_samples, phase_start=0, interp=lerp):
        """
        Simulate a given number of samples from the modeled baseband time series.

        Parameters
        ----------
        n_samples: The number of samples to use.
        phase_start: Phase of the first sample (in cycles).
        interp: Interpolation function to use. Should take two parameters,
                a template array and an array of sample points at which to
                evaluate the interpolated function (extended periodically).
                `fft_interp` and `lerp` (the default) both work.
        """
        samples_per_period = self.bandwidth/self.pulse_freq
        samples_per_bin = samples_per_period/self.template.nbin
        binno_start = phase_start*self.template.nbin
        binno_end = binno_start + n_samples/samples_per_bin
        binno = np.linspace(binno_start, binno_end, n_samples, endpoint=False)
        I = interp(self.template.I, binno)
        t = binno/self.template.nbin/self.pulse_freq
        noise1 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        noise2 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        noise3 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        if self.template.full_stokes:
            Q = interp(self.template.Q, binno)
            U = interp(self.template.U, binno)
            V = interp(self.template.V, binno)
            if self.feed_poln == 'LIN':
                X = np.sqrt((I + Q)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                Y = (U + 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                Y /= np.sqrt(2*(I + Q))
                Y += np.sqrt(self.noise_level)*noise3
                return BasebandData(t, X, Y, 'LIN')
            elif self.feed_poln == 'CIRC':
                L = np.sqrt((I + V)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                R = (Q - 1j*U)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                R /= np.sqrt(2*(I + V))
                R += np.sqrt(self.noise_level)*noise3
                return BasebandData(t, L, R, 'CIRC')
            else:
                raise ValueError(f"Invalid polarization type '{self.feed_poln}'.")
        else:
            A = np.sqrt(I/2)*noise1 + np.sqrt(self.noise_level)*noise3
            B = np.sqrt(I/2)*noise2 + np.sqrt(self.noise_level)*noise3
            return BasebandData(t, A, B, self.feed_poln)
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

@jitclass([
    ('t', nb.float64[:]),
    ('X', nb.complex128[:]),
    ('Y', nb.complex128[:]),
    ('L', nb.complex128[:]),
    ('R', nb.complex128[:]),
])
class BasebandData:
    feed_poln: str

    def __init__(self, t, A, B, feed_poln):
        self.t = t
        self.feed_poln = feed_poln
        if feed_poln.upper() == 'LIN':
            self.X = A
            self.Y = B
        elif feed_poln.upper() == 'CIRC':
            self.L = A
            self.R = B
        else:
            raise ValueError(f"Invalid polarization type '{feed_poln}'.")
