import numpy as np
from .interpolation import fft_interp, lerp

class BasebandModel:
    def __init__(self, template, bandwidth, pulse_freq, noise_level=0, rng=None):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        bandwidth: Bandwidth of simulated data (same units as `pulse_freq`).
        pulse_freq: Pulse period (same units as `bandwidth`).
        noise_level: Noise variance in intensity units.
        rng: Random number generator. Expected to be a `np.random.Generator`.
             If `None`, an instance of `np.random.default_rng()` will be created.
             Only the `normal()` method will ever be called.
        """
        self.template = template
        self.bandwidth = bandwidth
        self.pulse_freq = pulse_freq
        self.noise_level = noise_level
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def sample(self, n_samples, phase_start=0, interp=lerp):
        """
        Simulate a given number of samples from the specified baseband time series.

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
        noise1 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        noise2 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        noise3 = self.rng.normal(size=n_samples) + 1j*self.rng.normal(size=n_samples)
        if self.template.full_stokes:
            Q = interp(self.template.Q, binno)
            U = interp(self.template.U, binno)
            V = interp(self.template.V, binno)
            X = np.sqrt((I + Q)/2)*noise1 + np.sqrt(self.noise_level)*noise3
            Y = (U + 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
            Y /= np.sqrt(2*(I + Q)) + np.sqrt(self.noise_level)*noise3
        else:
            X = np.sqrt(I/2)*noise1 + np.sqrt(self.noise_level)*noise3
            Y = np.sqrt(I/2)*noise2 + np.sqrt(self.noise_level)*noise3
        return X, Y

    def sample_time(self, duration, phase_start=0, interp=lerp):
        """
        Simulate a given number of samples from the specified baseband time series.

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
