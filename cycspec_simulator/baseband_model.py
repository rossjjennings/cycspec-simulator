import numpy as np
from .interpolation import fft_interp

class BasebandModel:
    def __init__(self, template, bandwidth, pulse_freq, noise_level=0):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        bandwidth: Bandwidth of simulated data (same units as `pulse_freq`).
        pulse_freq: Pulse period (same units as `bandwidth`).
        noise_level: Noise variance in intensity units.
        """
        self.template = template
        self.bandwidth = bandwidth
        self.pulse_freq = pulse_freq
        self.noise_level = noise_level

    def sample(self, n_samples, phase_start=0, endpoint=False, interp=fft_interp,
               rng=np.random.default_rng()):
        samples_per_period = self.bandwidth/self.pulse_freq
        samples_per_bin = samples_per_period/self.template.nbin
        binno_start = phase_start*self.template.nbin
        binno_end = binno_start + n_samples/samples_per_bin
        binno = np.linspace(binno_start, binno_end, n_samples, endpoint=endpoint)
        I = fft_interp(self.template.I, binno)
        noise1 = rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples)
        noise2 = rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples)
        noise3 = rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples)
        if self.template.full_stokes:
            Q = fft_interp(self.template.Q, binno)
            U = fft_interp(self.template.U, binno)
            V = fft_interp(self.template.V, binno)
            X = np.sqrt((I + Q)/2)*noise1 + np.sqrt(self.noise_level)*noise3
            Y = (U + 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
            Y /= np.sqrt(2*(I + Q)) + np.sqrt(self.noise_level)*noise3
        else:
            X = np.sqrt(I/2)*noise1 + np.sqrt(self.noise_level)*noise3
            Y = np.sqrt(I/2)*noise2 + np.sqrt(self.noise_level)*noise3
        return X, Y
