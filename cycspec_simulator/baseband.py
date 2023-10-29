import numpy as np
import numba as nb
import dask.array as da

from .interpolation import fft_interp, lerp
from .time import Time

def complex_white_noise(shape, rng, dtype):
    real = rng.standard_normal(size=shape, dtype=dtype)
    imag = rng.standard_normal(size=shape, dtype=dtype)
    return (real + 1j*imag)/np.sqrt(2)

class BasebandModel:
    def __init__(self, template, predictor, chan_bw, nchan=1, obsfreq=0,
                 noise_level=0, feed_poln='LIN', rng=None):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        predictor: Pulse phase predictor.
        chan_bw: Channel bandwidth of simulated data (same units as `pulse_freq`).
        nchan: Number of channels in simulated data.
        obsfreq: Observing frequency (used in plotting and headers only, same
                 units as `chan_bw`).
        noise_level: Noise variance in intensity units.
        feed_poln: Feed polarization ('LIN' for linear or 'CIRC' for circular).
        rng: Random number generator. Expected to be a `np.random.Generator`.
             If `None`, an instance of `np.random.default_rng()` will be created.
             Only the `normal()` method will ever be called.
        """
        self.template = template
        self.predictor = predictor
        self.chan_bw = chan_bw
        self.nchan = nchan
        self.obsfreq = obsfreq
        self.noise_level = noise_level
        self.feed_poln = feed_poln.upper()
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def sample(self, n_samples, t_start=None, interp=lerp, dtype=np.float32):
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
        dtype = np.dtype(dtype)

        delayed = isinstance(self.rng, da.random.Generator)
        t = get_time_axis(t_start, n_samples, self.chan_bw, delayed=delayed)
        phase = self.predictor.phase(t) - int(self.predictor.phase(t_start))
        binno = (phase*self.template.nbin).astype(dtype)
        I = interp(self.template.I, binno)
        shape = (self.nchan, n_samples)
        noise1 = complex_white_noise(shape, self.rng, dtype)
        noise2 = complex_white_noise(shape, self.rng, dtype)
        noise3 = complex_white_noise(shape, self.rng, dtype)
        if self.template.full_stokes:
            Q = interp(self.template.Q, binno)
            U = interp(self.template.U, binno)
            V = interp(self.template.V, binno)
            if self.feed_poln == 'LIN':
                X = np.sqrt((I + Q)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                Y = (U - 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                Y /= np.sqrt(2*(I + Q))
                Y += np.sqrt(self.noise_level)*noise3
                return BasebandData(X, Y, t_start, 'LIN', self.chan_bw, self.obsfreq)
            elif self.feed_poln == 'CIRC':
                L = np.sqrt((I + V)/2)*noise1 + np.sqrt(self.noise_level)*noise3
                R = (Q - 1j*U)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                R /= np.sqrt(2*(I + V))
                R += np.sqrt(self.noise_level)*noise3
                return BasebandData(L, R, t_start, 'CIRC', self.chan_bw, self.obsfreq)
            else:
                raise ValueError(f"Invalid polarization type '{self.feed_poln}'.")
        else:
            A = np.sqrt(I/2)*noise1 + np.sqrt(self.noise_level)*noise3
            B = np.sqrt(I/2)*noise2 + np.sqrt(self.noise_level)*noise3
            return BasebandData(A, B, t_start, self.feed_poln, self.chan_bw, self.obsfreq)
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
        n_samples = np.int64(duration*self.chan_bw)
        return sample(n_samples, phase_start, interp)

def get_time_axis(start_time, n_samples, bandwidth, delayed=False):
    if delayed:
        linspace = da.linspace
    else:
        linspace = np.linspace
    t_span = n_samples/bandwidth
    return Time(
        start_time.mjd,
        start_time.second,
        start_time.offset + linspace(0, t_span, n_samples, endpoint=False),
    )

class BasebandData:
    def __init__(self, A, B, start_time, feed_poln, chan_bw, obsfreq):
        if not B.shape == A.shape:
            raise ValueError(f"A and B should be the same shape! Found: {A.shape} != {B.shape}")
        self.A = A
        self.B = B
        self.n_samples = self.A.shape[-1]
        self.delayed = isinstance(A, da.Array)
        self.start_time = start_time
        self.feed_poln = feed_poln.upper()
        self.chan_bw = chan_bw
        self.obsfreq = obsfreq

    @property
    def t(self):
        return get_time_axis(self.start_time, self.n_samples, self.chan_bw, self.delayed)

    @property
    def nchan(self):
        return self.A.shape[0]

    def compute_all(self):
        if self.delayed:
            return BasebandData(
                self.A.compute(),
                self.B.compute(),
                self.start_time,
                self.feed_poln,
                self.chan_bw,
                self.obsfreq,
            )
        else:
            return self
