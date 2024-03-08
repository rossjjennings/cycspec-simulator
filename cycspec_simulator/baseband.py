import numpy as np
import numba as nb
import dask.array as da

from .interpolation import fft_interp, lerp
from .time import Time
from .cycspec import PeriodicSpectrum, cycfold_cpu
from .cuda import have_cuda, cuda_failure
if have_cuda:
    from .cycspec_gpu import cycfold_gpu

def complex_white_noise(shape, rng, dtype):
    real = rng.standard_normal(size=shape, dtype=dtype)
    imag = rng.standard_normal(size=shape, dtype=dtype)
    return (real + 1j*imag)/np.sqrt(2)

class BasebandModel:
    def __init__(self, template, predictor, bandwidth, filters=None,
                 obsfreq=0, noise_level=0, feed_poln='LIN', rng=None):
        """
        Create a new model for generating simulated baseband data.

        Parameters
        ----------
        template: TemplateProfile object representing the pulse profile.
        predictor: Pulse phase predictor.
        bandwidth: Bandwidth of simulated data (same units as `pulse_freq`).
        nchan: Number of channels in simulated data.
        obsfreq: Observing frequency (used in plotting and headers only, same
                 units as `bandwidth`).
        noise_level: Noise variance in intensity units.
        feed_poln: Feed polarization ('LIN' for linear or 'CIRC' for circular).
        rng: Random number generator. Expected to be a `np.random.Generator`.
             If `None`, an instance of `np.random.default_rng()` will be created.
             Only the `normal()` method will ever be called.
        """
        self.template = template
        self.predictor = predictor
        self.bandwidth = bandwidth
        if filters is None:
            filters = []
        self.filters = filters
        self.obsfreq = obsfreq
        self.noise_level = noise_level
        self.feed_poln = feed_poln.upper()
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def add_filter(self, filtr):
        """
        Add a filter to be applied to the modeled baseband time series.

        Parameters
        ----------
        filtr: The filter to apply. Should be a LinearFilter object.
        """
        self.filters.append(filtr)

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
        dtype: Numpy dtype to use for samples.
        """
        if t_start is None:
            t_start = self.predictor.epoch
        dtype = np.dtype(dtype)

        for filtr in self.filters:
            n_samples += filtr.n_samples - 1

        delayed = hasattr(da.random, 'Generator') and isinstance(self.rng, da.random.Generator)
        t = get_time_axis(t_start, n_samples, self.bandwidth, delayed=delayed)
        phase = self.predictor.phase(t) - int(self.predictor.phase(t_start))
        binno = (phase*self.template.nbin).astype(dtype)
        I = interp(self.template.I, binno)
        noise1 = complex_white_noise(n_samples, self.rng, dtype)
        noise2 = complex_white_noise(n_samples, self.rng, dtype)
        if self.template.full_stokes:
            Q = interp(self.template.Q, binno)
            U = interp(self.template.U, binno)
            V = interp(self.template.V, binno)
            if self.feed_poln == 'LIN':
                X = np.sqrt((I + Q)/2)*noise1
                Y = (U - 1j*V)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                Y /= np.sqrt(2*(I + Q))
                A, B = X, Y
            elif self.feed_poln == 'CIRC':
                L = np.sqrt((I + V)/2)*noise1
                R = (Q - 1j*U)*noise1 + np.sqrt(I*I - Q*Q - U*U - V*V)*noise2
                R /= np.sqrt(2*(I + V))
                A, B = L, R
            else:
                raise ValueError(f"Invalid polarization type '{self.feed_poln}'.")
        else:
            A = np.sqrt(I/2)*noise1
            B = np.sqrt(I/2)*noise2

        data = BasebandData(A, B, t_start, self.feed_poln, self.bandwidth, self.obsfreq)
        for filtr in self.filters:
            data = filtr.apply(data)

        noise3 = complex_white_noise(data.n_samples, self.rng, dtype)
        data.A += np.sqrt(np.float32(self.noise_level))*noise3
        data.B += np.sqrt(np.float32(self.noise_level))*noise3

        return data

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
    def __init__(self, A, B, start_time, feed_poln, bandwidth, obsfreq):
        if not B.shape == A.shape:
            raise ValueError(f"A and B should be the same shape! Found: {A.shape} != {B.shape}")
        self.A = A
        self.B = B
        self.n_samples = self.A.shape[-1]
        self.delayed = isinstance(A, da.Array)
        self.start_time = start_time
        self.feed_poln = feed_poln.upper()
        self.bandwidth = bandwidth
        self.obsfreq = obsfreq

    @property
    def t(self):
        sample_freq = np.abs(self.bandwidth)
        return get_time_axis(self.start_time, self.n_samples, sample_freq, self.delayed)

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

    def cycfold(self, nchan, nbin, predictor, use_cuda=have_cuda,
                n_threads=nb.config.NUMBA_NUM_THREADS):
        """
        Compute the periodic spectrum from baseband data.

        Parameters
        ----------
        nchan: Number of channels in the periodic spectrum to compute
        nbin: Number of phase bins in which to accumulate
        predictor: `PhasePredictor` object to use in computing phases.
        use_cuda: Whether to use CUDA acceleration. Defaults to True if a CUDA
                  device is detected by CuPy. Otherwise defaults to False.
        n_threads: Number of CPU threads to use. Defaults to the total number of
                   CPUs, as detected by Numba. Has no effect if use_gpu is True.
        """
        if use_cuda and have_cuda:
            return cycfold_gpu(self, nchan, nbin, predictor)
        elif use_cuda:
            err = ValueError("use_cuda was specified, but no CUDA device was found")
            raise err from cuda_failure
        else:
            return cycfold_cpu(self, nchan, nbin, predictor, n_threads=n_threads)
