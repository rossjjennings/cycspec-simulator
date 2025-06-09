from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.polynomial import polynomial
import numba as nb
import dask.array as da

from .gpu import have_cuda
if have_cuda:
    import cupy as cp

from .time import Time

class PhasePredictor(metaclass=ABCMeta):
    """
    An abstract base class for phase predictors. A `PhasePredictor` instance
    is an object that can be used to predict the phase of a pulsar at a specific time.
    """
    @abstractmethod
    def phase(self, t):
        """
        Given a Time object `t` (possibly containing an array of times),
        return the phase of the pulsar described by this PhasePredictor at time `t`.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated.
        """
        pass

class FreqOnlyPredictor(PhasePredictor):
    """
    A phase predictor which assumes a constant pulse frequency.
    """
    def __init__(self, f0, epoch):
        """
        Create a phase predictor which assumes a constant pulse frequency.

        Parameters
        ----------
        f0: The pulse frequency to be used
        epoch: A Time object representing the time at which the phase is zero.
        """
        self.f0 = f0
        self.epoch = epoch

    def phase(self, t):
        """
        Return the phase of the pulsar at time `t`.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated (possibly an array).
        """
        if t.device:
            epoch = self.epoch.to_device()
        else:
            epoch = self.epoch
        return self.f0*(t - epoch)

class PolynomialPredictor(PhasePredictor):
    """
    A phase predictor based on a set of polynomial coefficients (i.e., "polyco")
    of the type produced by TEMPO.
    """
    def __init__(self, segments):
        """
        Create a phase predictor based on multiple segments, each described by
        a Taylor polynomial.

        Parameters
        ----------
        segments: List of `PolynomialSegment` objects representing the segments.
        """
        self.segments = segments
        self.epoch = segments[0].epoch

    @classmethod
    def parse(cls, lines):
        """
        Parse lines from a TEMPO `polyco.dat` file, potentially containing
        multiple segments.

        Parameters
        ----------
        lines: List of lines from the file.
        """
        i = 0
        segments = []
        while i < len(lines):
            if len(lines[i]) == 0:
                i += 1
                continue
            psr, date, utc, ref_mjd, dm, doppler, log10_fit_err = lines[i].split()
            i += 1
            ref_phase, ref_f0, site, span, ncoeff, ref_freq, *binary_phase = lines[i].split()
            ncoeff = int(ncoeff)
            i += 1
            j = 0
            coeffs = []
            while 3*(j + 1) <= ncoeff:
                coeffs.extend([float(part.replace('D', 'E')) for part in lines[i + j].split()])
                j += 1
            i += j
            segments.append(PolynomialSegment(
                span=int(span),
                site=site,
                epoch=Time.from_mjd(float(ref_mjd)),
                ref_freq=float(ref_freq),
                ref_phase=float(ref_phase),
                ref_f0=float(ref_f0),
                coeffs=np.array(coeffs),
                log10_fit_err=float(log10_fit_err),
            ))
        return cls(segments)

    @classmethod
    def from_file(cls, filename):
        """
        Create a phase predictor based on a TEMPO `polyco.dat` file.

        Parameters
        ----------
        filename: Path to the polyco file.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        return cls.parse(lines)

    def closest_segment(self, t):
        """
        Find the segment whose center is closest to the time `t`.
        Broadcasts over arrays.
        """
        diffs = []
        for segment in self.segments:
            if t.device:
                segment_epoch = segment.epoch.to_device()
            else:
                segment_epoch = segment.epoch
            diffs.append(t - segment_epoch)
        if t.delayed:
            xp = da
        elif t.device:
            xp = cp
        else:
            xp = np
        diffs = xp.stack(diffs, axis=0)
        closest_segment = xp.argmin(xp.abs(diffs), axis=0)
        return closest_segment

    def covers(self, t):
        """
        Return a boolean value (or array) indicating whether this phase predictor
        includes a segment covering the time `t`. Broadcasts over arrays.
        """
        if t.delayed:
            xp = da
        elif t.device:
            xp = cp
        else:
            xp = np
        return xp.any([segment.covers(t) for segment in self.segments], axis=0)

    def phase(self, t, check_bounds=True, reduce_refphase=True):
        """
        Return the phase of the pulsar at time `t`, as predicted by the
        segment whose center is closest to `t`.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated.
        check_bounds: Whether to raise an error if any times are out of bounds,
                      or just extrapolate based on the closest segment.
        reduce_refphase: Whether to reduce the reference phase modulo 1 before
                         computing the phase. This will bring the phase closer to
                         zero by a whole number of turns, increasing the precision
                         that can be retained in the fractional part.
        """
        if t.delayed:
            xp = da
        elif t.device:
            xp = cp
        else:
            xp = np
        closest_segment = self.closest_segment(t)
        phase = xp.empty_like(t.offset)
        for i, segment in enumerate(self.segments):
            sl = (closest_segment == i)
            phase[sl] = segment.phase(t[sl], check_bounds, reduce_refphase)

        return phase[()] # turns 0d arrays into scalars, otherwise harmless

class PolynomialSegment:
    """
    An object representing a segment of a piecewise polynomial model for phase
    as a function of time. Contains polynomial coefficients and various metadata.
    """
    def __init__(self, span, site, epoch, ref_freq, ref_phase, ref_f0, coeffs,
                 start_phase=0., date_produced='', version='', log10_fit_err=0.):
        """
        Create a phase predictor based on a single segment in which the phase
        can be described by a Taylor polynomial as a function of time.

        Parameters
        ----------
        span: Time span of the segment (minutes)
        site: Observatory code indicating the location where the function is valid
        epoch: Time at which the phase is zero
        ref_freq: Reference radio frequency at which the phase is evaluated
        ref_f0: Reference pulse frequency at the epoch
        coeffs: Coefficients of powers of (t-epoch) in the Taylor polynomial.
                `coeffs[k]` the coefficient of (t-epoch)**k.
        start_phase: Phase at the epoch

        Additional metadata (stored but not used)
        -----------------------------------------
        date_produced: Date when the approximate phase function was produced
        version: Version of Tempo used to produce the approximation
        log10_fit_err: Base-10 logarithm of the approximation error
        """
        self.date_produced = date_produced
        self.version = version
        self.span = span
        self.site = site
        self.epoch = epoch
        self.ref_freq = ref_freq
        self.start_phase = start_phase
        self.ref_phase = ref_phase
        self.ref_f0 = ref_f0
        self.log10_fit_err = log10_fit_err
        self.coeffs = coeffs

    @classmethod
    def from_record(cls, rec):
        """
        Create a `PolynomialSegment` from a record in a FITS HDU, such as might
        be found in the 'POLYCO' HDU of a PSRFITS file.
        """
        return cls(
            span = rec['NSPAN'],
            site = rec['NSITE'],
            epoch = Time.from_mjd(rec['REF_MJD']),
            ref_freq = rec['REF_FREQ'],
            ref_phase = rec['REF_PHS'],
            ref_f0 = rec['REF_F0'],
            coeffs = rec['COEFF'],
            start_phase = rec['PRED_PHS'],
            date_produced = rec['DATE_PRO'],
            version = rec['POLYVER'],
            log10_fit_err = rec['LGFITERR'],
        )

    def phase(self, t, check_bounds=True, reduce_refphase=True):
        """
        Calculate the phase at a particular time.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated.
        check_bounds: Whether to raise an error if any times are out of bounds,
                      or extrapolate beyond the bounds of the segment.
        reduce_refphase: Whether to reduce the reference phase modulo 1 before
                         computing the phase. This will bring the phase closer to
                         zero by a whole number of turns, increasing the precision
                         that can be retained in the fractional part.
        """
        if t.device:
            polyval = cp.polynomial.polynomial.polyval
        else:
            polyval = np.polynomial.polynomial.polyval

        dt = self.dt(t, check_bounds)
        ref_phase = (self.ref_phase % 1) if reduce_refphase else self.ref_phase

        if t.delayed:
            dphase = da.map_blocks(polyval, dt, self.coeffs, meta=dt._meta)
        else:
            dphase = polyval(dt, self.coeffs)

        phase = ref_phase + dt*60*self.ref_f0 + dphase
        return phase

    def dphase(self, t, check_bounds=True, ref_time=None):
        """
        Return the difference between the phase of the pulsar at time `t`
        and the phase at a reference time.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated.
        check_bounds: If True, raise an exception if any of the times
            represented by `t` are outside the bounds of this segment.
        ref_time: Reference time. If None, the model epoch will be used.
        """
        if t.device:
            polyval = cp.polynomial.polynomial.polyval
            if ref_time is not None:
                ref_time = ref_time.to_device()
        else:
            polyval = np.polynomial.polynomial.polyval

        dt = self.dt(t, check_bounds)

        if t.delayed:
            dphase = da.map_blocks(polyval, dt, self.coeffs, meta=dt._meta)
        else:
            dphase = polyval(dt, self.coeffs)

        if ref_time is None:
            phase = dt*60*self.ref_f0 + dphase
            phase -= self.coeffs[0] # equivalent to polynomial.polyval(0, self.coeffs)
        else:
            ref_dt = self.dt(ref_time)
            phase = (dt-ref_dt)*60*self.ref_f0 + dphase
            phase -= polynomial.polyval(ref_dt, self.coeffs)
        return phase

    def f0(self, t, check_bounds=True):
        """
        Return the instantaneous topocentric frequency of the pulsar at time `t`.

        Parameters
        ----------
        t: Time at which the phase is to be evaluated.
        check_bounds: If True, raise an exception if any of the times
            represented by `t` are outside the bounds of this segment.
        """
        if t.device:
            polyval = cp.polynomial.polynomial.polyval
        else:
            polyval = np.polynomial.polynomial.polyval

        dt = self.dt(t, check_bounds)
        der_coeffs = polynomial.polyder(self.coeffs)

        if t.delayed:
            df0 = da.map_blocks(polyval, dt, der_coeffs, meta=dt._meta)/60
        else:
            df0 = polyval(dt, der_coeffs)/60

        f0 = self.ref_f0 + df0
        return f0

    def covers(self, t):
        """
        Return a boolean value (or array) indicating whether this segment
        covers the time `t`. Broadcasts over arrays.
        """
        if t.delayed:
            xp = da
        elif t.device:
            xp = cp
        else:
            xp = np

        if t.device:
            epoch = self.epoch.to_device()
        else:
            epoch = self.epoch

        dt = (t - self.epoch)/60 # minutes
        return xp.abs(dt) <= self.span/2

    def dt(self, t, check_bounds=False):
        """
        Calculate the time difference, in minutes, between `t` and the model epoch,
        or raise an error if `t` is outside the bounds of this segment.

        Parameters
        ----------
        t: Specified time (possibly an array).
        check_bounds: Whether to raise an error if any times are out of bounds.
                      Warning: this check can take a significant amount of time
                      if `t` is a large array.
        """
        if t.delayed:
            xp = da
        elif t.device:
            xp = cp
        else:
            xp = np

        if t.device:
            epoch = self.epoch.to_device()
        else:
            epoch = self.epoch
        dt = (t - self.epoch)/60 # minutes

        if check_bounds:
            not_covered = ~self.covers(t)
            if xp.any(not_covered):
                i = xp.where(not_covered)[0][0]
                raise ValueError(f'Time at position {i} out of bounds.')
        return dt
