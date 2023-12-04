import numpy as np
from numpy.polynomial import polynomial
import numba as nb

from .time import Time

class FreqOnlyPredictor:
    def __init__(self, f0, epoch):
        self.f0 = f0
        self.epoch = epoch

    def phase(self, t):
        return self.f0*(t - self.epoch)

class PolynomialPredictor:
    def __init__(self, segments):
        self.segments = segments
        self.epoch = segments[0].epoch

    @classmethod
    def parse(cls, lines):
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
        with open(filename, 'r') as f:
            lines = f.readlines()
        return cls.parse(lines)

    def closest_segment(self, t):
        diffs = np.array([t - segment.epoch for segment in self.segments])
        closest_segment = np.argmin(np.abs(diffs), axis=0)
        return closest_segment

    def covers(self, t):
        return np.any([segment.covers(t) for segment in self.segments], axis=0)

    def phase(self, t, check_bounds=True):
        closest_segment = self.closest_segment(t)
        phase = np.empty_like(t.offset)
        for i, segment in enumerate(self.segments):
            sl = (closest_segment == i)
            phase[sl] = segment.phase(t[sl], check_bounds)

        return phase[()] # turns 0d arrays into scalars, otherwise harmless

class PolynomialSegment:
    def __init__(self, span, site, epoch, ref_freq, ref_phase, ref_f0, coeffs,
                 start_phase=0., date_produced='', version='', log10_fit_err=0.):
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

    def phase(self, t, check_bounds=True):
        dt = self.dt(t, check_bounds)
        phase = self.ref_phase + dt*60*self.ref_f0 + polynomial.polyval(dt, self.coeffs)
        return phase

    def dphase(self, t, check_bounds=True, ref_time=None):
        dt = self.dt(t, check_bounds)
        if ref_time is None:
            phase = dt*60*self.ref_f0 + polynomial.polyval(dt, self.coeffs)
            phase -= self.coeffs[0] # equivalent to polynomial.polyval(0, self.coeffs)
        else:
            ref_dt = self.dt(ref_time)
            phase = (dt-ref_dt)*60*self.ref_f0 + polynomial.polyval(dt, self.coeffs)
            phase -= polynomial.polyval(ref_dt, self.coeffs)
        return phase

    def f0(self, t, check_bounds=True):
        dt = self.dt(t, check_bounds)

        der_coeffs = polynomial.polyder(self.coeffs)
        f0 = self.ref_f0 + polynomial.polyval(dt, der_coeffs)/60
        return f0

    def covers(self, t):
        dt = (t - self.epoch)/60 # minutes
        return np.abs(dt) <= self.span/2

    def dt(self, t, check_bounds=True):
        dt = (t - self.epoch)/60 # minutes
        if check_bounds:
            not_covered = ~self.covers(t)
            if np.any(not_covered):
                i = np.where(not_covered)[0][0]
                raise ValueError(f'Time at position {i} out of bounds.')
        return dt
