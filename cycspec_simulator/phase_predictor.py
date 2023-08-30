import numba as nb
from numba.experimental import jitclass

@jitclass([('F0', nb.float64)])
class FreqOnlyPredictor:
    def __init__(self, F0):
        self.F0 = F0

    def phase(self, t):
        return self.F0*t

@jitclass([
    ('span', nb.int64),
    ('site', nb.types.unicode_type),
    ('ref_freq', nb.float64),
    ('ref_mjd', nb.float64),
    ('ref_phase', nb.float64),
    ('ref_f0', nb.float64),
    ('coeffs', nb.float64[:]),
    ('start_phase', nb.float64),
    ('date_produced', nb.types.unicode_type),
    ('version', nb.types.unicode_type),
    ('log10_fit_err', nb.float64),
])
class PolynomialPredictor:
    def __init__(self, span, site, ref_freq, ref_mjd, ref_phase, ref_f0, coeffs,
                 start_phase=0., date_produced='', version='', log10_fit_err=0.):
        self.date_produced = date_produced
        self.version = version
        self.span = span
        self.site = site
        self.ref_freq = ref_freq
        self.start_phase = start_phase
        self.ref_mjd = ref_mjd
        self.ref_phase = ref_phase
        self.ref_f0 = ref_f0
        self.log10_fit_err = log10_fit_err
        self.coeffs = coeffs

    @classmethod
    def from_record(cls, rec):
        return cls(
            span = rec['NSPAN'],
            site = rec['NSITE'],
            ref_freq = rec['REF_FREQ'],
            ref_mjd = rec['REF_MJD'],
            ref_phase = rec['REF_PHS'],
            ref_f0 = rec['REF_F0'],
            coeffs = rec['COEFF'],
            start_phase = rec['PRED_PHS'],
            date_produced = rec['DATE_PRO'],
            version = rec['POLYVER'],
            log10_fit_err = rec['LGFITERR'],
        )

    @classmethod
    def parse(cls, lines):
        i = 0
        polycos = []
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
            polycos.append(cls(
                span=int(span),
                site=site,
                ref_freq=float(ref_freq),
                ref_mjd=float(ref_mjd),
                ref_phase=float(ref_phase),
                ref_f0=float(ref_f0),
                coeffs=np.array(coeffs),
                log10_fit_err=float(log10_fit_err),
            ))
        return polycos

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

    def dt(self, t, check_bounds=True):
        mjd_start = self.ref_mjd - self.span/1440
        mjd_end = self.ref_mjd + self.span/1440
        if check_bounds and np.any((t < mjd_start) | (t > mjd_end)):
            raise ValueError(f'MJD out of bounds.')

        dt = (t - self.ref_mjd)*1440 # minutes
        return dt

