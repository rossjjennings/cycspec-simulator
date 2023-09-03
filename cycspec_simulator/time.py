import numpy as np
import numba as nb
import astropy.time

# Days with leap seconds. Current through at least 2024-06-28.
# For updates, see https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
# But note: that file gives the day _after_ the leap second.
leapsec_mjds = np.array([
    41316, # 1971-12-31
    41498, # 1972-06-30
    41682, # 1972-12-31
    42047, # 1973-12-31
    42412, # 1974-12-31
    42777, # 1975-12-31
    43143, # 1976-12-31
    43508, # 1977-12-31
    43873, # 1978-12-31
    44238, # 1979-12-31
    44785, # 1981-06-30
    45150, # 1982-06-30
    45515, # 1983-06-30
    46246, # 1985-06-30
    47160, # 1987-12-31
    47891, # 1989-12-31
    48256, # 1990-12-31
    48803, # 1992-06-30
    49168, # 1993-06-30
    49533, # 1994-06-30
    50082, # 1995-12-31
    50629, # 1997-06-30
    51178, # 1998-12-31
    53735, # 2005-12-31
    54831, # 2008-12-31
    56108, # 2012-06-30
    57203, # 2015-06-30
    57753, # 2016-12-31
])

class Time:
    """
    A one-dimensional array of UTC time values, represented as an epoch,
    specified by an integer MJD and integer second, and a 64-bit
    floating-point offset from that epoch, in seconds.
    Leap seconds are handled correctly for dates from 1972 through the
    next leap second after 2016-12-31T23:59:60. Future dates are handled
    assuming that no additional leap seconds will be inserted or removed.
    """
    def __init__(self, mjd, second, offset):
        """
        Create a Time object.

        Parameters
        ----------
        mjd: Integer MJD.
        second: Integer number of seconds, relative to UTC midnight
                on the specified MJD.
        offset: Offset from the epoch, in seconds. Can represent the fractional
                second part of the time, but can also be greater than 1 second.
                To maintain nanosecond accuracy, should be kept to less than
                2**22 seconds (48.5 days).
        """
        if second > 86400 or (second == 86400 and mjd not in leapsec_mjds):
            raise ValueError(f"Second {second} is beyond end of day {mjd}")
        self.mjd = mjd
        self.second = second
        self.offset = offset

    def __getitem__(self, sl):
        return Time(self.mjd, self.second, self.offset[sl])

    def __sub__(self, other):
        """
        Subtract two Time objects and produce a difference in seconds.
        """
        mjd_diff = self.mjd - other.mjd
        second_diff = 86400*mjd_diff + self.second - other.second
        second_diff += np.sum((leapsec_mjds >= other.mjd) & (leapsec_mjds < self.mjd))
        second_diff += self.offset - other.offset
        return second_diff

    @classmethod
    def from_mjd(cls, mjd, smear_leapsec=False):
        """
        Create a time object from a fractional MJD.
        The fractional part of the MJD is treated as (# of seconds)/86400
        unless the specified day includes a leap second _and_ `smear_leapsec`
        is `True`, in which case it is treated as (# of seconds)/86401.

        Setting `smear_leapsec=False` is consistent with TEMPO, Tempo2, and
        PINT's `PulsarMJD`, while `smear_leapsec=True` is consistent with
        IAU SOFA and is the default behavior of Astropy. For context, see
        https://github.com/astropy/astropy/issues/5369.
        """
        frac, mjd = np.modf(mjd)
        mjd = int(mjd)
        if smear_leapsec and mjd in leapsec_mjds:
            second = 86401*frac
        else:
            second = 86400*frac
        frac, second = np.modf(second)
        second = int(second)
        return cls(mjd, second, frac)
