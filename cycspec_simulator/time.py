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
    A one-dimensional array of time values, represented as an epoch,
    specified by an integer MJD and integer second, and a 64-bit
    floating-point offset from that epoch, in seconds.
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
        self.mjd = mjd
        self.second = second
        self.offset = offset

    def __getitem__(self, sl):
        return Time(self.mjd, self.second, self.offset[sl])

    def __sub__(self, other):
        mjd_diff = self.mjd - other.mjd
        second_diff = 86400*mjd_diff + self.second - other.second
        second_diff += np.sum((leapsec_mjds >= other.mjd) & (leapsec_mjds < self.mjd))
        second_diff += self.offset - other.offset
        return second_diff
