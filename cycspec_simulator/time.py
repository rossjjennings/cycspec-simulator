import numpy as np
import numba as nb
import astropy.time
from numba.experimental import jitclass

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

@jitclass([
    ('mjd', nb.int32),
    ('second', nb.int32),
    ('offset', nb.float64),
])
class Time:
    """
    A one-dimensional array of time values, represented as an epoch,
    specified by an integer MJD and integer second, and a 64-bit
    floating-point offset from that epoch, in seconds.
    Necessary because Numba doesn't play well with Astropy Time objects.
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

    def mid(self, other):
        offset = (self.offset + other.offset)/2
        return Time(self.mjd, self.second, offset)

    def diff(self, other):
        mjd_diff = self.mjd - other.mjd
        second_diff = 86400*mjd_diff + self.second - other.second
        #second_diff += np.sum((leapsec_mjds >= other.mjd) & (leapsec_mjds < self.mjd))
        second_diff += self.offset - other.offset
        return second_diff

@jitclass([
    ('mjd', nb.int32),
    ('second', nb.int32),
    ('offset', nb.float64[:]),
])
class TimeSequence:
    """
    A one-dimensional array of time values, represented as an epoch,
    specified by an integer MJD and integer second, and an array of
    64-bit floating-point offsets from that epoch, in seconds.
    Necessary because Numba doesn't play well with Astropy Time objects.
    The range of offset values should be kept small to maintain precision:
    nanosecond accuracy will be lost for offsets greater than about 48 days.
    However, values within 24 hours of the epoch will be stored to a
    precision better than 2e-11 seconds.
    """
    def __init__(self, mjd, second, offset):
        """
        Create a TimeSequence object.

        Parameters
        ----------
        mjd: Integer MJD of the epoch.
        second: Integer number of seconds of the epoch, relative to
                UTC midnight on the specified MJD.
        offset: Array of offsets from the epoch. To maintain nanosecond
                accuracy, should be kept to less than 2**22 seconds
                (48.5 days).
        """
        self.mjd = mjd
        self.second = second
        self.offset = offset

    def __getitem__(self, sl):
        return TimeSequence(self.mjd, self.second, self.offset[sl])

    def at(self, idx):
        return Time(self.mjd, self.second, self.offset[idx])

    def diff(self, other):
        mjd_diff = self.mjd - other.mjd
        second_diff = 86400*mjd_diff + self.second - other.second
        #second_diff += np.sum((leapsec_mjds >= other.mjd) & (leapsec_mjds < self.mjd))
        output_diff = np.empty_like(self.offset)
        for i in range(self.offset.size):
            output_diff[i] = second_diff + self.offset[i] - other.offset
        return output_diff

time_type = Time.class_type.instance_type
time_sequence_type = TimeSequence.class_type.instance_type
