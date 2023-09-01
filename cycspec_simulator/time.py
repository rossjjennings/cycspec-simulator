import numpy as np
import numba as nb
import astropy.time
from numba.experimental import jitclass

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
