import numpy as np
import numba as nb
from astropy.utils.iers import LeapSeconds

# Get the list of leap seconds from Astropy.
# It is updated automatically from the official IERS table at
#   https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat.
# That file gives the day _after_ the leap second, so we subtract 1 here.
leapsec_mjds = LeapSeconds.auto_open()['mjd'].astype('int64') - 1

class Time:
    """
    A one-dimensional array of UTC time values, represented as an epoch,
    specified by an integer MJD and integer second, and a 64-bit
    floating-point offset from that epoch, in seconds.
    Leap seconds are handled correctly as long as the Astropy leap second
    table is up to date. For recently-inserted leap seconds, this may require
    an internet connection. (But, as of December 2023, no new leap seconds
    have been inserted since 2016-12-31T23:59:60.)
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
        self.offset = np.asanyarray(offset)

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
        Create a Time object from a fractional MJD.
        The fractional part of the MJD is treated as (# of seconds)/86400
        unless the specified day includes a leap second _and_ `smear_leapsec`
        is `True`, in which case it is treated as (# of seconds)/86401.
        With `smear_leapsec=False`, there are no inputs to this function
        that produce times during a leap second.

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

    def to_mjd(self, smear_leapsec=False):
        """
        Convert a Time object into a fractional MJD.
        The fractional part of the MJD is derived as (# of seconds)/86400
        unless the specified day includes a leap second _and_ `smear_leapsec`
        is `True`, in which case it is derived as (# of seconds)/86401.
        With `smear_leapsec=False`, times during a leap second will be
        converted into times during the first second of the next day.

        Setting `smear_leapsec=False` is consistent with TEMPO, Tempo2, and
        PINT's `PulsarMJD`, while `smear_leapsec=True` is consistent with
        IAU SOFA and is the default behavior of Astropy. For context, see
        https://github.com/astropy/astropy/issues/5369.
        """
        if smear_leapsec and int(mjd) in leapsec_mjds:
            seconds_in_day = 86401
        else:
            seconds_in_day = 86400
        return self.mjd + (self.second + self.offset)/seconds_in_day
