import numba as nb
from numba.experimental import jitclass

@jitclass([
    ('t', nb.float64[:]),
    ('X', nb.complex128[:]),
    ('Y', nb.complex128[:]),
    ('L', nb.complex128[:]),
    ('R', nb.complex128[:]),
])
class BasebandData:
    feed_poln: str

    def __init__(self, t, A, B, feed_poln):
        self.t = t
        self.feed_poln = feed_poln
        if feed_poln.upper() == 'LIN':
            self.X = A
            self.Y = B
        elif feed_poln.upper() == 'CIRC':
            self.L = A
            self.R = B
        else:
            raise ValueError(f"Invalid polarization type '{feed_poln}'.")
