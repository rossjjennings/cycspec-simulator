import numba as nb
from numba.experimental import jitclass

@jitclass([('F0', nb.float64)])
class FreqOnlyPredictor:
    def __init__(self, F0):
        self.F0 = F0

    def phase(self, t):
        return self.F0*t
