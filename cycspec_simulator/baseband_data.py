class BasebandData:
    def __init__(self, t, A, B, feed_poln):
        if feed_poln.upper() == 'LIN':
            return LinearBasebandData(t, A, B)
        if feed_poln.upper() == 'CIRC':
            return CircularBasebandData(t, A, B)

class LinearBasebandData(BasebandData):
    feed_poln = 'LIN'
    def __init__(self, t, X, Y):
        self.t = t
        self.X = X
        self.Y = Y

class CircularBasebandData(BasebandData):
    feed_poln = 'CIRC'
    def __init__(self, t, L, R):
        self.t = t
        self.L = L
        self.R = R
