class BasebandData:
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
