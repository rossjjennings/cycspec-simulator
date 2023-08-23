class FreqOnlyPredictor:
    def __init__(self, F0=None, P=None):
        if F0 is not None:
            self.F0 = F0
        elif P is not None:
            self.F0 = 1/P
        else:
            raise ValueError("Specify either F0 or P")

    def __call__(self, t):
        return self.F0*t
