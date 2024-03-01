from abc import ABCMeta, abstractmethod

class LinearFilter(metaclass=ABCMeta):
    """
    Abstract base class for linear filters which can be applied to baseband data.
    An instance should have a well-defined impulse response function and corresponding
    frequency response function.
    """
    def apply(self, data):
        """
        Apply this filter pattern to baseband data.

        Parameters
        ----------
        data: A BasebandData object.

        Returns
        -------
        new_data: A BasebandData object.
        """
        pass
