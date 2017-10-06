"""
"""

__all__ = ('NumpyRNGContext', )


class NumpyRNGContext(object):
    """ Context manager enabling deterministic results from
    functions in np.random.

    The `NumpyRNGContext` context manager preserves
    the entry value of the input number seed upon exit.

    Notes
    -----
    This code has been lifted wholesale from `astropy.utils.misc`
    """
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        from numpy import random

        random.set_state(self.startstate)
