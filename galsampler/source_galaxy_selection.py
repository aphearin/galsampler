"""
"""
import numpy as np
from .cython_kernels import source_galaxy_index_selection_kernel


def source_galaxy_index_selection(idx, n):
    """
    Examples
    --------
    >>> idx = np.random.randint(0, 10, 100)
    >>> n = np.random.randint(1, 5, 100)
    >>> indices = source_galaxy_index_selection(idx, n)
    """
    idx = np.atleast_1d(idx)
    n = np.atleast_1d(n)
    return np.array(source_galaxy_index_selection_kernel(idx, n))

