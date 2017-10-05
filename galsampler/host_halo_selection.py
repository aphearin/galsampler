"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from .cython_kernels import source_halo_index_selection_kernel


def source_halo_index_selection(first, last, num_select):
    """
    Examples
    --------
    >>> first = np.array((2, 3, 4))
    >>> last = np.array((8, 9, 10))
    >>> num_select = 8
    >>> result = source_halo_index_selection(first, last, num_select)
    """
    result = source_halo_index_selection_kernel(first, last, num_select)
