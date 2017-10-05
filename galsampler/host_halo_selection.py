"""
"""
import numpy as np
from .cython_kernels import source_halo_index_selection_kernel


def source_halo_index_selection(first, last, num_select):
    """
    Examples
    --------
    >>> first = np.array((2, 3, 4))
    >>> last = np.array((8, 9, 10))
    >>> num_select = np.array((1, 1, 2))
    >>> result = source_halo_index_selection(first, last, num_select)
    """
    first = np.atleast_1d(first).astype('i8')
    last = np.atleast_1d(last).astype('i8')
    num_select = np.atleast_1d(num_select).astype('i8')
    assert len(first) == len(last) == len(num_select), "Input 1d arrays must be the same length"
    assert np.all(last - first > 0), "Must have at least one source halo per target"
    return np.array(source_halo_index_selection_kernel(first, last, num_select))
