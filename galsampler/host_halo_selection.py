"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .cython_kernels import source_halo_index_selection_kernel


def source_halo_index_selection(first, last, num_select):
    """
    Parameters
    ----------
    first : ndarray
        Numpy integer array of shape (nbins, ) storing the index
        of the first source halo in each bin

    last : ndarray
        Numpy integer array of shape (nbins, ) storing the index
        of the last source halo in each bin

    num_select : ndarray
        Numpy integer array of shape (nbins, ) storing the
        number of times to draw from each bin

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_select.sum(), ) storing the
        indices of the source halos to be selected

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
