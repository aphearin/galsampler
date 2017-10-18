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
    num_source_halos = len(idx)
    num_target_gals = np.sum(n)
    indices = np.zeros(num_target_gals).astype(long)

    cur = 0
    for i in range(num_source_halos):
        ifirst = idx[i]
        nselect = n[i]
        for j in range(nselect):
            indices[cur] = ifirst + j
            cur += 1

    return indices

