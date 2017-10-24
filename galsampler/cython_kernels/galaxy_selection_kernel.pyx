"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

cimport cython
import numpy as np


__all__ = ('source_galaxy_index_selection_kernel', 'alt_galaxy_selection_kernel')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def source_galaxy_index_selection_kernel(long[:] idx, long[:] n):
    """
    """
    cdef int num_source_halos = idx.shape[0]

    cdef int i, num_target_gals
    for i in range(num_source_halos):
        num_target_gals += n[i]

    cdef long[:] indices = np.zeros(num_target_gals).astype(long)

    cdef long cur = 0
    cdef long ifirst, j, nselect

    for i in range(num_source_halos):
        ifirst = idx[i]
        nselect = n[i]
        for j in range(nselect):
            indices[cur] = ifirst + j
            cur += 1

    return indices


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def alt_galaxy_selection_kernel(long[:] first_source_gal_indices, int[:] richness, int ngal_tot):
    """
    """
    cdef long[:] result = np.zeros(ngal_tot).astype('i8')
    cdef int i, j, n
    cdef int cur = 0
    cdef int ifirst, ilast
    cdef int nhalo = first_source_gal_indices.shape[0]
    for i in range(nhalo):
        ifirst = first_source_gal_indices[i]
        n = richness[i]
        if n > 0:
            ilast = ifirst + richness[i]
            for j in range(ifirst, ilast):
                result[cur] = j
                cur += 1
    return result
