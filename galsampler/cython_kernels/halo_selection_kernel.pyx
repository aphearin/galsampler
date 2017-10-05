"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor
import numpy as np


__all__ = ('source_halo_index_selection_kernel', )


cdef double cython_rand():
    cdef double r = rand()
    return r / RAND_MAX

cdef long cython_randint(int low, int high):
    """ Draw a single random integer in the interval [low, high)
    """
    return <long>floor(cython_rand()*(high-low) + low)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def source_halo_index_selection_kernel(long[:] first_index_source_bin,
            long[:] last_index_source_bin, long[:] num_select_per_bin):
    """
    """
    cdef int ibin, ihalo
    cdef int num_bins = first_index_source_bin.shape[0]

    cdef int i, num_select_ibin
    cdef long num_select_total = 0

    for i in range(num_bins):
        num_select_total += num_select_per_bin[i]

    cdef long[:] indices = np.zeros(num_select_total, dtype='i8')

    i = 0
    for ibin in range(num_bins):
        num_select_ibin = num_select_per_bin[ibin]
        low_ibin = first_index_source_bin[ibin]
        high_ibin = last_index_source_bin[ibin]
        for ihalo in range(num_select_ibin):
            indices[i] = cython_randint(low_ibin, high_ibin)
            i += 1

    return indices







