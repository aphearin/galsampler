"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

cimport cython
import numpy as np


__all__ = ('source_galaxy_index_selection_kernel', )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def source_galaxy_index_selection_kernel(long[:] first_index_source_bin,
            long[:] last_index_source_bin, long[:] num_select_per_bin):
    """
    """
    return None
