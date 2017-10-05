"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
cimport cython


__all__ = ('source_halo_index_selection_kernel', )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def source_halo_index_selection_kernel(long[:] first, long[:] last, int num_select):
    return 5
