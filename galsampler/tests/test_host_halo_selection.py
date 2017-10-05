"""
"""
from time import time
import numpy as np
from ..host_halo_selection import source_halo_index_selection


__all__ = ('test1', )


def test1():
    nbins = 100
    first = np.arange(nbins)
    last = np.arange(nbins) + nbins
    num_select = np.random.randint(0, 5, nbins)
    indices = source_halo_index_selection(first, last, num_select)
    assert len(indices) == num_select.sum(), "Incorrect output shape of source_halo_index_selection"


def test_performance():
    nbins = int(1e6)
    first = np.arange(nbins)
    last = np.arange(nbins) + nbins
    num_select = np.ones(nbins)

    start = time()
    __ = source_halo_index_selection(first, last, num_select)
    end = time()
    assert end-start < 1., "source_halo_index_selection function has performance bug"

