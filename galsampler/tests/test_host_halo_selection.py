"""
"""
from time import time
import numpy as np
from ..host_halo_selection import source_halo_index_selection, compute_richness


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


def test_compute_richness1():
    unique_halo_ids = [5, 2, 100]
    halo_id_of_galaxies = [100, 2, 100, 3, 2, 100, 100, 3]
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 2, 4])


def test_compute_richness2():
    unique_halo_ids = [400, 100, 200, 300]
    halo_id_of_galaxies = np.random.randint(0, 50, 200)
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 0, 0, 0])


def test_compute_richness3():
    unique_halo_ids = [400, 100, 200, 300]
    halo_id_of_galaxies = [0, 999, 100, 200, 100, 200, 999, 300, 200]
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 2, 3, 1])

