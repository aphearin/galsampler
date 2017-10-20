"""
"""
import numpy as np
from ..utils import compute_richness


__all__ = ('test_compute_richness1', )


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

