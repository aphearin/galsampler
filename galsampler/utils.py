"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from halotools.utils import crossmatch


def compute_richness(unique_halo_ids, halo_id_of_galaxies):
    """
    """
    unique_halo_ids = np.atleast_1d(unique_halo_ids)
    halo_id_of_galaxies = np.atleast_1d(halo_id_of_galaxies)
    richness_result = np.zeros_like(unique_halo_ids).astype(int)

    vals, counts = np.unique(halo_id_of_galaxies, return_counts=True)
    idxA, idxB = crossmatch(vals, unique_halo_ids)
    richness_result[idxB] = counts[idxA]
    return richness_result


