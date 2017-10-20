"""
"""
import numpy as np
from halotools.utils import crossmatch

from .source_halo_grouping import get_source_halo_bin_numbers
from .target_halo_grouping import get_target_halo_bin_numbers
from .utils import compute_richness
from .source_halo_selection import source_halo_index_selection


def source_galaxy_selection_indices(source_galaxies, source_halos, target_halos, **kwargs):
    """
    """
    target_halos['bin_number'] = get_target_halo_bin_numbers(target_halos, **kwargs)
    source_halos['bin_number'] = get_source_halo_bin_numbers(source_halos, **kwargs)

    source_halos['richness'] = compute_richness(
                source_halos['halo_id'], source_galaxies['halo_hostid'])

    idxA, idxB = crossmatch(source_galaxies['halo_hostid'].data, source_halos['halo_id'].data)
    source_galaxies['bin_number'] = 0
    source_galaxies['richness'] = 0
    source_galaxies['bin_number'][idxA] = source_halos['bin_number'][idxB]
    source_galaxies['richness'][idxA] = source_halos['richness'][idxB]
    source_galaxies.sort(['bin_number', 'halo_hostid'])

    source_halo_selection_indices = source_halo_index_selection(
            source_halos['bin_number'], target_halos['bin_number'], **kwargs)

    uval, indx_uval = np.unique(source_galaxies['halo_hostid'], return_index=True)
    selected_halo_ids = source_halos['halo_id'][source_halo_selection_indices]
    idxA, idxB = crossmatch(selected_halo_ids, uval)

    return indx_uval[idxB]
