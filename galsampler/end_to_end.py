"""
"""
import numpy as np
from halotools.utils import crossmatch
from .utils import compute_richness
from .source_halo_selection import source_halo_index_selection
from .source_galaxy_selection import source_galaxy_index_selection


def source_galaxy_selection_indices(source_galaxies, source_halos, target_halos,
            nhalo_min, *bins):
    """
    Examples
    --------
    source_galaxies : Numpy structured array
        Ndarray of shape (num_source_gals, ) storing the source galaxies that
        occupy the source halos.
        Column names must include ``halo_id`` and ``host_halo_id``.

    source_halos : Numpy structured array
        Ndarray of shape (num_source_halos, ) storing the catalog of host halos
        that are occupied by the source galaxies.
        Column names must include ``halo_id``, and ``bin_number``.
        The ``bin_number`` column can be computed using the
        `galsampler.host_halo_binning.halo_bin_indices` function.

    target_halos : Numpy structured array
        Ndarray of shape (num_target_halos, ) storing the catalog of host halos
        that will become populated by a Monte Carlo sampling of the source galaxies.
        Column names must include ``bin_number``.
        The ``bin_number`` column can be computed using the
        `galsampler.host_halo_binning.halo_bin_indices` function.

    nhalo_min : int
        Minimum permissible number of halos in source catalog for a cell to be
        considered well-sampled

    bins : sequence
        Sequence of arrays that were used to bin the halos

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the indices
        of the selected galaxies
    """
    source_halos['richness'] = compute_richness(
                source_halos['halo_id'], source_galaxies['host_halo_id'])

    #  Broadcast the ``bin_number`` and ``richness`` columns
    #  from the source halos to the source galaxies
    idxA, idxB = crossmatch(source_galaxies['host_halo_id'], source_halos['halo_id'])
    source_galaxies['bin_number'] = 0
    source_galaxies['richness'] = 0
    source_galaxies['bin_number'][idxA] = source_halos['bin_number'][idxB]
    source_galaxies['richness'][idxA] = source_halos['richness'][idxB]

    #  Sort the source galaxies so that members of a common halo are grouped together
    source_galaxies.sort(['bin_number', 'host_halo_id'])

    #  For every target halo, calculate the index of a source halo whose resident
    #  galaxies will populate the target halo.
    source_halo_selection_indices = source_halo_index_selection(
            source_halos['bin_number'], target_halos['bin_number'], nhalo_min, *bins)

    #  For each selected source halo, determine the index of the first
    #  appearance of a source galaxy that resides in that halo
    #  The algorithm below is predicated upon the source galaxies being sorted by ``host_halo_id``
    uval, indx_uval = np.unique(source_galaxies['host_halo_id'], return_index=True)
    selected_halo_ids = source_halos['halo_id'][source_halo_selection_indices]
    idxA, idxB = crossmatch(selected_halo_ids, uval)
    representative_galaxy_selection_indices = indx_uval[idxB]

    #  Call the cython kernel to calculate all relevant galaxy indices
    #  for each selected source halo
    return source_galaxy_index_selection(representative_galaxy_selection_indices,
                        source_galaxies['richness'][representative_galaxy_selection_indices])
