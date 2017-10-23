"""
"""
import numpy as np
from halotools.utils import crossmatch
from .utils import compute_richness
from .source_halo_selection import source_halo_index_selection
from .source_galaxy_selection import source_galaxy_index_selection


def source_galaxy_selection_indices(source_galaxies_host_halo_id,
            source_halos_halo_id, source_halos_bin_number, target_halos_bin_number, nhalo_min,
            *bins, **kwargs):
    """
    Examples
    --------
    source_galaxies_host_halo_id : ndarray
        Numpy integer array of shape (num_source_gals, )
        storing the ID of the host halo of each source galaxy.
        In particular, if the galaxy occupies a subhalo of some larger host halo,
        the value for source_galaxies_host_halo_id of that galaxy should be the
        ID of the larger host halo.

    source_halos_halo_id : ndarray
        Numpy integer array of shape (num_source_halos, )
        storing the ID of every halo in the source halo catalog.

        Note that it is important to include a *complete* sample of source halos,
        including those that do not host a source galaxy.

    source_halos_bin_number : ndarray
        Numpy integer array of shape (num_source_halos, )
        storing the bin number assigned to every halo in the source halo catalog.

        Note that it is important to include a *complete* sample of source halos,
        including those that do not host a source galaxy.

        The bin_number can be computed using the `galsampler.halo_bin_indices` function,
        `np.digitize`, or some other means.

    target_halos_bin_number : ndarray
        Numpy integer array of shape (num_target_halos, )
        storing the bin number assigned to every halo in the target halo catalog.

        The bin_number can be computed using the `galsampler.halo_bin_indices` function,
        `np.digitize`, or some other means.

    nhalo_min : int
        Minimum permissible number of halos in source catalog for a cell to be
        considered well-sampled

    *bins : sequence
        Sequence of arrays that were used to bin the halos

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the indices
        of the selected galaxies
    """
    source_halos_richness = compute_richness(
                source_halos_halo_id, source_galaxies_host_halo_id)

    #  Broadcast bin_number and richness from the source halos to the source galaxies
    idxA, idxB = crossmatch(source_galaxies_host_halo_id, source_halos_halo_id)
    source_galaxies_bin_number = np.zeros_like(source_galaxies_host_halo_id)
    source_galaxies_richness = np.zeros_like(source_galaxies_host_halo_id)
    source_galaxies_bin_number[idxA] = source_halos_bin_number[idxB]
    source_galaxies_richness[idxA] = source_halos_richness[idxB]

    #  Sort the source galaxies so that members of a common halo are grouped together
    idx_sorted_source_galaxies = np.argsort(source_galaxies_host_halo_id)
    source_galaxies_bin_number = source_galaxies_bin_number[idx_sorted_source_galaxies]
    source_galaxies_richness = source_galaxies_richness[idx_sorted_source_galaxies]
    source_galaxies_bin_number = source_galaxies_bin_number[idx_sorted_source_galaxies]
    source_galaxies_host_halo_id = source_galaxies_host_halo_id[idx_sorted_source_galaxies]

    #  For each target halo, calculate the index of the source halo whose resident
    #  galaxies will populate the target halo.
    source_halo_selection_indices = source_halo_index_selection(
            source_halos_bin_number, target_halos_bin_number, nhalo_min, *bins)
    selected_source_halo_ids = source_halos_halo_id[source_halo_selection_indices]

    #  For each selected source halo, determine the index of the first
    #  appearance of a source galaxy that resides in that halo
    #  The algorithm below is predicated upon the source galaxies being sorted by ``host_halo_id``
    uval_gals, indx_uval_gals = np.unique(source_galaxies_host_halo_id, return_index=True)
    uval_halos, indx_uval_halos, multiplicity_halos = np.unique(
                selected_source_halo_ids, return_index=True, return_counts=True)
    idxA, idxB = crossmatch(uval_halos, uval_gals)
    idxB_with_multiplicity = np.repeat(idxB, multiplicity_halos[idxA])
    representative_galaxy_selection_indices = indx_uval_gals[idxB_with_multiplicity]

    #  Call the cython kernel to calculate all relevant galaxy indices
    #  for each selected source halo
    return source_galaxy_index_selection(representative_galaxy_selection_indices,
                        source_galaxies_richness[representative_galaxy_selection_indices])


def _check_colname_correspondence_dictionary(d, catalog, catalog_varname):
    """
    """
    keyword_name = catalog_varname + "_colnames"
    for key, value in d.items():
        try:
            assert value in catalog.dtype.names
        except AssertionError:
            msg = ("{0} does not contain a ``{1}`` column name\n"
                "Either rename some column in {0} or use the {2} keyword")
            raise KeyError(msg.format(catalog_varname, value, keyword_name))


