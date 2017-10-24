"""
"""
import numpy as np
from halotools.utils import crossmatch
from .utils import compute_richness
from .source_halo_selection import source_halo_index_selection, alt_source_halo_index_selection
from .source_galaxy_selection import source_galaxy_index_selection
from .cython_kernels import alt_galaxy_selection_kernel


def source_galaxy_selection_indices(source_galaxies_host_halo_id,
            source_halos_halo_id, source_halos_bin_number,
            target_halos_bin_number, target_halo_ids,
            nhalo_min, *bins, **kwargs):
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

    target_halo_ids : ndarray
        Numpy integer array of shape (num_target_halos, )
        storing the ID of every halo in the target halo catalog.

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

    matching_target_halo_ids : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the halo ID
        of the target halo hosting each selected source galaxy
    """
    #  Sort the source galaxies so that members of a common halo are grouped together
    idx_sorted_source_galaxies = np.argsort(source_galaxies_host_halo_id)
    sorted_source_galaxies_host_halo_id = source_galaxies_host_halo_id[idx_sorted_source_galaxies]
    num_source_gals = len(sorted_source_galaxies_host_halo_id)

    source_halos_richness = compute_richness(
                source_halos_halo_id, sorted_source_galaxies_host_halo_id)

    #  For each target halo, calculate the index of the source halo whose resident
    #  galaxies will populate the target halo.
    source_halo_selection_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halos_bin_number, target_halos_bin_number, target_halo_ids, nhalo_min, *bins)
    selected_source_halo_ids = source_halos_halo_id[source_halo_selection_indices]
    selected_source_halo_richness = source_halos_richness[source_halo_selection_indices]
    target_galaxy_target_halo_ids = np.repeat(matching_target_halo_ids, selected_source_halo_richness)

    #  For every target halo, we now know the richness and also the index of the source halo

    #  Broadcast richness from the source halos to the source galaxies
    idxA, idxB = crossmatch(sorted_source_galaxies_host_halo_id, source_halos_halo_id)
    sorted_source_galaxies_richness = np.zeros_like(sorted_source_galaxies_host_halo_id)
    sorted_source_galaxies_richness[idxA] = source_halos_richness[idxB]

    #  For each selected source halo, determine the index of the first
    #  appearance of a source galaxy that resides in that halo
    #  The algorithm below is predicated upon the source galaxies being sorted by ``host_halo_id``
    uval_gals, indx_uval_gals = np.unique(sorted_source_galaxies_host_halo_id, return_index=True)
    uval_halos, indx_uval_halos, multiplicity_halos = np.unique(
                selected_source_halo_ids, return_index=True, return_counts=True)
    idxA, idxB = crossmatch(uval_halos, uval_gals)
    idxB_with_multiplicity = np.repeat(idxB, multiplicity_halos[idxA])
    representative_galaxy_selection_indices = indx_uval_gals[idxB_with_multiplicity]

    #  Call the cython kernel to calculate all relevant galaxy indices
    #  for each selected source halo
    sorted_source_galaxy_indices = source_galaxy_index_selection(representative_galaxy_selection_indices,
                        sorted_source_galaxies_richness[representative_galaxy_selection_indices])

    return sorted_source_galaxy_indices, target_galaxy_target_halo_ids


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


def alt_source_galaxy_selection_indices(source_galaxies_host_halo_id,
            source_halos_bin_number, source_halos_halo_id,
            target_halos_bin_number, target_halo_ids, nhalo_min, *bins):
    """
    """
    #  Sort the source galaxies so that members of a common halo are grouped together
    idx_sorted_source_galaxies = np.argsort(source_galaxies_host_halo_id)
    sorted_source_galaxies_host_halo_id = source_galaxies_host_halo_id[idx_sorted_source_galaxies]

    #  Calculate the index correspondence array that will undo the sorting at the end
    num_source_gals = len(source_galaxies_host_halo_id)
    idx_unsorted_galaxy_indices = np.arange(num_source_gals).astype('i8')[idx_sorted_source_galaxies]

    #  For each source halo, calculate the number of resident galaxies
    source_halos_richness = compute_richness(
                source_halos_halo_id, sorted_source_galaxies_host_halo_id)

    #  For each source halo, calculate the index of its first resident galaxy
    source_halo_sorted_source_galaxies_indices = _galaxy_table_indices(
                source_halos_halo_id, sorted_source_galaxies_host_halo_id)

    #  For each target halo, calculate the index of the associated source halo
    source_halo_selection_indices, matching_target_halo_ids = alt_source_halo_index_selection(
            source_halos_bin_number, target_halos_bin_number, target_halo_ids, nhalo_min, *bins)

    #  For each target halo, calculate the number of galaxies
    target_halo_richness = source_halos_richness[source_halo_selection_indices]
    num_target_gals = np.sum(target_halo_richness)

    #  For each target halo, calculate the halo ID of the associated source halo
    target_halo_source_halo_ids = source_halos_halo_id[source_halo_selection_indices]

    #  For each target halo, calculate the index of its first resident galaxy
    target_halo_first_sorted_source_gal_indices = (
                source_halo_sorted_source_galaxies_indices[source_halo_selection_indices])

    #  For every target halo, we know the index of the first and last galaxy to select
    #  Calculate an array of shape (num_target_gals, ) with the index of each selected galaxy
    sorted_source_galaxy_selection_indices = np.array(alt_galaxy_selection_kernel(
            target_halo_first_sorted_source_gal_indices.astype('i8'),
            target_halo_richness.astype('i4'), num_target_gals))

    #  For each target galaxy, calculate the halo ID of its source and target halo
    target_galaxy_target_halo_ids = np.repeat(matching_target_halo_ids, target_halo_richness)
    target_galaxy_source_halo_ids = np.repeat(target_halo_source_halo_ids, target_halo_richness)

    #  For each index in the sorted galaxy catalog,
    #  calculate the index of the catalog in its original order
    selection_indices = idx_unsorted_galaxy_indices[sorted_source_galaxy_selection_indices]

    return (selection_indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids)


def _galaxy_table_indices(source_halo_id, galaxy_host_halo_id):
    """ For every halo in the source halo catalog, calculate the index
    in the source galaxy catalog of the first appearance of a galaxy that
    occupies the halo, reserving -1 for source halos with no resident galaxies.

    Parameters
    ----------
    source_halo_id : ndarray
        Numpy integer array of shape (num_halos, )

    galaxy_host_halo_id : ndarray
        Numpy integer array of shape (num_gals, )

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_halos, ).
        All values will be in the interval [-1, num_gals)
    """
    uval_gals, indx_uval_gals = np.unique(galaxy_host_halo_id, return_index=True)
    idxA, idxB = crossmatch(source_halo_id, uval_gals)
    num_source_halos = len(source_halo_id)
    indices = np.zeros(num_source_halos) - 1
    indices[idxA] = indx_uval_gals[idxB]
    return indices

