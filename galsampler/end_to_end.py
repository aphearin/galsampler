"""
"""
import numpy as np
from halotools.utils import crossmatch
from astropy.table import Table
from .utils import compute_richness
from .source_halo_selection import source_halo_index_selection
from .source_galaxy_selection import source_galaxy_index_selection


def source_galaxy_selection_indices(source_galaxies, source_halos, target_halos,
            nhalo_min, *bins, **kwargs):
    """
    Examples
    --------
    source_galaxies : Numpy structured array
        Ndarray of shape (num_source_gals, ) storing the source galaxies that
        occupy the source halos.
        Column names must include ``halo_id`` and ``host_halo_id``,
        or otherwise be defined via the optional source_galaxies_colnames keyword.

    source_halos : Numpy structured array
        Ndarray of shape (num_source_halos, ) storing the catalog of host halos
        that are occupied by the source galaxies.
        Column names must include ``halo_id``, and ``bin_number``,
        or otherwise be defined via the optional source_halos_colnames keyword.

        The ``bin_number`` column can be computed using the
        `galsampler.halo_bin_indices` function.

    target_halos : Numpy structured array
        Ndarray of shape (num_target_halos, ) storing the catalog of host halos
        that will become populated by a Monte Carlo sampling of the source galaxies.
        Column names must include ``bin_number``.

        The ``bin_number`` column can be computed using the
        `galsampler.halo_bin_indices` function.

    nhalo_min : int
        Minimum permissible number of halos in source catalog for a cell to be
        considered well-sampled

    *bins : sequence
        Sequence of arrays that were used to bin the halos

    source_galaxies_colnames : dict, optional

    source_halos_colnames : dict, optional

    target_halos_colnames : dict, optional

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the indices
        of the selected galaxies
    """
    source_galaxies = Table(source_galaxies)
    source_halos = Table(source_halos)
    target_halos = Table(target_halos)

    source_galaxies_colnames = dict(halo_id='halo_id', host_halo_id='host_halo_id')
    source_galaxies_colnames.update(kwargs.get('source_galaxies_colnames', {}))
    _check_colname_correspondence_dictionary(source_galaxies_colnames, source_galaxies, "source_galaxies")

    source_halos_colnames = kwargs.get('source_halos_colnames',
            dict(halo_id='halo_id', bin_number='bin_number'))

    target_halos_colnames = kwargs.get('target_halos_colnames',
            dict(bin_number='bin_number'))

    source_halos['richness'] = compute_richness(
                source_halos[source_halos_colnames['halo_id']],
                source_galaxies[source_galaxies_colnames['host_halo_id']])

    #  Broadcast the ``bin_number`` and ``richness`` columns
    #  from the source halos to the source galaxies
    idxA, idxB = crossmatch(source_galaxies[source_galaxies_colnames['host_halo_id']],
            source_halos[source_halos_colnames['halo_id']])
    source_galaxies['bin_number'] = 0
    source_galaxies['richness'] = 0
    source_galaxies['bin_number'][idxA] = source_halos[source_halos_colnames['bin_number']][idxB]
    source_galaxies['richness'][idxA] = source_halos['richness'][idxB]

    #  Sort the source galaxies so that members of a common halo are grouped together
    source_galaxies.sort(source_galaxies_colnames['host_halo_id'])

    #  For every target halo, calculate the index of a source halo whose resident
    #  galaxies will populate the target halo.
    source_halo_selection_indices = source_halo_index_selection(
            source_halos[source_halos_colnames['bin_number']],
            target_halos[target_halos_colnames['bin_number']], nhalo_min, *bins)

    #  For each selected source halo, determine the index of the first
    #  appearance of a source galaxy that resides in that halo
    #  The algorithm below is predicated upon the source galaxies being sorted by ``host_halo_id``
    uval, indx_uval = np.unique(
        source_galaxies[source_galaxies_colnames['host_halo_id']], return_index=True)
    selected_halo_ids = source_halos[source_halos_colnames['halo_id']][source_halo_selection_indices]
    idxA, idxB = crossmatch(selected_halo_ids, uval)
    representative_galaxy_selection_indices = indx_uval[idxB]

    #  Call the cython kernel to calculate all relevant galaxy indices
    #  for each selected source halo
    return source_galaxy_index_selection(representative_galaxy_selection_indices,
                        source_galaxies['richness'][representative_galaxy_selection_indices])


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


