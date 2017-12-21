"""
"""
import numpy as np
from halotools.utils import distribution_matching_indices


__all__ = ('source_halo_index_selection', )


def get_source_bin_from_target_bin(source_bin_counts, bin_number, nhalo_min, bin_shapes):
    """
    """
    if source_bin_counts[bin_number] >= nhalo_min:
        return bin_number
    else:
        idx = np.unravel_index(bin_number, bin_shapes)
        num_cells_total = np.product(bin_shapes)

        seq = list((bin_number, taxicab_metric(idx, np.unravel_index(bin_number, bin_shapes)))
            for bin_number in range(num_cells_total) if source_bin_counts[bin_number] >= nhalo_min)
        sorted_seq = sorted(seq, key=lambda s: s[1])
        return sorted_seq[0][0]


def taxicab_metric(arr1, arr2):
    return sum(abs(y-x) for x, y in zip(arr1, arr2))


def _check_source_binning(source_bin_counts, nhalo_min, frac_good_required=0.5):
    num_bins_with_good_sampling = np.count_nonzero(source_bin_counts >= nhalo_min)
    frac_good = num_bins_with_good_sampling/float(len(source_bin_counts))
    msg = ("The fraction of cells in the source catalog \nwith "
    "more halos than nhalo_min={0} is {1:.2f} < frac_good_required={2:.2f}")

    if num_bins_with_good_sampling == 0:
        raise ValueError(msg.format(nhalo_min, frac_good, frac_good_required))


def source_halo_index_selection(source_halo_bin_numbers,
            target_halo_bin_numbers, target_halo_ids, nhalo_min, *bins, **kwargs):
    """ Randomly select the index of a host halo from the source catalog
    for every halo in the target halo catalog.

    When possible, only source halos from the same multi-dimensional bin as the
    host halo will be selected. However, if a cell has fewer than ``nhalo_min``
    halos in the source catalog, then a source halo will instead be selected from
    the nearest adjacent cell with more than ``nhalo_min`` objects.

    Parameters
    ----------
    source_halo_bin_numbers : ndarray
        Numpy integer array of shape (num_source_halos, ) storing the bin number
        of every halo in the source catalog. This bin number can be calculated
        using the `halo_bin_indices` function.

    target_halo_bin_numbers : ndarray
        Numpy integer array of shape (num_target_halos, ) storing the bin number
        of every halo in the target catalog. This bin number can be calculated
        using the `halo_bin_indices` function.

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
    selection_indices : ndarray
        Numpy integer array of shape (num_target_halos, ) storing the index of
        the halo selected from the source catalog whose galaxies will populate
        the target halo

    matching_target_halo_ids : ndarray
        Numpy integer array of shape (num_target_halos, ) storing the halo ID
        of the target halo corresponding to each selected source halo
    """
    intra_bin_selection_method = kwargs.get('intra_bin_selection_method', 'random')

    num_source_halos = len(source_halo_bin_numbers)
    selection_indices = np.arange(num_source_halos).astype('i8')

    bin_shapes = tuple(len(arr)-1 for arr in bins)
    num_cells_total = np.product(bin_shapes)

    cell_bins = np.arange(-0.5, num_cells_total+0.5, 1)
    source_bin_counts = np.histogram(source_halo_bin_numbers, cell_bins)[0]
    _check_source_binning(source_bin_counts, nhalo_min)

    result = np.zeros_like(target_halo_bin_numbers).astype('i8')
    matching_target_halo_ids = np.zeros_like(target_halo_bin_numbers).astype('i8')

    for target_bin in range(num_cells_total):
        target_bin_mask = target_halo_bin_numbers == target_bin
        num_target_halos_in_bin = np.count_nonzero(target_bin_mask)

        if num_target_halos_in_bin > 0:
            source_bin = get_source_bin_from_target_bin(
                    source_bin_counts, target_bin, nhalo_min, bin_shapes)
            source_bin_mask = source_halo_bin_numbers == source_bin
            source_bin_indices = selection_indices[source_bin_mask]

            if intra_bin_selection_method == 'random':
                result[target_bin_mask] = randomly_select_source_halos_within_bin(
                            source_bin_indices, num_target_halos_in_bin)
                matching_target_halo_ids[target_bin_mask] = target_halo_ids[target_bin_mask]
            elif intra_bin_selection_method == 'hod_matching':
                try:
                    source_bin_richness = kwargs['source_richness'][source_bin_mask]
                    data_bin_richness = np.atleast_1d(kwargs['data_richness'][target_bin])
                    assert data_bin_richness.shape[0] > 10
                    result[target_bin_mask] = hod_matching_halo_bin_selection(
                        source_bin_indices, source_bin_richness,
                        data_bin_richness, num_target_halos_in_bin)
                except KeyError:
                    required_kwargs = ('source_richness', 'data_richness')
                    msg = ("When selecting the `hod_matching` option, "
                        "you must also pass the following keyword arguments:\n{0}")
                    raise KeyError(msg.format(required_kwargs))
                except (IndexError, TypeError):
                    msg = ("``source_richness`` keyword argument must store "
                        "an integer ndarray of shape (num_source_halos, ) = ({0}, )\n"
                        "``data_richness`` keyword argument must store "
                        "a list of num_target_halo_bins={1} ndarrays of richness-matching data")
                    raise ValueError(msg.format(num_source_halos, num_cells_total))
                except AssertionError:
                    msg = ("For target_bin = {0}, there are only {1} elements of ``data_richness``")
                    raise ValueError(msg.format(target_bin, data_bin_richness.shape[0]))
            else:
                msg = ("keyword argument ``intra_bin_selection_method`` "
                    "can only take the following values:\n{0}")
                available_methods = ('random', 'hod_matching')
                raise ValueError(msg.format(available_methods))

    return result, matching_target_halo_ids


def randomly_select_source_halos_within_bin(source_bin_indices, num_target_halos_in_bin):
    """
    """
    return np.random.choice(source_bin_indices, num_target_halos_in_bin, replace=True)


def hod_matching_halo_bin_selection(source_bin_indices, source_bin_richness,
            data_bin_richness, num_target_halos_in_bin):
    """
    """
    max_richness = max(np.max(source_bin_richness), np.max(data_bin_richness))
    richness_bins = np.arange(0, max_richness+1) - 0.01
    return source_bin_indices[distribution_matching_indices(source_bin_richness, data_bin_richness,
            num_target_halos_in_bin, richness_bins)]
