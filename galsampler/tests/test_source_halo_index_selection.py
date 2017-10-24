"""
"""
import numpy as np
import pytest
from halotools.utils import crossmatch
from astropy.utils.misc import NumpyRNGContext

from ..source_halo_selection import source_halo_index_selection, get_source_bin_from_target_bin
from ..host_halo_binning import halo_bin_indices


__all__ = ('test_source_halo_index_selection_no_missing_source_cells', )


def test_source_halo_index_selection_no_missing_source_cells():
    """
    """
    nhalo_min = 25
    num_sources, num_target = int(1e4), int(1e5)
    num_bin1_edges, num_bin2_edges = 7, 4
    bin1 = np.linspace(0, 1, num_bin1_edges)
    bin2 = np.linspace(0, 1, num_bin2_edges)
    num_cells_total = (num_bin1_edges-1)*(num_bin2_edges-1)
    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_cells_total, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_cells_total, num_target)
        target_halo_ids = np.arange(num_target).astype('i8')

    source_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halo_bin_numbers, target_halo_bin_numbers, target_halo_ids,
            nhalo_min, bin1, bin2)

    unique_target_bins = np.unique(target_halo_bin_numbers)
    for ibin in unique_target_bins:
        mask = target_halo_bin_numbers == ibin
        assert np.all(source_halo_bin_numbers[source_indices[mask]] == ibin)


def test2_source_halo_index_selection_no_missing_source_cells():
    """
    """
    nhalo_min = 25
    num_sources, num_target = int(1e4), int(1e5)
    num_bin1_edges = 7
    bin1 = np.linspace(0, 1, num_bin1_edges)

    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_bin1_edges-1, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_bin1_edges-1, num_target)
        target_halo_ids = np.arange(num_target).astype('i8')

    source_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halo_bin_numbers, target_halo_bin_numbers, target_halo_ids, nhalo_min, bin1)

    unique_target_bins = np.unique(target_halo_bin_numbers)
    for ibin in unique_target_bins:
        mask = target_halo_bin_numbers == ibin
        assert np.all(source_halo_bin_numbers[source_indices[mask]] == ibin)


def test_source_halo_index_selection_missing_source_cells():
    """
    """
    nhalo_min = 25
    num_sources, num_target = int(1e3), int(1e5)
    num_bins1, num_bins2 = 3, 20
    bin1 = np.linspace(0, 1, num_bins1)
    bin2 = np.linspace(0, 1, num_bins2)
    num_bins = num_bins1*num_bins2
    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_bins, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_bins, num_target)
        target_halo_ids = np.arange(num_target).astype('i8')

    with pytest.raises(ValueError) as err:
        source_indices = source_halo_index_selection(
                source_halo_bin_numbers, target_halo_bin_numbers, target_halo_ids, nhalo_min, bin1, bin2)

    substr = "The fraction of cells in the source catalog"
    assert substr in err.value.args[0]


def test1_get_source_bin_from_target_bin():
    bin_shapes = (25, )
    source_bin_counts = np.random.randint(100, 500, 25)
    source_bin_counts[0] = 3
    bin_number = 0
    nhalo_min = 50
    result = get_source_bin_from_target_bin(source_bin_counts, bin_number, nhalo_min, bin_shapes)
    assert result == 1


def test2_get_source_bin_from_target_bin():
    num_bin_edges = 5
    xbins = np.linspace(0, 1, num_bin_edges)
    x = np.array((0.1, 0.1, 0.4, 0.4, 0.9, 0.9))
    bin_numbers = halo_bin_indices(x=(x, xbins))
    assert set(bin_numbers) == {0, 1, 3}

    counts = np.array([2, 2, 0, 2])
    nhalo_min = 2
    bin_shapes = (4, )
    assert get_source_bin_from_target_bin(counts, 0, nhalo_min, bin_shapes) == 0
    assert get_source_bin_from_target_bin(counts, 1, nhalo_min, bin_shapes) == 1
    assert get_source_bin_from_target_bin(counts, 2, nhalo_min, bin_shapes) in (1, 3)
    assert get_source_bin_from_target_bin(counts, 3, nhalo_min, bin_shapes) == 3


def test_bin_distribution_recovery():
    log_mhost_min, log_mhost_max, dlog_mhost = 10.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])

    num_source_halos_per_bin = 10
    source_halo_log_mhost = np.tile(log_mhost_mids, num_source_halos_per_bin)
    source_halo_bin_number = halo_bin_indices(log_mhost=(source_halo_log_mhost, log_mhost_bins))

    num_target_halos_per_source_halo = 11
    target_halo_bin_number = np.repeat(source_halo_bin_number, num_target_halos_per_source_halo)
    target_halo_log_mhost = np.repeat(source_halo_log_mhost, num_target_halos_per_source_halo)
    num_target_halos = len(target_halo_bin_number)
    target_halo_ids = np.arange(num_target_halos).astype('i8')

    nhalo_min = 5
    source_halo_selection_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halo_bin_number, target_halo_bin_number, target_halo_ids, nhalo_min, log_mhost_bins)

    idxA, idxB = crossmatch(matching_target_halo_ids, target_halo_ids)
    target_mass = target_halo_log_mhost[idxB]
    source_mass = source_halo_log_mhost[source_halo_selection_indices]
    assert np.allclose(target_mass, source_mass)

    source_halo_selection_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halo_bin_number, target_halo_bin_number, target_halo_ids, nhalo_min, log_mhost_bins)

    idxA, idxB = crossmatch(matching_target_halo_ids, target_halo_ids)
    target_mass = target_halo_log_mhost[idxB]
    source_mass = source_halo_log_mhost[source_halo_selection_indices]
    assert np.allclose(target_mass, source_mass)





