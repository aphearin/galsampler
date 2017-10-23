"""
"""
import numpy as np
import pytest

from ..source_halo_selection import source_halo_index_selection, get_source_bin_from_target_bin
from ..numpy_random_context import NumpyRNGContext


__all__ = ('test_source_halo_index_selection_no_missing_source_cells', )


def test_source_halo_index_selection_no_missing_source_cells():
    """
    """
    nhalo_min = 25
    num_sources, num_target = int(1e4), int(1e5)
    num_bins1, num_bins2 = 3, 4
    bin1 = np.linspace(0, 1, num_bins1)
    bin2 = np.linspace(0, 1, num_bins2)
    num_bins = num_bins1*num_bins2
    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_bins, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_bins, num_target)
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
    num_bins1 = 3
    bin1 = np.linspace(0, 1, num_bins1)
    num_bins = num_bins1
    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_bins, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_bins, num_target)
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


def test_get_source_bin_from_target_bin():
    bin_shapes = (25, )
    source_bin_counts = np.random.randint(100, 500, 25)
    source_bin_counts[0] = 3
    bin_number = 0
    nhalo_min = 50
    result = get_source_bin_from_target_bin(source_bin_counts, bin_number, nhalo_min, bin_shapes)
    assert result == 1
