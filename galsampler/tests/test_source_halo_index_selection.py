"""
"""
import numpy as np

from ..source_halo_selction import source_halo_index_selection
from ..numpy_random_context import NumpyRNGContext


__all__ = ('test_source_halo_index_selection_no_missing_source_cells', )


def test_source_halo_index_selection_no_missing_source_cells():
    num_sources, num_target = int(1e2), int(1e3)
    num_bins = 10
    with NumpyRNGContext(43):
        source_halo_bin_numbers = np.random.randint(0, num_bins, num_sources)
        target_halo_bin_numbers = np.random.randint(0, num_bins, num_target)

    source_indices = source_halo_index_selection(
            source_halo_bin_numbers, target_halo_bin_numbers)

    unique_target_bins = np.unique(target_halo_bin_numbers)
    for ibin in unique_target_bins:
        mask = target_halo_bin_numbers == ibin
        assert np.all(source_halo_bin_numbers[source_indices[mask]] == ibin)
