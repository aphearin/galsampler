"""
"""
import numpy as np

from halotools.utils import unsorting_indices

__all__ = ('source_halo_index_selection', )


def source_halo_index_selection(source_halo_bin_numbers, target_halo_bin_numbers):
    """
    """
    idx_sorted_source_halo_bin_numbers = np.argsort(source_halo_bin_numbers)
    idx_sorted_target_halo_bin_numbers = np.argsort(target_halo_bin_numbers)

    selection_indices = np.arange(len(source_halo_bin_numbers))[idx_sorted_source_halo_bin_numbers]

    sorted_source_halo_bin_numbers = source_halo_bin_numbers[idx_sorted_source_halo_bin_numbers]
    sorted_target_halo_bin_numbers = target_halo_bin_numbers[idx_sorted_target_halo_bin_numbers]

    unique_target_vals, idx_target, target_counts = np.unique(sorted_target_halo_bin_numbers,
            return_index=True, return_counts=True)

    result = np.zeros_like(target_halo_bin_numbers).astype('i8')

    gen = zip(unique_target_vals, idx_target, target_counts)
    for bin_index, starting_sorted_target_idx, num_target_halos in gen:
        ending_sorted_target_idx = starting_sorted_target_idx + num_target_halos

        low_sorted_source_idx, high_sorted_source_idx = np.searchsorted(
                sorted_source_halo_bin_numbers, [bin_index, bin_index+1])

        print(low_sorted_source_idx, high_sorted_source_idx, num_target_halos)
        randoms = np.random.randint(low_sorted_source_idx, high_sorted_source_idx, num_target_halos)

        result[starting_sorted_target_idx:ending_sorted_target_idx] = selection_indices[randoms]

    idx_target_unsorted = unsorting_indices(idx_sorted_target_halo_bin_numbers)
    return result[idx_target_unsorted]
