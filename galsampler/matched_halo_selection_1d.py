"""
"""
import numpy as np


__all__ = ('mass_matched_selection_indices', )


def mass_matched_selection_indices(x, y, assume_x_is_sorted=False):
    """
    """
    if assume_x_is_sorted:
        x_source_sorted = x
    else:
        idx_sorted_source = np.argsort(x)
        x_source_sorted = x[idx_sorted_source]

    num_source = len(x)
    idx_selection = np.searchsorted(x_source_sorted, y)
    idx_selection = np.where(idx_selection >= num_source, num_source-1, idx_selection)

    if assume_x_is_sorted:
        return idx_selection
    else:
        return idx_sorted_source[idx_selection]
