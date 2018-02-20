"""
"""
import numpy as np


__all__ = ('matched_value_selection_indices', )


def matched_value_selection_indices(x, y, assume_x_is_sorted=False):
    """ For each element of ``y``, find the index of the closest value in ``x``.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts1, )

    y : ndarray
        Numpy array of shape (npts2, )

    assume_x_is_sorted : bool, optional
        If True, array ``x`` will be assumed to be in monotonically increasing order,
        improving performance. Default is False.

    Returns
    -------
    selection_indices : ndarray
        Numpy integer array of shape (npts2, ) storing values in the interval [0, npts1).

    Examples
    --------
    >>> npts1, npts2 = 100, 300
    >>> x = np.random.rand(npts1)
    >>> y = np.random.rand(npts2)
    >>> idx = matched_value_selection_indices(x, y)
    >>> closest_matching_values = x[idx]
    """
    if assume_x_is_sorted:
        x_sorted = x
    else:
        idx_sorted_source = np.argsort(x)
        x_sorted = x[idx_sorted_source]

    num_source = len(x)
    idx_selection = np.searchsorted(x_sorted, y)
    idx_selection = np.where(idx_selection >= num_source, num_source-1, idx_selection)

    if assume_x_is_sorted:
        return idx_selection
    else:
        return idx_sorted_source[idx_selection]
