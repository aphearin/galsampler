"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..matched_halo_selection_1d import matched_value_selection_indices


fixed_seed = 43


def test1():
    """
    """
    npts1 = int(1e5)
    with NumpyRNGContext(fixed_seed):
        x = 10**np.random.uniform(10, 15, npts1)
        y = np.random.choice(x, len(x), replace=False)
    idx = matched_value_selection_indices(x, y)
    assert np.allclose(x[idx], y, rtol=0.01)
