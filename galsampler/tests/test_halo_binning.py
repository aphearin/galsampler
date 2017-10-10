"""
"""
import numpy as np

from ..host_halo_binning import halo_bin_indices


def test1():
    num_halos = 100
    num_bins_a, num_bins_b = 5, 15

    haloprop_a = np.linspace(0, 10, num_halos)
    haloprop_b = np.linspace(10, 20, num_halos)
    bins_a = np.linspace(0, 10, num_bins_a)
    bins_b = np.linspace(10, 20, num_bins_b)

    bin_indices = halo_bin_indices(a=(haloprop_a, bins_a), b=(haloprop_b, bins_b))
    assert np.shape(bin_indices) == (num_halos, )
    assert np.all(bin_indices <= num_bins_a*num_bins_b-1)
