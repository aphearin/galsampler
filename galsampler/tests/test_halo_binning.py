"""
"""
import numpy as np

from ..host_halo_binning import halo_bin_indices


def test1():
    num_halos = 100
    source_halo_arrays = dict(
            a=np.linspace(0, 10, num_halos), b=np.linspace(10, 20, num_halos))

    num_bins_a, num_bins_b = 5, 15
    halo_property_bins = dict(
            a=np.linspace(0, 10, num_bins_a), b=np.linspace(10, 20, num_bins_b))

    bin_indices_dict = halo_bin_indices(source_halo_arrays, halo_property_bins)
    assert np.all(bin_indices_dict['a'] >= 0)
    assert np.all(bin_indices_dict['b'] >= 0)
    assert np.all(bin_indices_dict['a'] <= num_bins_a-1)
    assert np.all(bin_indices_dict['b'] <= num_bins_b-1)
