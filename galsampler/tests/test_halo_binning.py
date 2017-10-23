"""
"""
import numpy as np

from ..host_halo_binning import halo_bin_indices


def test1_1bin():
    num_halos = 100
    num_bin_edges = 5
    num_bins = num_bin_edges-1
    bins = np.linspace(0, 1, num_bins)
    x = np.linspace(0, 1, num_halos)
    bin_numbers = halo_bin_indices(x=(x, bins))
    assert np.all(bin_numbers <= num_bins - 1)
    assert np.all(bin_numbers >= 0)


def test2_1bin():
    num_halos = 100
    num_bin_edges = 5
    num_bins = num_bin_edges-1
    bins = np.linspace(0, 1, num_bins)
    x = np.linspace(-100, 100, num_halos)
    bin_numbers = halo_bin_indices(x=(x, bins))
    assert np.all(bin_numbers <= num_bins - 1)
    assert np.all(bin_numbers >= 0)


def test1_2bins():
    num_halos = 100
    num_bins_a, num_bins_b = 5, 15

    haloprop_a = np.linspace(0, 10, num_halos)
    haloprop_b = np.linspace(10, 20, num_halos)
    bins_a = np.linspace(0, 10, num_bins_a)
    bins_b = np.linspace(10, 20, num_bins_b)

    bin_numbers = halo_bin_indices(a=(haloprop_a, bins_a), b=(haloprop_b, bins_b))
    assert np.shape(bin_numbers) == (num_halos, )
    assert np.all(bin_numbers <= num_bins_a*num_bins_b-1)
    assert np.all(bin_numbers >= 0)

