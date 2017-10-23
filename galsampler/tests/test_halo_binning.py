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


def test3_1bin():
    num_halos = 1111
    bin_edges = np.array((0, 0.25, 0.5, 0.75, 1.))
    x = np.linspace(-0.1, 1.1, num_halos)
    bin_numbers = halo_bin_indices(x=(x, bin_edges))

    #  Manual check of all bins
    bin0_mask = bin_numbers == 0
    assert np.all(x[bin0_mask] <= 0.25)
    bin1_mask = bin_numbers == 1
    assert np.all(x[bin1_mask] >= 0.25)
    assert np.all(x[bin1_mask] <= 0.5)
    bin2_mask = bin_numbers == 2
    assert np.all(x[bin2_mask] >= 0.5)
    assert np.all(x[bin2_mask] <= 0.75)
    bin3_mask = bin_numbers == 3
    assert np.all(x[bin3_mask] >= 0.75)

    assert set(bin_numbers) == {0, 1, 2, 3}


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


def test2_2bins():
    num_halos = 1111
    bin1_edges = np.array((0, 0.25, 0.5, 0.75, 1.))
    bin2_edges = np.array((0.25, 0.5, 0.75))
    x = np.linspace(-0.1, 1.1, num_halos)
    y = np.random.uniform(-0.1, 1.1, num_halos)

    bin_numbers = halo_bin_indices(x=(x, bin1_edges), y=(y, bin2_edges))
    assert set(bin_numbers) == {0, 1, 2, 3, 4, 5, 6, 7}

    #  Manual check of all bins
    bin0_mask = bin_numbers == 0
    assert np.all(x[bin0_mask] <= 0.25)
    assert np.all(y[bin0_mask] <= 0.5)

    bin1_mask = bin_numbers == 1
    assert np.all(x[bin1_mask] >= 0.25)
    assert np.all(x[bin1_mask] <= 0.5)
    assert np.all(y[bin1_mask] <= 0.5)

    bin2_mask = bin_numbers == 2
    assert np.all(x[bin2_mask] >= 0.5)
    assert np.all(x[bin2_mask] <= 0.75)
    assert np.all(y[bin2_mask] <= 0.5)

    bin3_mask = bin_numbers == 3
    assert np.all(x[bin3_mask] >= 0.75)
    assert np.all(y[bin3_mask] <= 0.5)

    bin4_mask = bin_numbers == 4
    assert np.all(x[bin4_mask] <= 0.25)
    assert np.all(y[bin4_mask] >= 0.5)

    bin5_mask = bin_numbers == 5
    assert np.all(x[bin5_mask] >= 0.25)
    assert np.all(x[bin5_mask] <= 0.5)
    assert np.all(y[bin5_mask] >= 0.5)

    bin6_mask = bin_numbers == 6
    assert np.all(x[bin6_mask] >= 0.5)
    assert np.all(x[bin6_mask] <= 0.75)
    assert np.all(y[bin6_mask] >= 0.5)

    bin7_mask = bin_numbers == 7
    assert np.all(x[bin7_mask] >= 0.75)
    assert np.all(y[bin7_mask] >= 0.5)
