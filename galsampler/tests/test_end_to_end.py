"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from ..end_to_end import source_galaxy_selection_indices


def test1_bijective_case():
    """
    Setup:

    * Each source halo belongs to a unique bin.
    * There exists a unique target halo for every source halo.
    * Each source halo is populated with a single galaxy.

    Verify:

    * The target galaxy catalog is an exact replica of the source galaxy catalog
    """
    num_source_halos = 10
    num_galaxies = num_source_halos
    num_target_halos = num_source_halos
    nhalo_min = 1

    source_halo_dt_list = [(str('halo_id'), str('i8')), (str('bin_number'), str('i4'))]
    source_halos_dtype = np.dtype(source_halo_dt_list)
    source_halos = np.zeros(num_source_halos, dtype=source_halos_dtype)
    source_halos['halo_id'] = np.arange(num_source_halos).astype(int)
    source_halos['bin_number'] = np.arange(num_source_halos).astype(int)

    source_galaxy_dt_list = [(str('halo_id'), str('i8')), (str('host_halo_id'), str('i4'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['halo_id'] = np.arange(num_galaxies).astype(int)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.arange(num_target_halos).astype(int)

    fake_bins = np.arange(num_source_halos)
    indices = source_galaxy_selection_indices(source_galaxies, source_halos, target_halos,
            nhalo_min, fake_bins)
    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == len(source_galaxies)
    assert np.all(selected_galaxies['halo_id'] == source_galaxies['halo_id'])
    assert np.all(selected_galaxies['host_halo_id'] == source_galaxies['host_halo_id'])
    assert np.all(indices == np.arange(len(indices)))


def test2_bijective_case():
    """
    Setup:

    * Each source halo belongs to a unique bin.
    * There exists 5 target halos for every source halo.
    * Each source halo is populated with a single galaxy.

    Verify:

    * The target galaxy catalog is a 5x repetition of the source galaxy catalog
    """
    num_source_halos = 10
    num_galaxies = num_source_halos
    num_target_halos = num_source_halos*5
    nhalo_min = 1

    source_halo_dt_list = [(str('halo_id'), str('i8')), (str('bin_number'), str('i4'))]
    source_halos_dtype = np.dtype(source_halo_dt_list)
    source_halos = np.zeros(num_source_halos, dtype=source_halos_dtype)
    source_halos['halo_id'] = np.arange(num_source_halos).astype(int)
    source_halos['bin_number'] = np.arange(num_source_halos).astype(int)

    source_galaxy_dt_list = [(str('halo_id'), str('i8')), (str('host_halo_id'), str('i4'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['halo_id'] = np.arange(num_galaxies).astype(int)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.repeat(source_halos['bin_number'], 5)

    fake_bins = np.arange(num_source_halos)
    indices = source_galaxy_selection_indices(source_galaxies, source_halos, target_halos,
            nhalo_min, fake_bins)
    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == num_target_halos
    assert np.all(selected_galaxies == np.repeat(source_galaxies, 5))


@pytest.mark.xfail
def test3_bijective_case():
    """
    Identical setup to test1_bijective_case,
    but the initialization uses the host_halo_binning_module
    for integration testing purposes.
    """
    log_mhost_min, log_mhost_max, dlog_mhost = 9.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])
    num_galaxies = len(log_mhost_mids)
    raise NotImplementedError


@pytest.mark.xfail
def test_constant_scaleup_case():
    """
    """
    raise NotImplementedError


@pytest.mark.xfail
def test_empty_halos_case():
    """
    """
    raise NotImplementedError


