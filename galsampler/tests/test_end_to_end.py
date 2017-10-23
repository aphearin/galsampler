
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from ..end_to_end import source_galaxy_selection_indices
from ..host_halo_binning import halo_bin_indices


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

    source_galaxy_dt_list = [(str('host_halo_id'), str('i8'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.arange(num_target_halos).astype(int)

    fake_bins = np.arange(num_source_halos)
    indices = source_galaxy_selection_indices(source_galaxies['host_halo_id'],
            source_halos['halo_id'], source_halos['bin_number'], target_halos['bin_number'], nhalo_min,
            fake_bins)

    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == len(source_galaxies)
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

    source_galaxy_dt_list = [(str('host_halo_id'), str('i8'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.repeat(source_halos['bin_number'], 5)

    fake_bins = np.arange(num_source_halos)
    indices = source_galaxy_selection_indices(source_galaxies['host_halo_id'],
            source_halos['halo_id'], source_halos['bin_number'], target_halos['bin_number'], nhalo_min,
            fake_bins)

    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == num_target_halos
    assert np.all(selected_galaxies == np.repeat(source_galaxies, 5))


def test3():
    """
    """
    #  Set up a source halo catalog with 100 halos in each mass bin
    log_mhost_min, log_mhost_max, dlog_mhost = 10.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])
    num_distinct_source_halo_masses = len(log_mhost_mids)

    #  source_halos column names must include ``halo_id`` and ``bin_number``
    num_source_halos_per_bin = 100
    source_halo_log_mhost = np.tile(log_mhost_mids, num_source_halos_per_bin)
    num_source_halos = len(source_halo_log_mhost)
    source_halo_id = np.arange(num_source_halos).astype(int)
    source_halo_bin_number = halo_bin_indices(log_mhost=(source_halo_log_mhost, log_mhost_bins))
    assert len(source_halo_bin_number) == num_distinct_source_halo_masses*num_source_halos_per_bin, "Bad setup of source_halos"

    #  source_galaxies column names must include ``halo_id`` and ``host_halo_id``
    ngals_per_source_halo = 3
    num_source_galaxies = num_source_halos*ngals_per_source_halo
    source_galaxy_host_halo_id = np.repeat(source_halo_id, ngals_per_source_halo)
    source_galaxy_host_mass = np.repeat(source_halo_log_mhost, ngals_per_source_halo)
    assert len(source_galaxy_host_mass) == num_source_galaxies, "Bad setup of source_galaxies"

    #  target_halos column names must include ``bin_number``
    num_target_halos_per_source_halo = 121
    target_halo_bin_number = np.repeat(source_halo_bin_number, num_target_halos_per_source_halo)
    num_target_halos = len(target_halo_bin_number)

    nhalo_min = 5
    indices = source_galaxy_selection_indices(source_galaxy_host_halo_id,
                source_halo_id, source_halo_bin_number, target_halo_bin_number,
                nhalo_min, log_mhost_bins)
    selected_galaxies_host_mass = source_galaxy_host_mass[indices]
    correct_num_target_galaxies = int(num_source_galaxies*(num_target_halos/float(num_source_halos)))
    msg = "target_galaxies does not have correct length"
    assert len(selected_galaxies_host_mass) == correct_num_target_galaxies, msg


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


