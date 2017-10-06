"""
"""
import pytest
import numpy as np

from .fake_catalogs import fake_target_halo_catalog

from ..populate_halos import _enforce_key_correspondence, format_target_catalog


def test1():
    source_galaxies = dict(x=4)
    target_halos = dict(x=4)
    halo_property_bins = dict(x=4)
    source_target_key_correspondence = {}

    _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
            source_target_key_correspondence)


def test2():
    source_galaxies = dict(x=4)
    target_halos = dict(x=4)
    halo_property_bins = dict(x=4, y=5)
    source_target_key_correspondence = {}

    with pytest.raises(KeyError) as err:
        _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
                source_target_key_correspondence)
    substr = "not in ``source_galaxies``"
    assert substr in err.value.args[0]


def test3():
    source_galaxies = dict(x=4, y=5)
    target_halos = dict(x=4)
    halo_property_bins = dict(x=4, y=5)
    source_target_key_correspondence = {}

    with pytest.raises(KeyError) as err:
        _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
                source_target_key_correspondence)
    substr = "but not in ``target_halos`` nor ``source_target_key_correspondence``"
    assert substr in err.value.args[0]


def test4():
    source_galaxies = dict(x=4, y=5)
    target_halos = dict(x=4, z=9)
    halo_property_bins = dict(x=4, y=5)
    source_target_key_correspondence = dict(y='t')

    with pytest.raises(KeyError) as err:
        _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
                source_target_key_correspondence)
    substr = "According to ``source_target_key_correspondence``"
    assert substr in err.value.args[0]


def test5():
    source_galaxies = dict(x=4, y=5)
    target_halos = dict(x=4, z=9)
    halo_property_bins = dict(x=4, y=5)
    source_target_key_correspondence = dict(y='z')

    _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
            source_target_key_correspondence)


def test6():
    target_halos = fake_target_halo_catalog()
    num_bins_mass, num_bins_conc = 12, 11
    mass_bins = np.logspace(10, 15, num_bins_mass)
    conc_bins = np.logspace(1.5, 20, num_bins_conc)

    halo_property_bins = dict(mass=mass_bins, conc2=conc_bins)

    source_target_key_correspondence = {'conc2': 'conc3'}
    with pytest.raises(KeyError) as err:
        formatted_target_halos = format_target_catalog(target_halos, halo_property_bins,
                source_target_key_correspondence=source_target_key_correspondence)
    substr = "should be a ``conc3`` key in ``target_halos``"
    assert substr in err.value.args[0]


def test7():
    target_halos = fake_target_halo_catalog()
    num_bins_mass, num_bins_conc = 12, 11
    mass_bins = np.logspace(10, 15, num_bins_mass)
    conc_bins = np.logspace(1.5, 20, num_bins_conc)

    halo_property_bins = dict(mass=mass_bins, conc2=conc_bins)

    source_target_key_correspondence = {'conc2': 'conc'}
    formatted_target_halos = format_target_catalog(target_halos, halo_property_bins,
            source_target_key_correspondence=source_target_key_correspondence)



