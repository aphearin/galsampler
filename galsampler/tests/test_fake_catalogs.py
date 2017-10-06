"""
"""
import numpy as np

from .fake_catalogs import fake_source_galaxy_catalog


def test1():
    n = int(1e3)
    Lbox = 250.
    galaxy_catalog = fake_source_galaxy_catalog(num_source_gals=n, Lbox=Lbox)
    ngal_msg = "Length of galaxy_catalog = {0} != {1}"
    ngal = len(galaxy_catalog)
    assert ngal == n, ngal_msg.format(ngal, n)

    fsat_msg = "Fsat = {0:.2f} of fake catalog should be between 0.1 and 0.5"
    fsat = np.mean(galaxy_catalog['satellite'])
    assert 0.1 < fsat < 0.75, fsat_msg.format(fsat)

    assert np.all(galaxy_catalog['x'] > 0)
    assert np.all(galaxy_catalog['y'] > 0)
    assert np.all(galaxy_catalog['z'] > 0)
    assert np.all(galaxy_catalog['x'] < Lbox)
    assert np.all(galaxy_catalog['y'] < Lbox)
    assert np.all(galaxy_catalog['z'] < Lbox)

    dx = galaxy_catalog['x'] - galaxy_catalog['host_x']
    dy = galaxy_catalog['y'] - galaxy_catalog['host_y']
    dz = galaxy_catalog['z'] - galaxy_catalog['host_z']
    dr = np.sqrt(dx**2 + dy**2 + dz**2)/galaxy_catalog['host_rvir']
    assert np.all(dr < 1)
