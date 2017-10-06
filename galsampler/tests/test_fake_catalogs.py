"""
"""
import numpy as np

from .fake_catalogs import fake_source_galaxy_catalog


def test1():
    n = int(1e3)
    galaxy_catalog = fake_source_galaxy_catalog(num_source_gals=n)
    assert len(galaxy_catalog) == n

    fsat_msg = "Fsat = {0:.2f} of fake catalog should be between 0.1 and 0.5"
    fsat = np.mean(galaxy_catalog['satellite'])
    assert 0.1 < fsat < 0.5, fsat_msg.format(fsat)
