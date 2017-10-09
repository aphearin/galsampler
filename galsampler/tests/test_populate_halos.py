"""
"""
import numpy as np
from ..populate_halos import populate_target_halos
from .fake_catalogs import fake_source_galaxy_catalog, fake_target_halo_catalog


def test1():
    source_galaxies = fake_source_galaxy_catalog()
    target_halos = fake_target_halo_catalog()
    raise NotImplementedError("Unfinished end-to-end test on fake catalogs")
