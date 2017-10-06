"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('populate_target_halos', )


def populate_target_halos(source_galaxies, target_halos, halo_property_bins):
    """
    Parameters
    ----------
    source_galaxies : ndarray
        Numpy structured array of shape (num_source_gals, ) storing
        the galaxy catalog that will be scaled up to the target simulation.

    target_halos : ndarray
        Numpy structured array of shape (num_target_halos, ) storing
        the halo catalog that will become populated with a Monte Carlo
        resampling of the source halos.

    halo_property_bins : dict
        Python dictionary storing the collection of bins used to match
        the halo populations in the source and target catalogs.
        Each key should be the name of the property to be binned;
        each value an ndarray storing the bin edges for that property.
        Each key of ``halo_property_bins`` must appear in both
        ``source_galaxies`` and ``target_halos``.

    Returns
    -------
    target_galaxies : ndarray
        Numpy structured array of shape (num_target_gals, ) storing
        the output galaxy catalog

    """
    raise NotImplementedError
