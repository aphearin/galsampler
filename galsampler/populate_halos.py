"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from halotools.utils import crossmatch

from .host_halo_binning import halo_bin_indices


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


def compute_richness(host_halo_ids):
    """
    """
    nhalos = host_halo_ids.shape[0]
    host_halo_ngals = np.zeros(nhalos).astype('i4')

    uval, idx_uval, counts = np.unique(host_halo_ids, return_index=True, return_counts=True)
    host_halo_ngals[idx_uval] = counts

    idxA, idxB = crossmatch(host_halo_ids, host_halo_ids[idx_uval])
    host_halo_ngals[idxA] = host_halo_ngals[idx_uval][idxB]

    return host_halo_ngals


def format_source_catalog(source_halos, halo_property_bins):
    """
    Examples
    --------
    >>> from galsampler.tests import fake_source_galaxy_catalog
    >>> source_halos = fake_source_galaxy_catalog()
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)
    >>> halo_property_bins = dict(host_halo_mass=mass_bins, host_halo_conc=conc_bins)
    >>> formatted_source_halos = format_source_catalog(source_halos, halo_property_bins)
    """
    source_halos = Table(source_halos)
    source_halos['host_halo_ngals'] = compute_richness(source_halos['host_halo_id'])

    source_halos['halo_bin_number'] = halo_bin_indices(
        {key: source_halos[key] for key in halo_property_bins.keys()}, halo_property_bins)

    source_halos.sort(('halo_bin_number', 'host_halo_id', 'satellite'))

    return source_halos
