"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from halotools.utils import crossmatch

from .cython_kernels import source_halo_index_selection_kernel
from .host_halo_binning import halo_bin_indices


def source_halo_index_selection(first, last, num_select):
    """
    Parameters
    ----------
    first : ndarray
        Numpy integer array of shape (nbins, ) storing the index
        of the first source halo in each bin

    last : ndarray
        Numpy integer array of shape (nbins, ) storing the index
        of the last source halo in each bin

    num_select : ndarray
        Numpy integer array of shape (nbins, ) storing the
        number of times to draw from each bin

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_select.sum(), ) storing the
        indices of the source halos to be selected

    Examples
    --------
    >>> first = np.array((2, 3, 4))
    >>> last = np.array((8, 9, 10))
    >>> num_select = np.array((1, 1, 2))
    >>> result = source_halo_index_selection(first, last, num_select)
    """
    first = np.atleast_1d(first).astype('i8')
    last = np.atleast_1d(last).astype('i8')
    num_select = np.atleast_1d(num_select).astype('i8')
    assert len(first) == len(last) == len(num_select), "Input 1d arrays must be the same length"
    assert np.all(last - first > 0), "Must have at least one source halo per target"
    return np.array(source_halo_index_selection_kernel(first, last, num_select))


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
