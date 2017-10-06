"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from halotools.utils import crossmatch

from .host_halo_binning import halo_bin_indices


__all__ = ('populate_target_halos', )


def populate_target_halos(source_galaxies, target_halos, halo_property_bins,
        source_target_key_correspondence={}):
    """
    Parameters
    ----------
    source_galaxies : ndarray
        Numpy structured array or Astropy Table of shape (num_source_gals, )
        storing the galaxy catalog that will be scaled up to the target simulation.

    target_halos : ndarray
        Numpy structured array or Astropy Table of shape (num_target_halos, )
        storing the halo catalog that will become populated
        with a Monte Carlo resampling of the source galaxies.

    halo_property_bins : dict
        Python dictionary storing the collection of bins used to match
        the halo populations in the source and target catalogs.
        Each key should be the name of the property to be binned;
        each value an ndarray storing the bin edges for that property.
        Each key of ``halo_property_bins`` must appear as a column name in
        ``source_galaxies``. Additionally, for any key that does not appear
        as a column name in ``target_halos``, there must be an entry in
        the ``source_target_key_correspondence`` dictionary.

    source_target_key_correspondence : dict, optional
        Python dictionary providing correspondence between
        source/target column names used to bin the halo properties.
        There must be a key for every key of ``halo_property_bins``
        that does not appear as a column name in ``target_halos``.
        The value bound to each such key should be the corresponding colum name
        in the ``target_halos`` catalog.

        Default is an empty dictionary, in which case it will be assumed that
        all column names correspond with one another.

    Returns
    -------
    target_galaxies : ndarray
        Numpy structured array of shape (num_target_gals, ) storing
        the output galaxy catalog

    """
    _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
        source_target_key_correspondence)

    source_galaxies = format_source_catalog(source_galaxies, halo_property_bins)
    target_halos = format_source_catalog(target_halos, halo_property_bins,
            source_target_key_correspondence)
    raise NotImplementedError


def compute_richness(host_halo_ids):
    """ Calculate the multiplicity of each entry of ``host_halo_ids``.

    Parameters
    ----------
    host_halo_ids : ndarray
        Numpy array of (possibly repeated) integers of shape (npts, )

    Returns
    -------
    richness : ndarray
        Numpy array of integers of shape (npts, ) storing the multiplicity of each entry

    Examples
    --------
    >>> host_halo_ids = np.random.randint(0, 100, 1000)
    >>> richness = compute_richness(host_halo_ids)
    """
    nhalos = host_halo_ids.shape[0]
    host_halo_ngals = np.zeros(nhalos).astype('i4')

    uval, idx_uval, counts = np.unique(host_halo_ids, return_index=True, return_counts=True)
    host_halo_ngals[idx_uval] = counts

    idxA, idxB = crossmatch(host_halo_ids, host_halo_ids[idx_uval])
    host_halo_ngals[idxA] = host_halo_ngals[idx_uval][idxB]

    return host_halo_ngals


def format_source_catalog(source_galaxies, halo_property_bins):
    """ Transform the source galaxy catalog into the required format.

    Examples
    --------
    >>> from galsampler.tests import fake_source_galaxy_catalog
    >>> source_galaxies = fake_source_galaxy_catalog()
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)
    >>> halo_property_bins = dict(host_halo_mass=mass_bins, host_halo_conc=conc_bins)
    >>> formatted_source_galaxies = format_source_catalog(source_galaxies, halo_property_bins)
    """
    source_galaxies = Table(source_galaxies)
    source_galaxies['host_halo_ngals'] = compute_richness(source_galaxies['host_halo_id'])

    source_galaxies['halo_bin_number'] = halo_bin_indices(
        {key: source_galaxies[key] for key in halo_property_bins.keys()}, halo_property_bins)

    source_galaxies.sort(('halo_bin_number', 'host_halo_id', 'satellite'))

    return source_galaxies


def _enforce_key_correspondence(source_galaxies, target_halos, halo_property_bins,
        source_target_key_correspondence):
    """
    """
    source_names = list(source_galaxies.keys())
    target_names = list(target_halos.keys())
    binning_names = list(halo_property_bins.keys())
    correspondence_names = list(source_target_key_correspondence.keys())

    missing_source_name_msg = ("Column name ``{0}`` appears in ``halo_property_bins`` "
        "but not in ``source_galaxies``")
    missing_key_correspondence_name_msg = ("Column name ``{0}`` appears in ``halo_property_bins`` "
        "but not in ``target_halos`` nor ``source_target_key_correspondence``")

    missing_target_name_msg = ("Column name ``{0}`` appears in ``halo_property_bins`` "
        "but not in ``source_galaxies``.\n"
        "According to ``source_target_key_correspondence``, there should be a "
        "``{1}`` key in target_halos\nbut this key does not exist in target_halos")

    for binning_name in binning_names:

        try:
            assert binning_name in source_names
        except:
            raise KeyError(missing_source_name_msg.format(binning_name))

        try:
            assert binning_name in target_names
        except:
            try:
                assert binning_name in correspondence_names
            except:
                raise KeyError(missing_key_correspondence_name_msg.format(binning_name))

            try:
                target_name = source_target_key_correspondence[binning_name]
                assert target_name in target_names
            except:
                raise KeyError(missing_target_name_msg.format(binning_name, target_name))


def format_target_catalog(target_halos, halo_property_bins,
        source_target_key_correspondence={}):
    """ Transform the target halo catalog into the required format.

    Parameters
    ----------
    target_halos : ndarray
        Numpy structured array or Astropy Table of shape (num_target_halos, )
        storing the halo catalog that will become populated
        with a Monte Carlo resampling of the source galaxies.

    halo_property_bins : dict
        Python dictionary storing the collection of bins used to match
        the halo populations in the source and target catalogs.
        Each key should be the name of the property to be binned;
        each value an ndarray storing the bin edges for that property.
        For any key of ``halo_property_bins`` that does not appear
        as a column name in ``target_halos``, there must be an entry in
        the ``source_target_key_correspondence`` dictionary.
        The value bound to each such key should be the corresponding colum name
        in the ``target_halos`` catalog.

        See Examples for an explicit demonstration.

    source_target_key_correspondence : dict, optional
        Python dictionary providing correspondence between
        source/target column names used to bin the halo properties.
        For every key of ``halo_property_bins``
        that does not appear as a column name in ``target_halos``,
        there must be a key in ``source_target_key_correspondence``.
        The value bound to each such key should be the corresponding colum name
        in the ``target_halos`` catalog.

        Default is an empty dictionary, in which case it will be assumed that
        all column names correspond with one another.

        See Examples for an explicit demonstration.

    Examples
    --------
    First we show the case where ``target_halos`` and ``halo_property_bins``
    have the same keys:

    >>> from galsampler.tests import fake_target_halo_catalog
    >>> target_halos = fake_target_halo_catalog()
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)
    >>> halo_property_bins = dict(mass=mass_bins, conc=conc_bins)
    >>> formatted_target_halos = format_target_catalog(target_halos, halo_property_bins)

    Now we show the case requiring the ``source_target_key_correspondence`` feature:

    >>> halo_property_bins = dict(mass=mass_bins, other_conc_name=conc_bins)
    >>> source_target_key_correspondence = dict(other_conc_name='conc')
    >>> formatted_target_halos = format_target_catalog(target_halos, halo_property_bins, source_target_key_correspondence=source_target_key_correspondence)

    """
    missing_colname_msg = ("Column name ``{0}`` appears in ``halo_property_bins`` "
        "but not in ``target_halos``.\n"
        "According to ``source_target_key_correspondence``, there should be a "
        "``{1}`` key in ``target_halos``\nbut this key does not exist in ``target_halos``")

    target_halos = Table(target_halos)

    source_halo_arrays = {}
    for binning_key in halo_property_bins.keys():
        try:
            source_halo_arrays[binning_key] = target_halos[binning_key]
        except KeyError:
            try:
                matching_key = source_target_key_correspondence[binning_key]
                source_halo_arrays[binning_key] = target_halos[matching_key]
            except KeyError:
                raise KeyError(missing_colname_msg.format(binning_key, matching_key))
                source_halo_arrays[binning_key] = target_halos

    target_halos['halo_bin_number'] = halo_bin_indices(source_halo_arrays, halo_property_bins)

    target_halos.sort('halo_bin_number')

    return target_halos
