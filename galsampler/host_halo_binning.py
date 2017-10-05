"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('halo_bin_indices', )


def halo_bin_indices(source_halo_arrays, halo_property_bins):
    """
    Function used to bin the properties of the source halos.

    Parameters
    ----------
    source_halo_arrays : dict
        Python dictionary storing the collection of properties of the source halos.
        Each key should be the name of the property to be binned;
        each value an ndarray of shape (num_halos, ).
        The keys must match the keys in ``halo_property_bins``.

    halo_property_bins : dict
        Python dictionary storing the collection of bins.
        Each key should be the name of the property to be binned;
        each value an ndarray storing the bin edges for that property.
        The keys must match the keys in ``source_halo_arrays``.

    Returns
    -------
    bin_indices_dict : dict
        Python dictionary storing the bin numbers of each halo for each property.
        The keys `bin_indices_dict` will be the same as the input dictionaries;
        the values will be integer arrays of shape (num_halos, ).

    Notes
    -----
    This function calls `numpy.digitize` separately for each array.
    For cases where the source halo property is larger than the largest bin edge,
    `numpy.digitize` returns `num_bins`. In such cases, the `halo_bin_indices` function
    over-writes these values with `num_bins-1`.

    Examples
    --------
    >>> num_halos = 50
    >>> mass = 10**np.random.uniform(10, 15, num_halos)
    >>> conc = np.random.uniform(1, 25, num_halos)
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)

    >>> source_halo_arrays = dict(mass=mass, conc=conc)
    >>> halo_property_bins = dict(mass=mass_bins, conc=conc_bins)
    >>> bin_indices_dict = halo_bin_indices(source_halo_arrays, halo_property_bins)
    """
    try:
        halo_properties = set(list(source_halo_arrays.keys()))
        _b = set(list(halo_property_bins.keys()))
        assert halo_properties == _b
        assert len(halo_properties) >= 1
    except:
        msg = ("Input ``source_halo_arrays`` and ``halo_property_bins`` must be \n"
            "non-empty python dictionaries with the same keys")
        raise ValueError(msg)

    bin_indices_dict = {}
    for key in halo_properties:
        array, bins = source_halo_arrays[key], halo_property_bins[key]
        bin_indices = np.minimum(np.digitize(array, bins), len(bins)-1)
        bin_indices_dict[key] = bin_indices

    return bin_indices_dict
