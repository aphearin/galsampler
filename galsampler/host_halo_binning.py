"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('halo_bin_indices', )


def halo_bin_indices(**haloprop_and_bins_dict):
    """ Calculate a unique cell ID for every host halo.

    Parameters
    ----------
    haloprop_and_bins_dict : dict
        Python dictionary storing the collection of halo properties and bins.
        Each key should be the name of the property to be binned;
        each value should be a two-element tuple storing two ndarrays,
        the first with shape (num_halos, ), the second with shape (nbins, ),
        where ``nbins`` is allowed to vary from property to property.

    Returns
    -------
    cell_ids : ndarray
        Numpy integer array of shape (num_halos, ) storing the integer of the
        (possibly multi-dimensional) bin of each halo.

    Examples
    --------
    In this example, we bin our halos simultaneously by mass and concentration:

    >>> num_halos = 50
    >>> mass = 10**np.random.uniform(10, 15, num_halos)
    >>> conc = np.random.uniform(1, 25, num_halos)
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)

    >>> cell_ids = halo_bin_indices(mass=(mass, mass_bins), conc=(conc, conc_bins))

    In this case, all values in the ``cell_ids`` array
    will be in the interval [0, num_bins_mass*num_bins_conc).
    """
    bin_indices_dict = {}
    for haloprop_name in haloprop_and_bins_dict.keys():
        arr, bins = haloprop_and_bins_dict[haloprop_name]
        bin_indices = np.minimum(np.digitize(arr, bins), len(bins)-1)
        bin_indices_dict[haloprop_name] = bin_indices

    num_bins_dict = {key: len(haloprop_and_bins_dict[key][1]) for key in haloprop_and_bins_dict.keys()}

    return np.ravel_multi_index(list(bin_indices_dict.values()),
            list(num_bins_dict.values()))
