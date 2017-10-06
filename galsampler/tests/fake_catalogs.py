"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import erf

from ..numpy_random_context import NumpyRNGContext

__all__ = ('fake_source_galaxy_catalog', )


source_gals_type_list = list((('gal_id', 'i8'), ('mass', 'f4'), ('conc', 'f4'),
    ('satellite', '?'), ('host_halo_id', 'i8'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('host_x', 'f4'), ('host_y', 'f4'), ('host_z', 'f4')))
default_dt_source_gals = np.dtype([(str(a[0]), str(a[1])) for a in source_gals_type_list])

target_halos_type_list = list((('mvir', 'f4'), ('nfw_conc', 'f4'),
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('halo_id', 'i8')))
default_dt_target_halos = np.dtype([(str(a[0]), str(a[1])) for a in target_halos_type_list])


def fake_source_galaxy_catalog(num_source_gals=int(1e3),
        dt_source_gals=default_dt_source_gals, seed=None, Lbox=250.):
    """
    """
    num_source_halos = num_source_gals*5
    halo_mass_array = mc_halo_mass(num_source_halos, seed=seed)

    galaxy_catalog = np.zeros(num_source_gals, dtype=dt_source_gals)

    return galaxy_catalog


def mean_ncen(logM, logM_min, sigma_logM):
    return 0.5*(1.0 + erf((logM - logM_min) / sigma_logM))


def mean_nsat(m, logM_min):
    log_M0 = logM_min-0.3
    log_M1 = log_M0+1.5
    M0, M1 = 10**log_M0, 10**log_M1
    return np.maximum((m - M0)/M1, 0.0001)


def mean_ngal(m, logM_min, sigma_logM=0.2):
    return mean_ncen(np.log10(m), logM_min, sigma_logM) + mean_nsat(m, logM_min)


def halo_abundance(logM, logM_min=9.):
    return 0.5*(1.0 + erf((logM - logM_min) / 3))


def mc_halo_mass(num_halos=int(1e3), seed=None):
    logM_table = np.linspace(10., 15.5, 500)

    unscaled_cdf_inv_table = halo_abundance(logM_table)
    ymin = unscaled_cdf_inv_table.min()
    ymax = unscaled_cdf_inv_table.max()
    dy = ymax-ymin
    cdf_inv_table = unscaled_cdf_inv_table/dy
    cdf_inv_table -= cdf_inv_table.min()

    with NumpyRNGContext(seed):
        randoms = np.random.rand(num_halos)

    return np.interp(randoms, cdf_inv_table, logM_table)

