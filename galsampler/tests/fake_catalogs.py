"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import erf
from scipy.stats import poisson

from ..numpy_random_context import NumpyRNGContext

__all__ = ('fake_source_galaxy_catalog', )


source_gals_type_list = list((('gal_id', 'i8'),
    ('host_halo_mass', 'f4'), ('host_halo_conc', 'f4'), ('host_halo_rvir', 'f4'),
    ('satellite', '?'), ('host_halo_id', 'i8'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('host_halo_x', 'f4'), ('host_halo_y', 'f4'), ('host_halo_z', 'f4')))
default_dt_source_gals = np.dtype([(str(a[0]), str(a[1])) for a in source_gals_type_list])

target_halos_type_list = list((('mvir', 'f4'), ('nfw_conc', 'f4'),
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('halo_id', 'i8')))
default_dt_target_halos = np.dtype([(str(a[0]), str(a[1])) for a in target_halos_type_list])


def fake_source_galaxy_catalog(num_source_gals=int(1e3),
        dt_source_gals=default_dt_source_gals, seed=None, Lbox=250.,
        logM_min=12, sigma_logM=0.25):
    """
    Examples
    --------
    >>> galaxy_catalog = fake_source_galaxy_catalog()
    """
    num_source_halos = num_source_gals*5

    m = 10**mc_halo_mass(num_source_halos, seed=seed)
    host_x = np.random.uniform(1, Lbox-1, num_source_halos)
    host_y = np.random.uniform(1, Lbox-1, num_source_halos)
    host_z = np.random.uniform(1, Lbox-1, num_source_halos)
    host_conc = 10**np.random.normal(
        loc=np.log10(mean_halo_concentration(np.log10(m))), scale=0.1)
    rvir = np.random.rand(num_source_halos)

    mc_ncen = np.random.rand(num_source_halos) < mean_ncen(np.log10(m), logM_min, sigma_logM)
    mc_nsat = poisson.rvs(mean_nsat(m, logM_min))
    ngal = mc_ncen + mc_nsat
    halo_mass_array = np.repeat(m, ngal)
    halo_id_galaxies = np.repeat(np.arange(len(m)).astype(int), ngal)
    gal_id_array_galaxies = np.arange(len(halo_id_galaxies)).astype(int)
    host_x_galaxies = np.repeat(host_x, ngal)
    host_y_galaxies = np.repeat(host_y, ngal)
    host_z_galaxies = np.repeat(host_z, ngal)
    host_rvir_galaxies = np.repeat(rvir, ngal)
    host_conc_galaxies = np.repeat(host_conc, ngal)
    unique_ids, idx = np.unique(halo_id_galaxies, return_index=True)
    satellite = np.ones(len(halo_id_galaxies), dtype=bool)
    satellite[idx] = False

    galaxy_catalog = np.zeros(num_source_gals, dtype=dt_source_gals)
    galaxy_catalog['gal_id'] = gal_id_array_galaxies[:num_source_gals]
    galaxy_catalog['host_halo_mass'] = halo_mass_array[:num_source_gals]
    galaxy_catalog['host_halo_conc'] = host_conc_galaxies[:num_source_gals]
    galaxy_catalog['satellite'] = satellite[:num_source_gals]
    galaxy_catalog['host_halo_id'] = halo_id_galaxies[:num_source_gals]
    galaxy_catalog['host_halo_x'] = host_x_galaxies[:num_source_gals]
    galaxy_catalog['host_halo_y'] = host_y_galaxies[:num_source_gals]
    galaxy_catalog['host_halo_z'] = host_z_galaxies[:num_source_gals]
    galaxy_catalog['x'] = host_x_galaxies[:num_source_gals]
    galaxy_catalog['y'] = host_y_galaxies[:num_source_gals]
    galaxy_catalog['z'] = host_z_galaxies[:num_source_gals]
    galaxy_catalog['host_halo_rvir'] = host_rvir_galaxies[:num_source_gals]

    satmask = galaxy_catalog['satellite'] == True
    nsats = np.count_nonzero(satmask)
    with NumpyRNGContext(seed):
        dx = np.random.uniform(0, 1/3., nsats)*galaxy_catalog['host_halo_rvir'][satmask]
        dy = np.random.uniform(0, 1/3., nsats)*galaxy_catalog['host_halo_rvir'][satmask]
        dz = np.random.uniform(0, 1/3., nsats)*galaxy_catalog['host_halo_rvir'][satmask]

    galaxy_catalog['x'][satmask] += dx
    galaxy_catalog['y'][satmask] += dy
    galaxy_catalog['z'][satmask] += dz

    idx_ransort = np.random.choice(np.arange(num_source_gals), num_source_gals, replace=False)

    return galaxy_catalog[idx_ransort]


def mean_halo_concentration(logM):
    logM_table = [10, 15]
    conc_table = [15, 4.1]
    return np.interp(logM, logM_table, conc_table)


def mean_ncen(logM, logM_min, sigma_logM):
    return 0.5*(1.0 + erf((logM - logM_min) / sigma_logM))


def mean_nsat(m, logM_min):
    log_M0 = logM_min-0.3
    log_M1 = log_M0+2
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

