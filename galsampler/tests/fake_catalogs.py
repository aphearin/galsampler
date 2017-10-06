"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import erf

from .numpy_random_context import NumpyRNGContext


default_dt_source_gals = np.dtype(
        [('gal_id', 'i8'), ('mass', 'f4'), ('conc', 'f4'), ('satellite', bool),
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('host_x', 'f4'), ('host_y', 'f4'), ('host_z', 'f4'),
        ('host_halo_id', 'i8')])

default_dt_target_halos = np.dtype(
        [('mvir', 'f4'), ('nfw_conc', 'f4'),
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('halo_id', 'i8')])


def fake_source_galaxy_catalog(num_source_gals=int(1e3),
        dt_source_gals=default_dt_source_gals, seed=None, Lbox=250.):
    """
    """

    with NumpyRNGContext(seed):
        x = np.random.rand(num_source_gals)


def mean_ncen(logM, logM_min=12, sigma_logM=0.2):
    return 0.5*(1.0 + erf((logM - logM_min) / sigma_logM))


def mean_nsat(m, M0=10.**12):
    M1 = 20.*M0
    return (m - M0)/M1


