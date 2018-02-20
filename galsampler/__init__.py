# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.
    from .host_halo_binning import halo_bin_indices, matching_bin_dictionary
    from .source_halo_selection import source_halo_index_selection
    from .source_galaxy_selection import source_galaxy_selection_indices
    from .matched_halo_selection_1d import matched_value_selection_indices
