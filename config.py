'''Paths to the input data and outputs. Defaults are data/ and figures/ inside the repository; each path can be
overridden with the environment variable given next to it.'''

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(var, default):
    return os.path.expanduser(os.environ.get(var, default))


DATA_DIR = _path('SES_DATA_DIR', os.path.join(ROOT, 'data'))
OUT_DIR = _path('SES_OUT_DIR', os.path.join(ROOT, 'figures'))

# MEOP-SMS seal profiles with the NeurOST fields along track (input of computing_vertical_velocities/compute_w_main.py)
RAW_SEAL_FILE = _path('SES_RAW_SEAL_FILE', os.path.join(DATA_DIR, 'updated_along_track_strain_raw_seal_data.nc'))
# all profiles of the study region, before the vertical velocity is computed (Fig. 1b profile counts)
ALL_PROFILES_FILE = _path('SES_ALL_PROFILES_FILE', os.path.join(DATA_DIR, 'seal_data.nc'))
# seasonal files with w(15 m) = 0 (output of the compute_w pipeline)
SEASONAL_DIR = _path('SES_SEASONAL_DIR', os.path.join(DATA_DIR, 'W_east_katya'))
SEASON_FILES = {'summer': 'seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc',
                'winter': 'seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc',
                'fall': 'seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc',
                'spring': 'seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc'}
SEASONS = ['summer', 'winter', 'fall', 'spring']
# corrections dw added to w: upper boundary condition w(15 m) = w_m (dw_<season>.nc) and depth-dependent strain
# cases A, B, C (dw_<case>_<season>.nc)
DW_WM_DIR = _path('SES_DW_WM_DIR', os.path.join(DATA_DIR, 'W_east_katya_wm'))
DW_STRAIN_DIR = _path('SES_DW_STRAIN_DIR', os.path.join(DATA_DIR, 'W_east_katya_strainF'))

# NeurOST daily SSH-SST files (https://doi.org/10.5067/NEURO-STV24) and the time-mean strain cached from them (Fig. 1a)
NEUROST_DIR = _path('SES_NEUROST_DIR', os.path.join(DATA_DIR, 'ssh_neuro'))
NEUROST_GLOB = os.path.join(NEUROST_DIR, 'NeurOST_SSH-SST_*.nc')
STRAIN_CACHE = _path('SES_STRAIN_CACHE', os.path.join(DATA_DIR, 'fig1_strain_mean.nc'))
# RTopo-2 30 arc-second bedrock topography (Fig. 1 bathymetry contours)
RTOPO_FILE = _path('SES_RTOPO_FILE', os.path.join(DATA_DIR, 'RTopo-2.0.1_30sec_bedrock_topography.nc'))
# mixed layer depth along the Figure 2 section (written by analysis/mld_comparison.py)
MLD_FILE = _path('SES_MLD_FILE', os.path.join(DATA_DIR, 'mld_holte_talley_section.nc'))

# Figure 2 section
SECTION = ('ft22-879-18', '2018-11-06', '2018-11-15')


def season_file(season):
    return os.path.join(SEASONAL_DIR, SEASON_FILES[season])


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
