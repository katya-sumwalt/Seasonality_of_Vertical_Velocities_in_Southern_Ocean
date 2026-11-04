'''Plotting helpers shared by the figure scripts: season colours, depth-profile panels and section panels.'''

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config

COLORS = {'summer': 'tab:blue', 'winter': 'tab:green', 'fall': 'tab:orange', 'spring': 'tab:red'}
LABELS = {'summer': 'Summer [DJF]', 'winter': 'Winter [JJA]', 'fall': 'Fall [MAM]', 'spring': 'Spring [SON]'}
SEASON_RANGES = {'winter': ('2000-06-01', '2000-09-01'), 'spring': ('2000-09-01', '2000-12-01'),
                 'summer': ('2000-12-01', '2001-03-01'), 'fall': ('2001-03-01', '2001-06-01')}
DEPTH = np.arange(15, 501)
PAD = 0.2     # colorbar padding of the section panels


###### DEPTH PROFILES ############################################################################
def profile(ax, x, ci, depth, color, ls='-'):
    ax.fill_betweenx(depth, x - ci, x + ci, alpha=0.3, color=color)
    ax.plot(x, depth, color=color, ls=ls, lw=3, alpha=1)


def depth_axis(ax, label=True):
    ax.set_ylim([500, 15])
    ax.set_yticks([100, 200, 300, 400, 500])
    if label:
        ax.set_ylabel('Depth [m]', fontsize=25)
    ax.grid()


def panel_label(ax, letter, y=1.08):
    ax.text(0.0, y, letter, transform=ax.transAxes, fontsize=27, fontweight='bold', va='top')


def histograms(ax, stats, key):
    for s in ['summer', 'winter', 'spring', 'fall']:
        density, edges = stats[s]['new'][key]
        ax.stairs(density, edges, color=COLORS[s], label=LABELS[s], lw=3, alpha=1)
    ax.set_yscale('log')
    ax.set_ylabel('PDF', fontsize=25)
    ax.grid()


def season_bands(ax, y=0.97):
    '''Shaded seasons on a Jun-May climatology axis.'''
    for s, (start, end) in SEASON_RANGES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(start, end, color=COLORS[s], alpha=0.12, zorder=0)
        ax.text(start + (end - start) / 2, y, s.capitalize(), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=22, fontweight='bold', color=COLORS[s], alpha=0.9)


###### SECTIONS (Figure 2 layout) ################################################################
def section_label(ax, letter):
    ax.text(0.0, 1.1, letter, transform=ax.transAxes, fontsize=26, fontweight="bold", va="top")


def add_colorbar(ax, img, label, sci=False):
    cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=PAD)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(label)
    if sci:
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
    return cbar


def section_panel(ax, data, mld, letter, cmap, vmin, vmax, label, sci=False, crop_top=True):
    '''Depth-section panel with mixed-layer depth overlay and colorbar.'''
    section_label(ax, letter)
    img = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(mld - 15, lw=2, color='k')   # depth index 0 is 15 m
    if crop_top:
        ax.set_ylim((15, 486))
    ax.invert_yaxis()
    ax.set_ylabel('Depth [m]')
    ax.set_yticks([85, 185, 285, 385, 485])
    ax.set_yticklabels([100, 200, 300, 400, 500])
    ax.tick_params(axis='x', labelbottom=False)
    add_colorbar(ax, img, label, sci)


def load_section():
    '''Figure 2 section on a 1 km distance grid (as in compute_w_main.py), and its de Boyer Montegut MLD [m].'''
    name, start, end = config.SECTION
    ds = xr.open_dataset(config.season_file('spring'))
    sec = ds.where(ds.name == name, drop=True).swap_dims({'distance': 'time'}).sortby('time').sel(time=slice(start, end))
    sec = sec.swap_dims({'time': 'distance'}).assign_coords(distance=np.arange(sec.time.size) * 1000.).load()
    mld_file = xr.open_dataset(config.MLD_FILE)
    assert np.array_equal(mld_file.time.values, sec.time.values), 'MLD and section profiles differ'
    return sec, mld_file.mld_dbm.values
