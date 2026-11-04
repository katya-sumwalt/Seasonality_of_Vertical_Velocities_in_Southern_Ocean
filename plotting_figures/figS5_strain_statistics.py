'''Figure S5: Figure 3 (all profiles) with w(15 m) = w_m (Ponte et al. 2013, eq. 5) for every run, and its sensitivity to
a depth-dependent strain u_x(x, d) = u_x(x, 0) F(d) (cases A, B, C of computing_vertical_velocities/compute_dw_strain_profile.py).
Colors: seasons; line styles: strain runs; shading: 95% CI of the F = 1 run.

w_stored = vertical_velocity - dw_wm - dw_case (dw in the panel-e convention, linear superposition).

usage: python -m plotting_figures.figS5_strain_statistics
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import config
from analysis.statistics import (load_season, load_dw, version_stats, rms_over_all_dims, mean_over_all_dims,
                                 climatology)
from computing_vertical_velocities.compute_dw_strain_profile import RUN_LABEL
from plotting_figures.plot_utils import COLORS, LABELS, depth_axis, panel_label, season_bands

SEASONS = config.SEASONS
RUNS = ['F=1', 'A', 'B', 'C']
RUN_STYLE = {'F=1': dict(lw=3.5), 'A': dict(lw=2.5, ls='--'), 'B': dict(lw=2.5, ls='-.'), 'C': dict(lw=2.5, ls=':')}
LAYERS = {'15-100 m': (15, 100), '200-500 m': (200, 500)}

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 22})


def season_stats(season):
    ds = load_season(season)
    w = ds.vertical_velocity - load_dw(season)
    out = {'strain_rms': rms_over_all_dims(ds.strain_neuro), 'strain_mean': mean_over_all_dims(ds.strain_neuro)}
    for run in RUNS:
        w_run = w if run == 'F=1' else w - load_dw(season, run)
        out[run] = version_stats(ds, w_run)
    return out


def run_handles(color='k'):
    return [Line2D([0], [0], color=color, label=RUN_LABEL[run], **RUN_STYLE[run]) for run in RUNS]


def run_legend(ax, loc, fontsize=14):
    ax.add_artist(ax.legend(handles=run_handles(), loc=loc, fontsize=fontsize))


def histograms(ax, stats, key):
    for s in ['summer', 'winter', 'spring', 'fall']:
        for run in RUNS:
            density, edges = stats[s][run][key]
            ax.stairs(density, edges, color=COLORS[s], label=LABELS[s] if run == 'F=1' else None, **RUN_STYLE[run])
    ax.set_yscale('log')
    ax.set_ylabel('PDF', fontsize=25)
    ax.grid()


def profiles(ax, stats, key, depth):
    for s in SEASONS:
        for run in RUNS:
            x, ci = stats[s][run][key]
            if run == 'F=1':
                ax.fill_betweenx(depth, x - ci, x + ci, alpha=0.25, color=COLORS[s])
            ax.plot(x, depth, color=COLORS[s], **RUN_STYLE[run])


def plot_figure_s5(stats, depth, out):
    fig = plt.figure(figsize=(25, 25))
    gs = GridSpec(3, 3, wspace=0.25, hspace=0.25)
    ax, ax2, ax3 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
    ax4, ax5, ax6 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
    ax7 = plt.subplot(gs[2, :])

    # a: PDF of w
    panel_label(ax, 'a')
    histograms(ax, stats, 'w_pdf')
    ax.set_xlim([-150, 150])
    ax.set_ylim([1e-5, 1])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel(r'w [m.d$^{-1}$]', fontsize=25)
    run_legend(ax, 'upper right', 12)
    ax.legend(loc='upper left', fontsize=15)

    # b: rms w
    panel_label(ax2, 'b')
    profiles(ax2, stats, 'w_rms', depth)
    ax2.set_xlim([0, 25])
    ax2.set_xticks([5, 10, 15, 20, 25])
    ax2.set_xlabel(r'rms w [m.d$^{-1}$]', fontsize=25)
    depth_axis(ax2)
    run_legend(ax2, 'upper right')

    # c: surface strain (the forcing that F(d) scales; the same for all runs)
    panel_label(ax3, 'c')
    order = ['winter', 'spring', 'summer', 'fall']
    rms = [stats[s]['strain_rms'] for s in order]
    mean = [stats[s]['strain_mean'] for s in order]
    for i, s in enumerate(order):
        ax3.errorbar(i, 1e5 * rms[i][0], yerr=1e5 * rms[i][1], color=COLORS[s], ms=7, elinewidth=3, fmt='D-', capsize=5)
        ax3.errorbar(i, 1e5 * mean[i][0], yerr=1e5 * mean[i][1], color=COLORS[s], ms=7, elinewidth=3, fmt='o-', capsize=5)
    ax3.plot(range(4), [1e5 * r[0] for r in rms], color='grey', alpha=0.5)
    ax3.plot(range(4), [1e5 * m[0] for m in mean], color='grey', alpha=0.5)
    ax3.set_xticks(range(4), order)
    ax3.set_ylabel(r'mean,rms $\sigma$ [s$^{-1}$]')
    ax3.text(-0.13, 0.992, r'$\times 10^{-5}$', transform=ax3.transAxes, fontsize=14, va='top')
    ax3.legend(handles=[Line2D([0], [0], marker='D', ms=5, lw=1, color='k', label=r'rms $\sigma$'),
                        Line2D([0], [0], marker='o', ms=5, lw=1, color='k', label=r'mean $\sigma$')], loc='lower left')
    ax3.set_title(r'surface strain, $\sigma$(d) = $\sigma$(0) F(d)', fontsize=18, loc='right')
    ax3.grid()

    # d: PDF of VHT
    panel_label(ax4, 'd')
    histograms(ax4, stats, 'vht_pdf')
    ax4.set_xlabel(r'VHT [W.m$^{-2}$]', fontsize=25)
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax4.xaxis.get_offset_text().set_visible(False)
    ax4.text(0.920, -0.05, r'$\times 10^{4}$', transform=ax4.transAxes, fontsize=14, va='bottom')
    ax4.set_ylim([10**-7.5, 10**-1.5])
    ax4.set_xlim([-1.4e4, 1.4e4])
    run_legend(ax4, 'upper right', 12)

    # e: rms VHT
    panel_label(ax5, 'e')
    profiles(ax5, stats, 'vht_rms', depth)
    ax5.set_xlabel(r'rms VHT [W.m$^{-2}$]', fontsize=25)
    depth_axis(ax5)
    run_legend(ax5, 'upper right')

    # f: mean VHT
    panel_label(ax6, 'f')
    profiles(ax6, stats, 'vht_mean', depth)
    ax6.set_xlabel(r'<VHT> [W.m$^{-2}$]', fontsize=25)
    depth_axis(ax6)
    run_legend(ax6, 'upper right')

    # g: climatological time series of <VHT>
    panel_label(ax7, 'g', 1.09)
    season_bands(ax7)
    for run in RUNS:
        for layer, color, label in [('ts_upper', 'grey', '15–100 m'), ('ts_deep', 'k', '200–500 m')]:
            mean = pd.concat([stats[s][run][layer][0] for s in SEASONS])
            ci = pd.concat([stats[s][run][layer][1] for s in SEASONS])
            clim = climatology({'mean': mean, 'ci': ci})
            ax7.plot(clim.index, clim['mean'], c=color, label=label if run == 'F=1' else None, **RUN_STYLE[run])
            if run == 'F=1':
                ax7.fill_between(clim.index, clim['mean'] - clim['ci'], clim['mean'] + clim['ci'], alpha=0.2, color=color)
    ax7.axhline(0, color='k', lw=0.8)
    ax7.set_xlim(pd.Timestamp('2000-06-01'), pd.Timestamp('2001-05-31'))
    ax7.set_ylabel(r'$\langle$VHT$\rangle$ [W.m$^{-2}$]')
    ax7.xaxis.set_major_locator(mdates.MonthLocator())
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax7.add_artist(ax7.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.27, 0.88)))
    ax7.legend(handles=run_handles(), frameon=False, loc='upper right', bbox_to_anchor=(1, 0.88), fontsize=16)

    fig.suptitle(r'w(15 m) = w$_m$ (Ponte et al. 2013) for all runs', y=0.92, fontsize=28)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


def summary(stats, depth):
    '''Layer means of <VHT> and rms w, ratio to F = 1, seasonal ordering; seasonal means of the Fig. 3g climatology.'''
    for key in ['vht_mean', 'w_rms']:
        print(f'\n{key}: layer mean (ratio to F = 1) | seasonal ordering')
        for name, (top, bottom) in LAYERS.items():
            sel = (depth >= top) & (depth <= bottom)
            for run in RUNS:
                vals = {s: np.nanmean(stats[s][run][key][0][sel]) for s in SEASONS}
                ref = {s: np.nanmean(stats[s]['F=1'][key][0][sel]) for s in SEASONS}
                order = ' > '.join(sorted(SEASONS, key=lambda s: -vals[s]))
                print(f'  {name:9s} {run:3s} ' + ', '.join(f'{s[:3]} {vals[s]:6.1f} ({vals[s] / ref[s]:.2f})'
                                                        for s in ['winter', 'spring', 'summer', 'fall']) + f' | {order}')
    print('\nFig. 3g: seasonal mean of the climatology [W/m2]')
    ranges = {'winter': ('2000-06-01', '2000-08-31'), 'spring': ('2000-09-01', '2000-11-30'),
              'summer': ('2000-12-01', '2001-02-28'), 'fall': ('2001-03-01', '2001-05-31')}
    for layer in ['ts_upper', 'ts_deep']:
        for run in RUNS:
            clim = climatology({'mean': pd.concat([stats[s][run][layer][0] for s in SEASONS])})['mean']
            print(f'  {layer:8s} {run:3s} ' + ', '.join(f'{s[:3]} {clim[a:b].mean():6.1f}' for s, (a, b) in ranges.items()))


if __name__ == '__main__':
    stats = {}
    for s in SEASONS:
        stats[s] = season_stats(s)
        print('done', s, flush=True)
    depth = np.arange(15, 501)
    summary(stats, depth)
    out = config.out_path('figS5_strain_statistics')
    plot_figure_s5(stats, depth, out)
    print('saved', out)
