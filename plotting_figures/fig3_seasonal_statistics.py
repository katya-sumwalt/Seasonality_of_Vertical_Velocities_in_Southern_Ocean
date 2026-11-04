'''Figure 3 (seasonal statistics of w and VHT), Figure S3 (difference between w(15 m) = w_m and w(15 m) = 0) and
Figure S9 (conditional means of VHT and w'b'), all profiles, w(15 m) = w_m (Ponte et al. 2013, eq. 5).

Figure S3: (a) rms w, (b) rms VHT, (c) <VHT> differences [w_m - (w = 0)] vs depth, shading: 95% CI of the w = 0 estimate
centred on zero (differences inside it are smaller than the sampling uncertainty); (d) difference of the Fig. 3g
climatologies of the layer-mean <VHT>.

usage: python -m plotting_figures.fig3_seasonal_statistics
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
from analysis.statistics import season_stats, climatology
from plotting_figures.plot_utils import (COLORS, LABELS, SEASON_RANGES, DEPTH, profile, depth_axis, panel_label,
                                         histograms, season_bands)

SEASONS = config.SEASONS
LAYERS = {'15-100 m': (15, 100), '200-500 m': (200, 500)}

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 22})


###### FIGURE 3 ##################################################################################
def plot_figure3(stats, depth, out):
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
    ax.legend(loc='upper left', fontsize=16)

    # b: rms w
    panel_label(ax2, 'b')
    for s in SEASONS:
        profile(ax2, *stats[s]['new']['w_rms'], depth, COLORS[s])
    ax2.set_xlim([0, 25])
    ax2.set_xticks([5, 10, 15, 20, 25])
    ax2.set_xlabel(r'rms w [m.d$^{-1}$]', fontsize=25)
    depth_axis(ax2)

    # c: strain (independent of w)
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

    # e: rms VHT
    panel_label(ax5, 'e')
    for s in SEASONS:
        profile(ax5, *stats[s]['new']['vht_rms'], depth, COLORS[s])
    ax5.set_xlabel(r'rms VHT [W.m$^{-2}$]', fontsize=25)
    depth_axis(ax5)

    # f: mean VHT
    panel_label(ax6, 'f')
    for s in SEASONS:
        profile(ax6, *stats[s]['new']['vht_mean'], depth, COLORS[s])
    ax6.set_xlabel(r'<VHT> [W.m$^{-2}$]', fontsize=25)
    depth_axis(ax6)

    # g: climatological time series of <VHT>
    panel_label(ax7, 'g', 1.09)
    season_bands(ax7)
    for layer, color, label in [('ts_upper', 'grey', '15–100 m'), ('ts_deep', 'k', '200–500 m')]:
        mean = pd.concat([stats[s]['new'][layer][0] for s in SEASONS])
        ci = pd.concat([stats[s]['new'][layer][1] for s in SEASONS])
        clim = climatology({'mean': mean, 'ci': ci})
        ax7.plot(clim.index, clim['mean'], c=color, label=label, lw=3, alpha=1)
        ax7.fill_between(clim.index, clim['mean'] - clim['ci'], clim['mean'] + clim['ci'], alpha=0.26, color=color)
    ax7.axhline(0, color='k', lw=0.8)
    ax7.set_xlim(pd.Timestamp('2000-06-01'), pd.Timestamp('2001-05-31'))
    ax7.set_ylabel(r'$\langle$VHT$\rangle$ [W.m$^{-2}$]')
    ax7.xaxis.set_major_locator(mdates.MonthLocator())
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax7.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.27, 0.88))

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


###### FIGURE S9 #################################################################################
def counts_box(ax, stats, key):
    box = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8)
    ax.text(0.03, 0.25, 'n+/n-', transform=ax.transAxes, va='bottom', fontsize=15, bbox=box, color='k')
    for y, s in zip([0.20, 0.15, 0.10, 0.05], ['summer', 'winter', 'fall', 'spring']):
        c = stats[s]['new'][key]
        ax.text(0.03, y, f"{c['n_pos']:,}/{c['n_neg']:,}", transform=ax.transAxes, va='bottom', fontsize=15,
                bbox=box, color=COLORS[s])


def conditional_panel(ax, stats, key, depth):
    for s in ['summer', 'winter', 'spring', 'fall']:
        c = stats[s]['new'][key]
        profile(ax, *c['pos'], depth, COLORS[s])
        profile(ax, *c['neg'], depth, COLORS[s], ls='--')
    ax.axvline(0, color='grey', lw=3)
    counts_box(ax, stats, key)


def plot_figure_s9(stats, depth, out):
    fig = plt.figure(figsize=(30, 10))
    plt.rcParams['mathtext.fontset'] = 'stix'

    ax = plt.subplot(131)
    panel_label(ax, 'a', 1.06)
    conditional_panel(ax, stats, 'vht_cond', depth)
    ax.set_xlim([-740, 740])
    ax.set_xlabel(r'Conditional <VHT> [W.m$^{-2}$]', fontsize=28)
    depth_axis(ax)
    # legend also added as an artist (drawn twice), as in the published figure
    ax.add_artist(ax.legend(handles=[Line2D([0], [0], color=COLORS[s], lw=3, label=LABELS[s])
                                     for s in ['summer', 'winter', 'spring', 'fall']], loc='upper left', fontsize=18))

    ax2 = plt.subplot(132)
    panel_label(ax2, 'b', 1.06)
    conditional_panel(ax2, stats, 'wb_cond', depth)
    ax2.set_xlim([-2.5e-7, 2.5e-7])
    ax2.set_xlabel(r"Conditional <w'b'> [m$^{2}$.s$^{-3}$]", fontsize=28)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax2.xaxis.get_offset_text().set_visible(False)
    ax2.text(0.9, -0.065, r'$\times 10^{-7}$', transform=ax2.transAxes, fontsize=20, va='top')
    depth_axis(ax2)
    ax2.legend(handles=[Line2D([0], [0], color='k', lw=3, label='upward'),
                        Line2D([0], [0], color='k', lw=3, ls='--', label='downward')], loc='upper right', fontsize=18)

    ax3 = plt.subplot(133)
    panel_label(ax3, 'c', 1.06)
    for s in SEASONS:
        profile(ax3, *stats[s]['new']['wb_mean'], depth, COLORS[s])
    ax3.set_xlabel(r"<w'b'> [m$^{2}$.s$^{-3}$]", fontsize=28)
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax3.xaxis.get_offset_text().set_visible(False)
    ax3.text(0.9, -0.065, r'$\times 10^{-7}$', transform=ax3.transAxes, fontsize=20, va='top')
    depth_axis(ax3)

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


###### FIGURE S3 #################################################################################
def season_climatology(stats, ver, layer):
    return climatology({'mean': pd.concat([stats[s][ver][layer][0] for s in SEASONS])})['mean']


def plot_figure_s3(stats, out):
    fig = plt.figure(figsize=(25, 18))
    gs = GridSpec(2, 3, wspace=0.25, hspace=0.3, height_ratios=[1.5, 1])
    for i, (key, label) in enumerate([('w_rms', r'$\Delta$ rms w [m.d$^{-1}$]'),
                                      ('vht_rms', r'$\Delta$ rms VHT [W.m$^{-2}$]'),
                                      ('vht_mean', r'$\Delta$ <VHT> [W.m$^{-2}$]')]):
        ax = plt.subplot(gs[0, i])
        panel_label(ax, 'abc'[i])
        for s in SEASONS:
            new, (old, ci) = stats[s]['new'][key][0], stats[s]['old'][key]
            ax.fill_betweenx(DEPTH, -ci, ci, color=COLORS[s], alpha=0.15, lw=0)
            ax.plot(new - old, DEPTH, color=COLORS[s], lw=3, label=LABELS[s])
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel(label, fontsize=25)
        depth_axis(ax, label=i == 0)
        if i == 0:
            ax.legend(loc='lower left', fontsize=16)
    fig.text(0.5, 0.935, r'w(15 m) = w$_m$ minus w(15 m) = 0; shading: 95% CI of the w(15 m) = 0 estimate',
             ha='center', fontsize=20)

    ax = plt.subplot(gs[1, :])
    panel_label(ax, 'd', 1.12)
    season_bands(ax, 0.95)
    for layer, color, label in [('ts_upper', 'grey', '15–100 m'), ('ts_deep', 'k', '200–500 m')]:
        diff = season_climatology(stats, 'new', layer) - season_climatology(stats, 'old', layer)
        ax.plot(diff.index, diff, color=color, lw=3, label=label)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(pd.Timestamp('2000-06-01'), pd.Timestamp('2001-05-31'))
    ax.set_ylabel(r'$\Delta\langle$VHT$\rangle$ [W.m$^{-2}$]')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.27, 0.85))
    ax.grid(lw=0.5)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


def summary(stats):
    for key in ['w_rms', 'vht_rms', 'vht_mean']:
        print(f'\n{key}: layer mean, w=0 -> w_m (ratio); fraction of depths where |diff| < 95% CI of w=0')
        for name, (top, bottom) in LAYERS.items():
            sel = (DEPTH >= top) & (DEPTH <= bottom)
            for s in ['winter', 'spring', 'summer', 'fall']:
                new, (old, ci) = stats[s]['new'][key][0], stats[s]['old'][key]
                a, b = np.nanmean(old[sel]), np.nanmean(new[sel])
                inside = np.mean(np.abs(new - old)[sel] < ci[sel])
                print(f'  {name:9s} {s:7s} {a:7.1f} -> {b:7.1f} ({b / a:.2f}); inside CI {100 * inside:3.0f}%')
    print('\nFig. 3g seasonal mean of the climatology, w=0 -> w_m [W/m2]')
    for layer in ['ts_upper', 'ts_deep']:
        old, new = season_climatology(stats, 'old', layer), season_climatology(stats, 'new', layer)
        print(f'  {layer:8s} ' + ', '.join(f'{s[:3]} {old[a:b].mean():6.1f} -> {new[a:b].mean():6.1f}'
                                           for s, (a, b) in SEASON_RANGES.items()))


if __name__ == '__main__':
    stats = {}
    for s in SEASONS:
        stats[s] = season_stats(s)
        print('done', s, flush=True)
    summary(stats)
    plot_figure_s3(stats, config.out_path('figS3_upper_bc_difference'))
    for s in SEASONS:
        for key in ['vht_cond', 'wb_cond']:
            c = stats[s]['new'][key]
            print(f"{s:7s} {key}: n+/n- {c['n_pos']:,}/{c['n_neg']:,} (upward {c['n_pos'] / (c['n_pos'] + c['n_neg']):.3f})")
    plot_figure3(stats, DEPTH, config.out_path('fig3_seasonal_statistics'))
    plot_figure_s9(stats, DEPTH, config.out_path('figS9_conditional_means'))   # sets mathtext.fontset = 'stix'
    print('saved to', config.OUT_DIR)
