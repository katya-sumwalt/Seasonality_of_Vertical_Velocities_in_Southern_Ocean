'''Figures S6 and S7: Figure 3 statistics with summer, winter and spring repeatedly downsampled to the fall sampling and
block-bootstrapped (analysis/bootstrap.py, which also prints and saves the Table S2 numbers).

Figure S6: number of observations vs depth for every resample. Figure S7: Figure 3 with the median over resamples (thick
lines), 2.5-97.5 percentiles (shading) and all profiles without resampling (thin faded lines), and the bootstrap
distributions of the layer-mean <VHT> and of its strength/frequency split.

usage: python -m plotting_figures.figS6_S7_bootstrap [--bc wm|zero] [--case A|B|C] [--n-iter N]
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import config
from analysis.bootstrap import SEASONS, LAYERS, DEPTH, W_BINS, VHT_BINS, run, suffix, parse_args
from plotting_figures.plot_utils import COLORS, LABELS, depth_axis, panel_label, season_bands

plt.rcParams.update({"font.family": "serif", "font.size": 22})


def band(ax, boot, full, key, s, ls='-'):
    lo, med, hi = np.nanpercentile(boot[s][key], [2.5, 50, 97.5], axis=0)
    ax.fill_betweenx(DEPTH, lo, hi, color=COLORS[s], alpha=0.3, lw=0)
    ax.plot(med, DEPTH, color=COLORS[s], lw=3, ls=ls)
    ax.plot(full[s][key], DEPTH, color=COLORS[s], lw=1.5, ls=ls, alpha=0.5)


def pdf_band(ax, boot, full, key, bins):
    for s in ['summer', 'winter', 'spring', 'fall']:
        lo, med, hi = np.nanpercentile(boot[s][key], [2.5, 50, 97.5], axis=0)
        x = 0.5 * (bins[1:] + bins[:-1])
        ax.fill_between(x, lo, hi, color=COLORS[s], alpha=0.3, lw=0, step='mid')
        ax.stairs(med, bins, color=COLORS[s], lw=3, label=LABELS[s])
        ax.stairs(full[s][key], bins, color=COLORS[s], lw=1.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('PDF', fontsize=25)
    ax.grid()


def style_handles(n_iter):
    return [Line2D([0], [0], color='k', lw=3, label=f'median of {n_iter} resamples (95% band)'),
            Line2D([0], [0], color='k', lw=1.5, alpha=0.5, label='all profiles')]


def plot_figure_s7(boot, full, clim, out):
    n_iter = len(boot['fall']['w_rms'])
    fig = plt.figure(figsize=(25, 34))
    gs = GridSpec(4, 3, wspace=0.25, hspace=0.3)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    panel_label(ax[0], 'a')
    pdf_band(ax[0], boot, full, 'w_pdf', W_BINS)
    ax[0].set_xlim([-150, 150])
    ax[0].set_ylim([1e-5, 1])
    ax[0].set_xlabel(r'w [m.d$^{-1}$]', fontsize=25)
    ax[0].legend(loc='upper left', fontsize=16)

    for a_, key, letter, label in [(ax[1], 'w_rms', 'b', r'rms w [m.d$^{-1}$]'),
                                   (ax[4], 'vht_rms', 'e', r'rms VHT [W.m$^{-2}$]'),
                                   (ax[5], 'vht_mean', 'f', r'<VHT> [W.m$^{-2}$]')]:
        panel_label(a_, letter)
        for s in SEASONS:
            band(a_, boot, full, key, s)
        a_.set_xlabel(label, fontsize=25)
        depth_axis(a_)
        a_.legend(handles=style_handles(n_iter), loc='lower right', fontsize=14)
    ax[1].set_xlim([0, 25])

    panel_label(ax[2], 'c')
    order = ['winter', 'spring', 'summer', 'fall']
    for key, marker in [('strain_rms', 'D'), ('strain_mean', 'o')]:
        lo, med, hi = (np.array([1e5 * np.percentile(boot[s][key], q) for s in order]) for q in (2.5, 50, 97.5))
        for i, s in enumerate(order):
            ax[2].errorbar(i, med[i], yerr=[[med[i] - lo[i]], [hi[i] - med[i]]], color=COLORS[s], fmt=marker, ms=7,
                           elinewidth=3, capsize=5)
        ax[2].plot(range(4), med, color='grey', alpha=0.5)
    ax[2].set_xticks(range(4), order)
    ax[2].set_ylabel(r'mean,rms $\sigma$ [$\times 10^{-5}$ s$^{-1}$]')
    ax[2].legend(handles=[Line2D([0], [0], marker='D', ms=5, lw=1, color='k', label=r'rms $\sigma$'),
                          Line2D([0], [0], marker='o', ms=5, lw=1, color='k', label=r'mean $\sigma$')], loc='lower left')
    ax[2].grid()

    panel_label(ax[3], 'd')
    pdf_band(ax[3], boot, full, 'vht_pdf', VHT_BINS)
    ax[3].set_xlim([-1.4e4, 1.4e4])
    ax[3].set_ylim([10 ** -7.5, 10 ** -1.5])
    ax[3].set_xlabel(r'VHT [W.m$^{-2}$]', fontsize=25)
    ax[3].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # g: climatological time series of the layer <VHT>
    a_ = fig.add_subplot(gs[2, :])
    panel_label(a_, 'g', 1.09)
    full_clim, boot_clim = clim
    season_bands(a_)
    for name, color in [('15-100 m', 'grey'), ('200-500 m', 'k')]:
        lo, med, hi = np.nanpercentile(boot_clim[name], [2.5, 50, 97.5], axis=0)
        a_.fill_between(full_clim.index, lo, hi, color=color, alpha=0.26, lw=0)
        a_.plot(full_clim.index, med, c=color, lw=3, label=name)
        a_.plot(full_clim.index, full_clim[name], c=color, lw=1.5, alpha=0.5)
    a_.axhline(0, color='k', lw=0.8)
    a_.set_xlim(pd.Timestamp('2000-06-01'), pd.Timestamp('2001-05-31'))
    a_.set_ylabel(r'$\langle$VHT$\rangle$ [W.m$^{-2}$]')
    a_.xaxis.set_major_locator(mdates.MonthLocator())
    a_.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    a_.add_artist(a_.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.27, 0.88)))
    a_.legend(handles=style_handles(n_iter), frameon=False, loc='upper right', bbox_to_anchor=(1, 0.88), fontsize=16)

    # h, i: bootstrap distributions of the layer-mean <VHT>
    for j, name in enumerate(LAYERS):
        a_ = fig.add_subplot(gs[3, j])
        panel_label(a_, 'hi'[j])
        key = f'layer {name}'
        x = np.concatenate([boot[s][key] for s in SEASONS])
        bins = np.linspace(x.min(), x.max(), 60)
        for s in SEASONS:
            a_.hist(boot[s][key], bins, color=COLORS[s], alpha=0.6, label=LABELS[s])
            a_.axvline(full[s][key], color=COLORS[s], lw=2, ls='--')
        a_.set_xlabel(rf'<VHT>, {name} [W.m$^{{-2}}$]', fontsize=25)
        a_.set_ylabel('resamples')
        a_.grid()
        if j == 0:
            a_.legend(fontsize=16)
    a_ = fig.add_subplot(gs[3, 2])
    panel_label(a_, 'j')
    name = '200-500 m'
    for i, s in enumerate(['winter', 'spring', 'summer', 'fall']):
        V_p, V_m, P = boot[s][f'V+ {name}'], boot[s][f'V- {name}'], boot[s][f'P+ {name}']
        strength, frequency = 0.5 * (V_p + V_m), 0.5 * (2 * P - 1) * (V_p - V_m)
        for k, (x, hatch) in enumerate([(strength, ''), (frequency, '//')]):
            lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
            a_.bar(i + (k - 0.5) * 0.38, med, 0.38, yerr=[[med - lo], [hi - med]], color=COLORS[s], hatch=hatch,
                   edgecolor='k', capsize=5, label=['strength', 'frequency'][k] if i == 0 else None)
    a_.set_xticks(range(4), ['winter', 'spring', 'summer', 'fall'])
    a_.set_ylabel(r'contribution to <VHT> [W.m$^{-2}$]', fontsize=20)
    a_.set_title(name, fontsize=20)
    a_.legend(fontsize=16)
    a_.grid(axis='y')

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


def plot_figure_s6(boot, full, out):
    '''Fig. 1d equivalent: number of observations vs depth for every resample.'''
    n_iter = len(boot['fall']['w_rms'])
    fall = full['fall']['counts']
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    for s in ['fall', 'summer', 'spring', 'winter']:   # fall (bootstrap only, not downsampled) drawn underneath
        c = boot[s]['counts']
        lo, med, hi = np.percentile(c, [0, 50, 100], axis=0)
        for ax, ref in [(axes[0], 0), (axes[1], fall)]:
            if s == 'fall':
                ax.fill_betweenx(DEPTH, lo - ref, hi - ref, facecolor='none', edgecolor=COLORS[s], hatch='..', lw=0.8,
                                 label='fall bootstrap spread (not downsampled)' if ref is fall else None)
            else:
                ax.plot((c - ref)[::10].T, DEPTH, color=COLORS[s], lw=0.3, alpha=0.3)   # every 10th resample
                ax.fill_betweenx(DEPTH, lo - ref, hi - ref, color=COLORS[s], alpha=0.25, lw=0)
            ax.plot(med - ref, DEPTH, color=COLORS[s], lw=3 if s != 'fall' else 1.5,
                    label=LABELS[s] if ref is not fall else None)
        axes[0].plot(full[s]['counts'], DEPTH, color=COLORS[s], lw=1.5, ls=':')
    axes[0].plot(fall, DEPTH, color='k', lw=1.5, ls='--', label='fall, all profiles (target)')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of observations')
    axes[0].set_title('dotted: all profiles', loc='right', fontsize=16)
    axes[1].axvline(0, color='k', lw=1.5, ls='--')
    axes[1].set_xlabel('Observations minus fall (all profiles)')
    axes[1].set_title(f'thick: median, shading: min-max of {n_iter} resamples', loc='right', fontsize=16)
    axes[1].legend(fontsize=14, loc='lower left')
    depth_axis(axes[0])
    depth_axis(axes[1], label=False)
    axes[0].legend(fontsize=15, loc='lower left')
    panel_label(axes[0], 'a', 1.1)
    panel_label(axes[1], 'b', 1.1)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args(__doc__.splitlines()[0])
    boot, full, clim = run(args.bc, args.case, args.n_iter)
    sfx = suffix(args.bc, args.case)
    plot_figure_s6(boot, full, config.out_path('figS6_obs_vs_depth' + sfx))
    plot_figure_s7(boot, full, clim, config.out_path('figS7_bootstrap_statistics' + sfx))
    print('saved figures to', config.OUT_DIR)
