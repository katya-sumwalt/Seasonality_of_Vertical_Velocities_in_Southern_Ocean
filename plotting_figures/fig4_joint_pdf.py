'''Figure 4: joint PDF of w' and CT' in winter (200-500 m), its contribution to <VHT>, and the seasonal decompositions
<w'CT'> = r rms(w') rms(CT') and <VHT> = strength + frequency (analysis/joint_pdf.py), with w(15 m) = w_m.

usage: python -m plotting_figures.fig4_joint_pdf [--bc wm|zero] [--case A|B|C]
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import config
from analysis.joint_pdf import SEASONS, PDF_LAYER, seasonal_decomposition, joint_pdf, parse_args
from plotting_figures.plot_utils import COLORS

SHORT = {'winter': 'Win', 'spring': 'Spr', 'summer': 'Sum', 'fall': 'Fall'}

plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 26,
                     "xtick.labelsize": 22, "ytick.labelsize": 22})


def plot_figure4(stats, pdf, wc, tc, contrib, quad, out):
    total = contrib.sum()
    fig = plt.figure(figsize=(30, 19))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.38, wspace=0.5)
    gs_bottom = gs[1, :].subgridspec(1, 4, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    img = ax.pcolormesh(wc, tc, pdf.T, norm=LogNorm(vmin=1e-5, vmax=pdf.max()), cmap='viridis', shading='nearest')
    ax.contour(wc, tc, pdf.T, levels=10.0 ** np.arange(-4, 0), colors='w', linewidths=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel(r"w' [m d$^{-1}$]")
    ax.set_ylabel("CT' [°C]")
    ax.set_title(f'a   winter, {PDF_LAYER}', loc='left', fontweight='bold')
    plt.colorbar(img, ax=ax, label=r"joint PDF [(m d$^{-1}$ °C)$^{-1}$]")

    ax = fig.add_subplot(gs[0, 1])
    vmax = np.abs(contrib).max()
    img = ax.pcolormesh(wc, tc, contrib.T, norm=TwoSlopeNorm(0, -vmax, vmax), cmap='RdBu_r', shading='nearest')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    for (k, sel), (x, y) in zip(quad.items(), [(0.97, 0.97), (0.03, 0.03), (0.97, 0.03), (0.03, 0.97)]):
        ax.text(x, y, f'{contrib[sel].sum():+.0f} W m$^{{-2}}$', transform=ax.transAxes, ha='right' if x > 0.5 else 'left',
                va='top' if y > 0.5 else 'bottom', fontsize=21, bbox=dict(facecolor='white', alpha=0.85, lw=0))
    ax.set_xlabel(r"w' [m d$^{-1}$]")
    ax.set_ylabel("CT' [°C]")
    ax.set_title(f'b   <VHT> = {total:.0f} W m$^{{-2}}$, by bin', loc='left', fontweight='bold')
    plt.colorbar(img, ax=ax, label=r"W m$^{-2}$ per bin")

    ax = fig.add_subplot(gs[0, 2])
    for s in SEASONS:
        d = stats[(PDF_LAYER, s)]
        ax.scatter(d['rms_t'], d['r'], s=20 * d['vht'] + 80, color=COLORS[s], edgecolor='k', zorder=3, alpha=0.85)
        above = s == 'fall'   # avoid overlapping the summer label
        ax.annotate(f"{s.capitalize()}\n{d['vht']:.0f} W m$^{{-2}}$", (d['rms_t'], d['r']), xytext=(0, 30 if above else -50),
                    textcoords='offset points', ha='center', va='bottom' if above else 'top', fontsize=20, color=COLORS[s])
    ax.set_xlabel("rms CT' [°C]")
    ax.set_ylabel(r"r(w', CT')")
    ax.set_title(f'c   {PDF_LAYER} (size ∝ <VHT>)', loc='left', fontweight='bold')
    ax.grid(lw=0.5)
    ax.set_xlim(-0.15, 2.2)
    ax.set_ylim(0, 0.17)

    for i, (key, label) in enumerate([('rms_t', "rms CT' [°C]"), ('rms_w', r"rms w' [m d$^{-1}$]"), ('r', "r(w', CT')")]):
        ax = fig.add_subplot(gs_bottom[i])
        x = np.arange(len(SEASONS))
        for j, (name, hatch) in enumerate([('15-100 m', '//'), ('200-500 m', '')]):
            ax.bar(x + (j - 0.5) * 0.38, [stats[(name, s)][key] for s in SEASONS], width=0.38,
                   color=[COLORS[s] for s in SEASONS], hatch=hatch, edgecolor='k', alpha=0.9 if j else 0.5,
                   label=name)
        ax.set_xticks(x, [SHORT[s] for s in SEASONS])
        ax.set_ylabel(label)
        ax.set_title('def'[i], loc='left', fontweight='bold')
        ax.grid(axis='y', lw=0.5)
        if i == 0:
            ax.legend(fontsize=20, loc='upper right')

    # g: frequency vs strength split of <VHT>
    ax = fig.add_subplot(gs_bottom[3])
    x = np.arange(len(SEASONS))
    for j, (key, hatch, label) in enumerate([('strength', '', 'strength'), ('frequency', '//', 'frequency')]):
        ax.bar(x + (j - 0.5) * 0.38, [stats[(PDF_LAYER, s)][key] for s in SEASONS], width=0.38,
               color=[COLORS[s] for s in SEASONS], hatch=hatch, edgecolor='k', alpha=0.9, label=label)
    total = [stats[(PDF_LAYER, s)]['vht'] for s in SEASONS]
    ax.scatter(x, total, marker='_', s=900, lw=4, color='k', zorder=3, label='total <VHT>')
    for xi, s in zip(x, SEASONS):
        d = stats[(PDF_LAYER, s)]
        ax.text(xi, max(d['vht'], d['strength']) + 6, f"P$_+$={d['P+']:.2f}", ha='center', va='bottom', fontsize=17)
    ax.set_xticks(x, [SHORT[s] for s in SEASONS])
    ax.set_ylabel(r'<VHT> [W m$^{-2}$]')
    ax.set_ylim(0, 1.2 * max(total))
    ax.set_title(f'g   {PDF_LAYER}', loc='left', fontweight='bold')
    ax.grid(axis='y', lw=0.5)
    ax.legend(fontsize=17, loc='upper right')

    fig.savefig(out + '.png', bbox_inches='tight', dpi=120)
    fig.savefig(out + '.pdf', bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    args = parse_args(__doc__.splitlines()[0])
    out = config.out_path('fig4_joint_pdf' + ('' if args.bc == 'wm' else '_w0') + (f'_strain{args.case}' if args.case else ''))
    stats, w_win, t_win = seasonal_decomposition(args.bc, args.case)
    pdf, wc, tc, contrib, quad = joint_pdf(w_win, t_win, stats[(PDF_LAYER, 'winter')]['vht'])
    plot_figure4(stats, pdf, wc, tc, contrib, quad, out)
