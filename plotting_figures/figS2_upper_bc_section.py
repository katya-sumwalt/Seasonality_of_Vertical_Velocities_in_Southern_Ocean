'''Figure S2: w along the Figure 2 section with w(15 m) = 0, with w(15 m) = w_m (Ponte et al. 2013, eq. 5) and their
difference, solved directly with the omega equation, with the de Boyer Montegut et al. (2004) MLD.

usage: python -m plotting_figures.figS2_upper_bc_section
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
from analysis.statistics import DAY
from computing_vertical_velocities.compute_w_surface_bc import compute_w, w_mixing
from plotting_figures.plot_utils import load_section

NAME, START, END = config.SECTION

plt.rcParams.update({"font.family": "serif", "font.size": 18})


def section(ax, x_km, depth, mld, data, letter, title, vmax):
    img = ax.pcolormesh(x_km, depth, data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
    ax.plot(x_km, mld, 'k', lw=1.5)
    ax.set_ylim(500, 15)
    ax.set_ylabel('Depth [m]')
    ax.set_title(title, loc='left')
    ax.text(-0.04, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=22)
    cax = make_axes_locatable(ax).append_axes('right', size='1%', pad=0.2)
    plt.colorbar(img, cax=cax, label=r'w [m d$^{-1}$]')


if __name__ == '__main__':
    out = config.out_path('figS2_upper_bc_section')
    sec, mld = load_section()
    x_km, depth, mask = sec.distance.values / 1000, -sec.depth.values, sec.maskprof

    w0 = compute_w(sec).transpose('depth', 'distance') * DAY
    w1 = compute_w(sec, w_top=w_mixing(sec).values).transpose('depth', 'distance') * DAY

    fig, axes = plt.subplots(3, 1, figsize=(20, 17), sharex=True)
    section(axes[0], x_km, depth, mld, w0 * mask, 'a', f'{NAME} | {START} → {END} | w, w(15 m) = 0', 32)
    section(axes[1], x_km, depth, mld, w1 * mask, 'b', r'w, w(15 m) = w$_m$', 32)
    section(axes[2], x_km, depth, mld, (w1 - w0) * mask, 'c', 'difference (b) − (a)', 10)
    axes[2].set_xlabel('Along-track distance [km]')

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=120)
    print('saved', out)
