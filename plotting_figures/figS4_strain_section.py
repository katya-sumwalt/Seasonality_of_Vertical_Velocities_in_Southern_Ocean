'''Figure S4: w along the Figure 2 section for a depth-independent strain (F = 1) and the depth-dependent strains A, B, C
(u_x(x, d) = u_x(x, 0) F(d), computing_vertical_velocities/compute_dw_strain_profile.py), solved directly on the section,
with the de Boyer Montegut et al. (2004) MLD.

With w(15 m) = w_m the linearity used for the seasonal statistics, w(F, w_m) = w(F, 0) + w(unforced, w_m), is checked.

usage: python -m plotting_figures.figS4_strain_section [--bc wm|zero]
'''

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config
from analysis.statistics import DAY
from computing_vertical_velocities.compute_w_surface_bc import compute_w, w_mixing
from computing_vertical_velocities.compute_dw_strain_profile import CASES, RUN_LABEL, strain_profile
from plotting_figures.plot_utils import load_section

NAME, START, END = config.SECTION
RUN_STYLE = {'F=1': dict(color='k', lw=3.5),
             'A': dict(color='#7b3294', lw=2.5, ls='--'),
             'B': dict(color='#1b7837', lw=2.5, ls='-.'),
             'C': dict(color='#e08214', lw=2.5, ls=':')}

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 18})


def strained(sec, run):
    F = xr.DataArray(strain_profile(-sec.depth.values, *CASES[run]), dims='depth', coords={'depth': sec.depth})
    return sec.assign(strain_F=sec.strain_neuro * F)


def section(ax, x_km, depth, mld, data, title, vmax, letter):
    img = ax.pcolormesh(x_km, depth, data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
    ax.plot(x_km, mld, 'k', lw=1.5)
    ax.set_ylim(500, 15)
    ax.set_yticks([100, 200, 300, 400, 500])
    ax.set_ylabel('Depth [m]')
    ax.set_title(title, loc='left')
    ax.text(-0.05, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=22)
    cax = make_axes_locatable(ax).append_axes('right', size='1.5%', pad=0.2)
    plt.colorbar(img, cax=cax, label=r'w [m d$^{-1}$]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure S4')
    parser.add_argument('--bc', choices=['wm', 'zero'], default='wm', help='upper boundary condition w(15 m)')
    bc = parser.parse_args().bc
    out = config.out_path('figS4_strain_section' + ('' if bc == 'wm' else '_w0'))

    sec, mld = load_section()
    x_km, depth, mask = sec.distance.values / 1000, -sec.depth.values, sec.maskprof

    w_top = w_mixing(sec) if bc == 'wm' else 0.0
    w = {'F=1': compute_w(sec, w_top=w_top)}
    for run in ['A', 'B', 'C']:
        w[run] = compute_w(strained(sec, run), w_top=w_top, strain_variable='strain_F')
    if bc == 'wm':
        run = 'A'
        superposed = (compute_w(strained(sec, run), strain_variable='strain_F')
                      + compute_w(sec.assign(bx=sec.bx * 0), w_top=w_top))
        err = ((w[run] - superposed) * mask).values
        print(f'linearity check (case {run}): max |direct - superposed| {np.nanmax(np.abs(err)) * DAY:.2e} m/d')
    w = {k: v.transpose('depth', 'distance') * DAY * mask for k, v in w.items()}

    m = mask.values == 1
    for run in ['A', 'B', 'C']:
        print(f'{run}: rms w {np.sqrt(np.nanmean(w[run].values[m]**2)):.2f} m/d (F=1: {np.sqrt(np.nanmean(w["F=1"].values[m]**2)):.2f}), '
              f'rms(w - w_F=1) {np.sqrt(np.nanmean((w[run] - w["F=1"]).values[m]**2)):.2f} m/d')

    fig, axes = plt.subplots(4, 2, figsize=(30, 22))   # no shared x: panel e is against depth
    for i, run in enumerate(['F=1', 'A', 'B', 'C']):
        section(axes[i, 0], x_km, depth, mld, w[run], f'w, {RUN_LABEL[run]}', 32, 'abcd'[i])
    for i, run in enumerate(['A', 'B', 'C'], start=1):
        section(axes[i, 1], x_km, depth, mld, w[run] - w['F=1'], f'difference {run} − (F = 1)', 10, 'fgh'[i - 1])
    for ax in list(axes[:, 0]) + list(axes[1:, 1]):
        ax.set_xlim(x_km.min(), x_km.max())
    for ax in list(axes[:3, 0]) + list(axes[1:3, 1]):
        ax.tick_params(labelbottom=False)
    axes[3, 0].set_xlabel('Along-track distance [km]')
    axes[3, 1].set_xlabel('Along-track distance [km]')

    # top right: F(d)
    ax = axes[0, 1]
    ax.text(-0.05, 1.08, 'e', transform=ax.transAxes, fontweight='bold', fontsize=22)
    ax.axhline(1, **RUN_STYLE['F=1'], label=RUN_LABEL['F=1'])
    for run in ['A', 'B', 'C']:
        ax.plot(depth, strain_profile(depth, *CASES[run]), label=RUN_LABEL[run], **RUN_STYLE[run])
    ax.set_xlim(15, 500)
    ax.set_ylim(0.65, 1.02)
    ax.set_xlabel('Depth [m]')
    ax.set_ylabel(r'F(d) = u$_x$(d) / u$_x$(0)')
    ax.set_title('strain profiles', loc='left')
    ax.grid(lw=0.5)
    ax.legend(fontsize=15, loc='lower left')
    make_axes_locatable(ax).append_axes('right', size='1.5%', pad=0.2).axis('off')

    bc_label = r'w(15 m) = w$_m$ (Ponte et al. 2013)' if bc == 'wm' else 'w(15 m) = 0'
    fig.suptitle(f'{NAME} | {START} → {END} | {bc_label}', y=0.995)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=110)
    print('saved', out)
