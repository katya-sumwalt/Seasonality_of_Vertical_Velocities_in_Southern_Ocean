'''Figure 2: section of strain, CT', bx, N2, w and VHT along the track of ft22-879-18 (2018-11-06 to 2018-11-15), with
the de Boyer Montegut et al. (2004) mixed layer depth (0.03 kg m-3 increase of sigma0 from 15 m, analysis/mld_comparison.py).

w(15 m) = w_m (Ponte et al. 2013, eq. 5) in panels e and f: w_stored = vertical_velocity - dw_wm for every season, before
concatenating the seasons and computing VHT (anomalies relative to the mean over all profiles).

usage: python -m plotting_figures.fig2_section [--bc wm|zero]
'''

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean
import config
from analysis.statistics import load_dw, calculate_vht
from plotting_figures.plot_utils import section_label, add_colorbar, section_panel

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 24})


def plot_figure_2(ds_tag, time_slice, name, out_paths, ct_lim=1.5, mld=None):
    '''Plot strain, CT', bx, N2, w and VHT for one tag and time window; save to each path in out_paths.
    ct_lim: CT' colour range [degC]; mld: MLD [m] on the section's time coordinate (default: the stored MLD).'''
    sub = ds_tag.sel(time=time_slice)
    mask = sub.maskprof
    mld = sub.MLD if mld is None else mld
    t0, t1 = str(sub.time.values[0])[:10], str(sub.time.values[-1])[:10]
    n_prof = sub.time.size

    fig, axes = plt.subplots(6, 1, figsize=(30, 40), sharex=True)

    # a: strain (line plot; a dummy colorbar keeps its width aligned with the image panels)
    ax = axes[0]
    section_label(ax, "a")
    tmp_img = ax.imshow(np.zeros((10, n_prof)), origin='lower', aspect='auto')
    ax.set_xlim(tmp_img.get_extent()[:2])
    add_colorbar(ax, tmp_img, '').remove()
    tmp_img.remove()
    ax.plot(np.arange(n_prof), sub.strain_neuro * 1e5, color='black', lw=2)
    ax.set_title(f"{name} | {t0} → {t1}")
    ax.set_ylim(-0.001, 2.83)
    ax.set_ylabel(r'$\sigma$ [s$^{-1}$]')
    ax.text(-0.049, 1.0, r" $\times$10$^{-5}$", transform=ax.transAxes, fontsize=20, va="top")
    ax.grid(True, lw=0.5)
    ax.tick_params(axis='x', labelbottom=False)

    CT_prime = (sub.temperature - sub.temperature.mean('time')) * mask
    section_panel(axes[1], CT_prime, mld, "b", cmocean.cm.thermal, -ct_lim, ct_lim,
                  "CT′ [°C]", crop_top=False)
    section_panel(axes[2], sub.bx * mask, mld, "c", 'RdBu_r', -2.22e-7, 2.22e-7,
                  "bₓ [s⁻²]", sci=True, crop_top=False)
    section_panel(axes[3], sub.N2 * mask, mld, "d", cmocean.cm.matter, 0, 5.3e-5,
                  r"N$^2$ [s$^{-2}$]", sci=True)
    section_panel(axes[4], -1 * sub.vertical_velocity * mask * 86400, mld, "e", 'RdBu_r', -32, 32,
                  r"w [m.d$^{-1}$]")
    section_panel(axes[5], sub.vht * mask, mld, "f", cmocean.cm.curl, -4430, 4430,
                  r"VHT [W.m$^{-2}$]", sci=True)

    axes[5].tick_params(axis='x', labelbottom=True)
    axes[5].set_xlabel("Along-track distance [km]")

    for path in out_paths:
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


def load(season, bc):
    ds = xr.open_dataset(config.season_file(season)).drop_vars('vht', errors='ignore')
    if bc == 'wm':
        dw = load_dw(season)
        ds['vertical_velocity'] = ds.vertical_velocity - dw.transpose(*ds.vertical_velocity.dims)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure 2 section')
    parser.add_argument('--bc', choices=['wm', 'zero'], default='wm', help='upper boundary condition w(15 m)')
    bc = parser.parse_args().bc
    name, start, end = config.SECTION
    out = config.out_path('fig2_section' + ('' if bc == 'wm' else '_w0'))

    ds_east = xr.concat([load(s, bc) for s in config.SEASONS], dim='distance')
    ds_east['vht'] = calculate_vht(ds_east)
    ds_tag = ds_east.where(ds_east.name == name, drop=True).swap_dims({'distance': 'time'}).sortby('time')

    m = xr.open_dataset(config.MLD_FILE)
    mld = xr.DataArray(m.mld_dbm.values, dims='time', coords={'time': m.time.values})
    section_time = ds_tag.sel(time=slice(start, end)).time
    assert np.array_equal(section_time.values, mld.time.values), 'MLD and section profiles differ'
    plot_figure_2(ds_tag, slice(start, end), name, [f'{out}.png', f'{out}.pdf'], ct_lim=2.0, mld=mld)
