'''Mixed layer depth along the Figure 2 section: de Boyer Montegut et al. (2004) threshold (as in the paper:
0.03 kg m-3 increase of sigma0 from 15 m) vs Holte & Talley (2009) hybrid algorithm (python package holteandtalley).

temperature/salinity in the seal files are CT/SA; sigma0 from gsw.
Holte & Talley thresholds are referenced to the shallowest level (15 m), as the data start there.

usage: python -m analysis.mld_comparison
output: config.MLD_FILE (de Boyer Montegut MLD used in Figures 2, S2, S4 and S8) and config.OUT_DIR/mld_comparison.png
'''

import os
import numpy as np
import xarray as xr
import gsw
import cmocean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from holteandtalley import HolteAndTalley
from holteandtalley.profile import Profile
import config


def _calculateMLTFIT(self, values, gradients):
    '''holteandtalley Profile.calculateMLTFIT with the numpy fix `if len(out[1])` (the package uses `if out[1]`,
    which fails on empty residual arrays with recent numpy). Logic otherwise unchanged.'''
    errors, fits = [], []
    for num in range(2, len(self.pressures)):
        out = np.polyfit(self.pressures[0:num], values[0:num], 1, full=True)
        fits.append(out[0])
        errors.append(out[1][0] if len(out[1]) else 0)
    errorsum = np.sum(errors)
    self.errors = errors
    mltBestFit = None
    for index in range(len(errors)):
        if errors[index] / errorsum > 10**-10:
            mltBestFit = fits[index - 1]
            self.mltfitline = mltBestFit
            self.mltfitindex = index - 1
            break
    steepest = np.argwhere(np.abs(gradients) == np.max(np.abs(gradients)))[-1][0] + 1
    thermoclineFit = np.polyfit(self.pressures[steepest - 1:steepest + 2], values[steepest - 1:steepest + 2], 1, full=True)[0]
    if thermoclineFit[0] != mltBestFit[0]:
        depth = abs(float(thermoclineFit[1] - mltBestFit[1]) / float(thermoclineFit[0] - mltBestFit[0]))
    else:
        depth = self.pressures[steepest]
    return self.findNearestPressureIndex(depth), self.pressures[self.findNearestPressureIndex(depth)]


Profile.calculateMLTFIT = _calculateMLTFIT
if not hasattr(np, 'round_'):   # removed in numpy 2, still used by holteandtalley
    np.round_ = np.round

NAME, START, END = config.SECTION
DAY = 86400

plt.rcParams.update({"font.family": "serif", "font.size": 18})


def mld_threshold(z, sigma0, dsigma=0.03):
    '''First depth where sigma0 exceeds its 15 m value by dsigma (linear interpolation).'''
    i = np.argmax(sigma0 - sigma0[0] > dsigma)
    if i == 0:
        return np.nan
    return np.interp(sigma0[0] + dsigma, sigma0[i - 1:i + 1], z[i - 1:i + 1])


def mld_holte_talley(p, ct, sa, sigma0):
    '''Holte & Talley (2009) density- and temperature-algorithm MLD [dbar].'''
    try:
        h = HolteAndTalley(list(p), list(ct), list(sa), list(sigma0))
        return float(h.densityMLD), float(h.tempMLD)
    except Exception:
        return np.nan, np.nan


def main():
    out = config.out_path('mld_comparison')
    # Figure 2 section
    ds = xr.open_dataset(config.season_file('spring'))
    sec = ds.where(ds.name == NAME, drop=True).swap_dims({'distance': 'time'}).sortby('time').sel(time=slice(START, END))
    sec = sec.swap_dims({'time': 'distance'}).assign_coords(distance=np.arange(sec.time.size) * 1000.).load()
    x_km, depth = sec.distance.values / 1000, -sec.depth.values
    mask = sec.maskprof

    ct = sec.temperature.where(mask.notnull()).values
    sa = sec.salinity.where(mask.notnull()).values
    sigma0 = gsw.sigma0(sa, ct)

    n = sec.sizes['distance']
    mld_dbm, mld_ht, mld_ht_temp = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for j in range(n):
        valid = np.isfinite(sigma0[:, j])
        if valid.sum() < 20:
            continue
        z = depth[valid]
        lat = float(sec.lat[j])
        p = gsw.p_from_z(-z, lat)
        mld_dbm[j] = mld_threshold(z, sigma0[valid, j])
        p_d, p_t = mld_holte_talley(p, ct[valid, j], sa[valid, j], sigma0[valid, j])
        mld_ht[j] = -gsw.z_from_p(p_d, lat)
        mld_ht_temp[j] = -gsw.z_from_p(p_t, lat)

    diff = mld_ht - mld_dbm
    print(f'stored MLD vs recomputed dBM: median |diff| {np.nanmedian(np.abs(sec.MLD.values - mld_dbm)):.1f} m, '
          f'max {np.nanmax(np.abs(sec.MLD.values - mld_dbm)):.1f} m')
    print(f'dBM: median {np.nanmedian(mld_dbm):.0f} m, range {np.nanmin(mld_dbm):.0f}-{np.nanmax(mld_dbm):.0f} m')
    print(f'H&T density: median {np.nanmedian(mld_ht):.0f} m, range {np.nanmin(mld_ht):.0f}-{np.nanmax(mld_ht):.0f} m, NaN {np.isnan(mld_ht).sum()}')
    print(f'H&T - dBM: median {np.nanmedian(diff):.1f} m, mean {np.nanmean(diff):.1f} m, rms {np.sqrt(np.nanmean(diff**2)):.1f} m, '
          f'|diff| < 10 m: {np.nanmean(np.abs(diff) < 10) * 100:.0f}% of profiles, > 50 m: {np.nanmean(np.abs(diff) > 50) * 100:.0f}%')

    # figure
    fig, axes = plt.subplots(5, 1, figsize=(24, 30), sharex=True)

    def mld_lines(ax, legend=False):
        ax.plot(x_km, mld_dbm, 'k', lw=2, label='de Boyer Montégut (2004), 0.03 kg m$^{-3}$ from 15 m')
        ax.plot(x_km, mld_ht, color='magenta', lw=2, ls='--', label='Holte & Talley (2009), density algorithm')
        if legend:
            ax.legend(loc='lower right', fontsize=15)

    def section(ax, data, cmap, vmin, vmax, label, letter, title):
        img = ax.pcolormesh(x_km, depth, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        mld_lines(ax, legend=letter == 'a')
        ax.set_ylim(500, 15)
        ax.set_yticks([100, 200, 300, 400, 500])
        ax.set_ylabel('Depth [m]')
        ax.set_title(title, loc='left')
        ax.text(-0.04, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=22)
        plt.colorbar(img, cax=make_axes_locatable(ax).append_axes('right', size='1%', pad=0.2), label=label)

    section(axes[0], sigma0, cmocean.cm.dense, np.nanpercentile(sigma0, 1), np.nanpercentile(sigma0, 99),
            r'$\sigma_0$ [kg m$^{-3}$]', 'a', f'{NAME} | {START} → {END}')
    section(axes[1], (sec.temperature - sec.temperature.mean('distance')) * mask, cmocean.cm.thermal, -1.5, 1.5,
            'CT′ [°C]', 'b', 'CT′')
    section(axes[2], -1 * sec.vertical_velocity * mask * DAY, 'RdBu_r', -32, 32, r'w [m d$^{-1}$]', 'c', 'w (paper)')

    ax = axes[3]
    ax.plot(x_km, mld_dbm, 'k', lw=2, label='de Boyer Montégut (2004)')
    ax.plot(x_km, mld_ht, color='magenta', lw=2, ls='--', label='Holte & Talley (2009), density')
    ax.plot(x_km, mld_ht_temp, color='tab:cyan', lw=1, alpha=0.7, label='Holte & Talley (2009), temperature')
    ax.set_ylim(np.nanmax([mld_dbm, mld_ht]) + 20, 0)
    ax.set_ylabel('MLD [m]')
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(lw=0.5)
    ax.text(-0.04, 1.08, 'd', transform=ax.transAxes, fontweight='bold', fontsize=22)
    make_axes_locatable(ax).append_axes('right', size='1%', pad=0.2).axis('off')

    ax = axes[4]
    ax.bar(x_km, diff, width=1, color=np.where(diff > 0, 'tab:red', 'tab:blue'))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('MLD$_{H&T}$ − MLD$_{dBM}$ [m]')
    ax.set_title(f'median {np.nanmedian(diff):.1f} m, rms {np.sqrt(np.nanmean(diff**2)):.1f} m; '
                 f'|Δ| < 10 m for {np.nanmean(np.abs(diff) < 10) * 100:.0f}% of profiles', loc='left', fontsize=16)
    ax.set_xlabel('Along-track distance [km]')
    ax.grid(lw=0.5)
    ax.text(-0.04, 1.08, 'e', transform=ax.transAxes, fontweight='bold', fontsize=22)
    make_axes_locatable(ax).append_axes('right', size='1%', pad=0.2).axis('off')

    fig.savefig(out + '.png', bbox_inches='tight', dpi=100)
    fig.savefig(out + '.pdf', bbox_inches='tight')
    os.makedirs(os.path.dirname(config.MLD_FILE) or '.', exist_ok=True)
    xr.Dataset({'mld_dbm': ('distance', mld_dbm), 'mld_ht_density': ('distance', mld_ht),
                'mld_ht_temperature': ('distance', mld_ht_temp), 'mld_stored': ('distance', sec.MLD.values)},
               coords={'distance': sec.distance, 'time': ('distance', sec.time.values)}).to_netcdf(config.MLD_FILE)
    print('saved', out, config.MLD_FILE)


if __name__ == '__main__':
    main()
