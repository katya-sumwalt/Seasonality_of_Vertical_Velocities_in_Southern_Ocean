'''Figure S8: CT', SA', b' and bx along the Figure 2 section, with the de Boyer Montegut MLD, to check whether salinity
compensates the temperature anomalies in the buoyancy.

Anomalies are departures from the section mean at each depth, as CT' in Figure 2. The temperature and salinity
contributions to b' are estimated by a least-squares fit b' = a CT' + c SA' over the section (a = g alpha, c = -g beta);
the fit is printed.

usage: python -m plotting_figures.figS8_salinity_section
'''

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean
import config
from plotting_figures.plot_utils import section_panel

NAME, START, END = config.SECTION

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 24})

if __name__ == '__main__':
    out = config.out_path('figS8_salinity_section')
    ds = xr.concat([xr.open_dataset(config.season_file(s))[['temperature', 'salinity', 'buoyancy', 'bx', 'maskprof',
                                                            'name']] for s in config.SEASONS], dim='distance')
    sub = ds.where(ds.name == NAME, drop=True).swap_dims({'distance': 'time'}).sortby('time').sel(time=slice(START, END))
    mask = sub.maskprof
    m = xr.open_dataset(config.MLD_FILE)
    assert np.array_equal(m.time.values, sub.time.values), 'MLD and section profiles differ'
    mld = m.mld_dbm.values

    ct_p = (sub.temperature - sub.temperature.mean('time')) * mask
    sa_p = (sub.salinity - sub.salinity.mean('time')) * mask
    b_p = (sub.buoyancy - sub.buoyancy.mean('time')) * mask

    ok = np.isfinite(ct_p.values) & np.isfinite(sa_p.values) & np.isfinite(b_p.values)
    X = np.column_stack([ct_p.values[ok], sa_p.values[ok]])
    (a, c), *_ = np.linalg.lstsq(X, b_p.values[ok], rcond=None)
    b_t, b_s = a * ct_p, c * sa_p
    resid = b_p.values[ok] - X @ np.array([a, c])
    print(f'b\' = {a:.3e} CT\' + {c:.3e} SA\' (g alpha, -g beta); explained variance '
          f'{1 - resid.var() / b_p.values[ok].var():.3f}')
    print(f'rms b\' {np.nanstd(b_p.values):.2e} | rms temperature part {np.nanstd(b_t.values):.2e} | '
          f'rms salinity part {np.nanstd(b_s.values):.2e} | corr(b_T, b_S) '
          f'{np.corrcoef(b_t.values[ok], b_s.values[ok])[0, 1]:.2f}')
    below = ok & (np.abs(sub.depth.values)[:, None] > mld[None, :])
    print(f'below the MLD: rms b\'/ rms temperature part = {np.std(b_p.values[below]) / np.std(b_t.values[below]):.2f}, '
          f'corr(b_T, b_S) = {np.corrcoef(b_t.values[below], b_s.values[below])[0, 1]:.2f}')

    fig, axes = plt.subplots(4, 1, figsize=(30, 27), sharex=True)
    section_panel(axes[0], ct_p, mld, 'a', cmocean.cm.thermal, -2, 2, "CT′ [°C]", crop_top=False)
    section_panel(axes[1], sa_p, mld, 'b', cmocean.cm.haline, -0.25, 0.25, r"S$_A$′ [g kg$^{-1}$]", crop_top=False)
    section_panel(axes[2], b_p, mld, 'c', 'RdBu_r', -2.5e-3, 2.5e-3, r"b′ [m s$^{-2}$]", sci=True, crop_top=False)
    section_panel(axes[3], sub.bx * mask, mld, 'd', 'RdBu_r', -2.22e-7, 2.22e-7, "bₓ [s⁻²]", sci=True, crop_top=False)
    axes[0].set_title(f'{NAME} | {START} → {END}')
    axes[3].tick_params(axis='x', labelbottom=True)
    axes[3].set_xlabel('Along-track distance [km]')

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=100)
    print('saved', out)
