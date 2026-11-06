'''Rossby number Ro = zeta / f of the NeurOST surface relative vorticity along the seal tracks (one value per profile),
to check that the flow sampled is in the quasi-geostrophic regime (Ro << 1).

usage: python -m analysis.rossby_number
'''

import numpy as np
import xarray as xr
import config


def rossby(season):
    ds = xr.open_dataset(config.season_file(season))[['zeta_neuro', 'f']]
    ro = (ds.zeta_neuro / ds.f.mean('depth')).values   # f is constant with depth
    return ro[np.isfinite(ro)]


if __name__ == '__main__':
    ro = {s: rossby(s) for s in config.SEASONS}
    for s in ['fall', 'summer', 'winter', 'spring']:
        print(f'{s:7s}: mean |Ro| {np.mean(np.abs(ro[s])):.3f} ({ro[s].size:,} profiles)')
    x = np.concatenate([ro[s] for s in config.SEASONS])
    print(f'all    : mean |Ro| {np.mean(np.abs(x)):.3f}, rms {np.sqrt(np.mean(x ** 2)):.3f}, '
          f'95th percentile of |Ro| {np.percentile(np.abs(x), 95):.3f} ({x.size:,} profiles)')
