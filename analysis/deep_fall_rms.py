'''rms w (w(15 m) = w_m) in deep layers for all profiles and for summer, winter and spring downsampled to the fall
sampling (100 draws), with the number of observations at mid-layer: tests whether the weak deep fall rms w comes from
the smaller fall sample.

usage: python -m analysis.deep_fall_rms
'''

import numpy as np
import xarray as xr
import config
from analysis.statistics import DAY, load_dw
from analysis.bootstrap import profile_length, select_profiles

LAYERS = [(200, 300), (300, 400), (400, 450), (450, 500)]
N_DRAWS = 100


def layer_rms(w, depth, a, b):
    sel = (depth >= a) & (depth < b)
    return np.sqrt(np.nanmean(w[sel] ** 2))


if __name__ == '__main__':
    data = {}
    for s in ['fall', 'summer', 'winter', 'spring']:
        ds = xr.open_dataset(config.season_file(s))[['vertical_velocity', 'maskprof']].load()
        dw = load_dw(s)
        w = ((ds.vertical_velocity - dw.transpose(*ds.vertical_velocity.dims)) * ds.maskprof * DAY).values
        data[s] = (w, *profile_length(ds))
    depth = np.abs(ds.depth.values)
    nf = data['fall'][1]
    for a, b in LAYERS:
        i = np.argmin(np.abs(depth - (a + b) / 2))
        print(f'{a}-{b} m')
        for s, (w, n, c) in data.items():
            full = layer_rms(w, depth, a, b)
            cnt = np.isfinite(w[i]).sum()
            if s == 'fall':
                print(f'  {s:7s} all {full:5.2f} m/d (n at {depth[i]:.0f} m: {cnt})')
                continue
            r = [layer_rms(w[:, select_profiles(n, c, nf, np.random.default_rng(k))], depth, a, b) for k in range(N_DRAWS)]
            print(f'  {s:7s} all {full:5.2f} (n {cnt}) | downsampled median {np.median(r):5.2f} '
                  f'[{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}]')
