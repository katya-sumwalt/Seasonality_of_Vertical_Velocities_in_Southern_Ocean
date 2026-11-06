'''Change in w from replacing w(15 m) = 0 by w(15 m) = w_m (Ponte et al. 2013, eq. 5), for one season.

The omega equation is linear, so w_new = w_paper + dw, where dw solves the unforced equation
(same coefficients and mask) with w(15 m) = w_m, w = 0 at 700 m and at the lateral chunk edges.
Sections are split into contiguous 1 km segments per tag, then into chunks of <= 1000 profiles.

usage: python -m computing_vertical_velocities.compute_dw_mixing_bc <season>
output: config.DW_WM_DIR/dw_<season>.nc, with dw(depth, distance) [m s-1] in the panel-e convention
        (w positive upward = -1 * stored vertical_velocity) and w_m(distance) [m s-1].
'''

import os
import argparse
import numpy as np
import xarray as xr
import config
from computing_vertical_velocities.compute_w_surface_bc import compute_w, w_mixing

CHUNK = 1000


def chunks(ds):
    '''Index slices of contiguous (same tag, 1 km step) segments, cut into pieces of <= CHUNK profiles.'''
    name, dist = ds.name.values, ds.distance.values
    breaks = np.where((name[1:] != name[:-1]) | (np.diff(dist) != 1000))[0] + 1
    edges = np.concatenate([[0], breaks, [name.size]])
    for i0, i1 in zip(edges[:-1], edges[1:]):
        for j0 in range(i0, i1, CHUNK):
            yield slice(j0, min(j0 + CHUNK, i1))


def main(season):
    os.makedirs(config.DW_WM_DIR, exist_ok=True)
    ds = xr.open_dataset(config.season_file(season))[['bx', 'f', 'N2', 'strain_neuro', 'name']].load()

    dw = np.zeros((ds.sizes['depth'], ds.sizes['distance']))
    wm = np.zeros(ds.sizes['distance'])
    for k, sl in enumerate(chunks(ds)):
        sec = ds.isel(distance=sl)
        if sec.sizes['distance'] < 3:
            continue
        sec = sec.assign_coords(distance=np.arange(sec.sizes['distance']) * 1000.)
        w_top = w_mixing(sec)
        unforced = sec.assign(bx=sec.bx * 0)   # zero forcing, same NaN mask as the full problem
        dw[:, sl] = compute_w(unforced, w_top=w_top).transpose('depth', 'distance').values
        wm[sl] = w_top.values
        print(f'{season} chunk {k}: profiles {sl.start}-{sl.stop}', flush=True)

    out = xr.Dataset({'dw': (('depth', 'distance'), dw), 'w_m': ('distance', wm)},
                     coords={'depth': ds.depth, 'distance': ds.distance})
    out.attrs['description'] = __doc__
    out.to_netcdf(os.path.join(config.DW_WM_DIR, f'dw_{season}.nc'))
    print('saved', season)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dw for w(15 m) = w_m, one season')
    parser.add_argument('season', choices=config.SEASONS)
    main(parser.parse_args().season)
