'''Change in w from a depth-dependent strain u_x(x, d) = u_x(x, 0) F(d) (profile approximating the first baroclinic
mode of Hua & Haidvogel 1986), for one case and one season.

The forcing becomes F(d) x the original forcing, so by linearity w_case = w_paper + dw, where dw solves the omega
equation forced by (F - 1) x the original forcing, with w(15 m) = 0 and the same N2, bx, grid and chunks.
F is evaluated on the 15-500 m grid and its 500 m value is repeated down to 700 m with the other fields.

usage: python -m computing_vertical_velocities.compute_dw_strain_profile <case A|B|C> <season>
output: config.DW_STRAIN_DIR/dw_<case>_<season>.nc, dw(depth, distance) [m s-1] in the panel-e convention
        (w positive upward = -1 * stored vertical_velocity).
'''

import os
import argparse
import numpy as np
import xarray as xr
import config
from computing_vertical_velocities.compute_w_surface_bc import compute_w
from computing_vertical_velocities.compute_dw_mixing_bc import chunks

CASES = {'A': (4000.0, 0.350),   # moderate decrease, F(500 m) = 0.712
         'B': (4000.0, 0.574),   # weaker decrease,   F(500 m) = 0.834
         'C': (5500.0, 0.467)}   # weaker decrease,   F(500 m) = 0.880
RUN_LABEL = {'F=1': 'F = 1', **{k: f'{k}: H={H:.0f} m, h/H={r:.3f}' for k, (H, r) in CASES.items()}}


def strain_profile(depth, H, h_over_H):
    '''F(d), d positive downward [m]; H total ocean depth [m]. F(0) = 1.'''
    h = h_over_H * H
    stretched_depth = np.expm1(-depth / h) / np.expm1(-H / h)
    return 1.0 - 0.625 * (1.0 - np.cos(np.pi * stretched_depth))


def main(case, season):
    os.makedirs(config.DW_STRAIN_DIR, exist_ok=True)
    ds = xr.open_dataset(config.season_file(season))[['bx', 'f', 'N2', 'strain_neuro', 'name']].load()
    F = xr.DataArray(strain_profile(np.abs(ds.depth.values), *CASES[case]), dims='depth', coords={'depth': ds.depth})
    ds['strain_anom'] = ds.strain_neuro * (F - 1)   # (F - 1) x surface strain -> (F - 1) x original forcing

    dw = np.zeros((ds.sizes['depth'], ds.sizes['distance']))
    for k, sl in enumerate(chunks(ds)):
        sec = ds.isel(distance=sl)
        if sec.sizes['distance'] < 3:
            continue
        sec = sec.assign_coords(distance=np.arange(sec.sizes['distance']) * 1000.)
        dw[:, sl] = compute_w(sec, strain_variable='strain_anom').transpose('depth', 'distance').values
        print(f'{case} {season} chunk {k}: profiles {sl.start}-{sl.stop}', flush=True)

    out = xr.Dataset({'dw': (('depth', 'distance'), dw), 'F': F}, coords={'depth': ds.depth, 'distance': ds.distance})
    out.attrs['description'] = __doc__
    out.attrs['H'], out.attrs['h_over_H'] = CASES[case]
    out.to_netcdf(os.path.join(config.DW_STRAIN_DIR, f'dw_{case}_{season}.nc'))
    print('saved', case, season)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dw for a depth-dependent strain case, one season')
    parser.add_argument('case', choices=list(CASES))
    parser.add_argument('season', choices=config.SEASONS)
    args = parser.parse_args()
    main(args.case, args.season)
