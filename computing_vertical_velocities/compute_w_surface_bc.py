'''Vertical velocity from the 2D QG omega equation with a prescribed (non-zero) w at the upper boundary.

Adapted from compute_w.py and cleanup_seal_data.py (K. Sumwalt).

Differences from compute_w.py:
  - iParams BCs ['reflect', 'fixed'] -> ['fixed', 'fixed']: 'reflect' is not an xinvert option and behaved
    as 'fixed' (newer xinvert versions reject it). With fixed BCs the top and bottom rows of the solution
    keep the values of the initial guess (icbc), so writing w_top into the 15 m row of icbc imposes
    w(15 m) = w_top. The repo uses w_top = 0.
  - N2 <= 0 is replaced by min|N2| (the repo has N2.where(N2 < 0, ...), which on these files replaces all
    positive N2 instead). With this and the 700 m extension, w_top = 0 reproduces the vertical_velocity
    stored in the seasonal files to within 1e-6 (correlation -1: the stored field is -1 x this output).
  - one inversion with icbc instead of two (the first one only served as initial guess).

Sign convention: the output is w as plotted in Figure 2 panel e, i.e. -1 * the stored 'vertical_velocity'.
'''

import numpy as np
import xarray as xr
from xinvert import FiniteDiff
from xinvert.apps import __template, __mask_FS, inv_standard2D

_undeftmp = -9.99e8

default_iParams = {
    'BCs'      : ['fixed', 'fixed'],   # [depth, distance]
    'undef'    : np.nan,
    'mxLoop'   : 2000,
    'tolerance': 1e-10,
}


def invert_omega_2D(F, dims, coords='cartesian', icbc=None, mParams=None, iParams=default_iParams):
    return __template(__coeffs_omega_2D, inv_standard2D, 2, F, dims, coords,
                      icbc, ['f', 'N2'], mParams, iParams)


def __coeffs_omega_2D(force, dims, coords, mParams, iParams, icbc):
    '''Coefficients of f^2 w_zz + N2 w_xx = F (dims[0] is z, dims[1] is x).'''
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)
    A = zero + mParams['f']**2
    B = zero
    C = zero + mParams['N2']
    F = maskF.where(maskF != _undeftmp, _undeftmp)
    return F, initS, (A, B, C)


def extend_to_depth(ds, depth_to_extend_to=700):
    '''Repeat the deepest level down to depth_to_extend_to (as cleanup_seal_data.extend_dataset_to_depth).'''
    z = ds.depth.values
    sign = np.sign(z[-1])
    new_depths = np.arange(np.abs(z[-1]) + 1, depth_to_extend_to + 1)
    if new_depths.size == 0:
        return ds
    with_depth = [v for v in ds.data_vars if 'depth' in ds[v].dims]
    deepest = ds[with_depth].isel(depth=-1)
    ext = xr.concat([ds[with_depth]] + [deepest.assign_coords(depth=sign * d) for d in new_depths], dim='depth')
    for v in ds.data_vars:
        if v not in with_depth:
            ext[v] = ds[v]
    return ext


def w_mixing(ds, Av=4e-2):
    '''Mixing-driven vertical velocity at the shallowest level, Ponte et al. (2013, JPO) eq. 5:

        w_m = g / (f^2 rho0) * Av * lap(rho) = -(Av / f^2) * d(bx)/dx     (b = -g rho / rho0, 1D along track)

    Input:
        ds: dataset with dims (depth, distance), variables bx [s-2] and f [s-1]; distance in m.
        Av: mixed-layer turbulent viscosity [m2 s-1] (4e-2 in Ponte et al. 2013).
    Output:
        w_m(distance) [m s-1], positive upward (same convention as compute_w output); NaN -> 0.
    '''
    i_top = int(np.argmin(np.abs(ds.depth.values)))
    bx = ds['bx'].isel(depth=i_top)
    f = ds['f'].isel(depth=i_top)
    bxx = bx.differentiate('distance')
    return (-Av / f**2 * bxx).fillna(0)


def compute_w(ds, w_top=0.0, strain_variable='strain_neuro', depth_to_extend_to=700, iParams=default_iParams):
    '''Solve the omega equation on a (depth, distance) seal section with w = w_top at the shallowest level.

    Input:
        ds: dataset with dims (depth, distance) and variables bx, f, N2 and strain_variable;
            distance in m (1 km spacing), depth in m (either sign, shallowest first).
        w_top: w imposed at the shallowest depth (15 m) [m s-1], same convention as the output.
               Scalar, array of length ds.distance, or DataArray on 'distance'. 0 reproduces the repo.
        strain_variable: 'strain_neuro' or a depth-varying strain variable.
        depth_to_extend_to: the deepest level is repeated down to this depth [m] before solving
                            (bottom boundary w = 0 there); the output is cropped back to ds.depth.
    Output:
        w(depth, distance) [m s-1], as plotted in Figure 2 panel e (= -1 * stored 'vertical_velocity').
    '''
    sec = extend_to_depth(ds[['bx', 'f', 'N2', strain_variable]], depth_to_extend_to)

    # QG forcing
    fd = FiniteDiff({'Z': 'depth', 'X': 'distance'}, BCs={'Z': 'reflect', 'X': 'fixed'}, coords='cartesian')
    Q = sec['bx'] * sec[strain_variable]
    divQ = -2 * fd.divg(Q, dims=['X']).load()
    FQvec = xr.where(np.isfinite(divQ), divQ, np.nan).transpose('depth', 'distance').load()

    N2 = sec['N2']
    N2 = N2.where(N2 > 0, np.nanmin(np.abs(N2)))
    mParams = {'N2': N2, 'f': sec['f']}

    # initial guess / boundary values: zero everywhere except the top row.
    # The solution is multiplied by -1 below (as in the repo), so the top row holds -w_top.
    i_top = int(np.argmin(np.abs(sec.depth.values)))
    w_top = np.broadcast_to(np.asarray(w_top, dtype=float), sec.distance.shape)
    icbc = xr.zeros_like(FQvec)
    icbc[i_top, :] = -1 * w_top

    W = invert_omega_2D(FQvec, dims=['depth', 'distance'], coords='cartesian',
                        icbc=icbc, iParams=dict(iParams), mParams=mParams)
    return (-1 * W).sel(depth=ds.depth)
