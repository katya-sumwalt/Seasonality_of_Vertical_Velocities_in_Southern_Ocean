'''Joint PDF of w' and CT' (winter, 200-500 m), its contribution to <VHT>, and the seasonal decompositions
<w'CT'> = r * rms(w') * rms(CT')  (r = <w'CT'> / (rms(w') rms(CT'))) and <VHT> = strength + frequency (Figure 4).

Anomalies are departures from the seasonal mean at each depth (over distance), w positive upward
(= -1 x stored vertical_velocity + dw), masked with maskprof; VHT = rho cp w' CT'.

usage: python -m analysis.joint_pdf [--bc wm|zero] [--case A|B|C]   (prints the numbers quoted in the text)
'''

import argparse
import numpy as np
import xarray as xr
import config
from analysis.statistics import RCP, DAY, load_dw

SEASONS = ['winter', 'spring', 'summer', 'fall']
LAYERS = {'200-500 m': (200, 500), '15-100 m': (15, 100)}
PDF_LAYER = '200-500 m'
W_EDGES = np.linspace(-130, 130, 131)   # 2 m/d bins; range covers all winter 200-500 m data
T_EDGES = np.linspace(-3.5, 7, 106)     # 0.1 degC bins


def anomalies(season, top, bottom, bc='wm', case=None):
    '''w' [m/d] and CT' [degC] at all valid points in a depth layer.'''
    ds = xr.open_dataset(config.season_file(season))[['temperature', 'vertical_velocity', 'maskprof']]
    w = -1 * ds.vertical_velocity
    if bc == 'wm':
        w = w + load_dw(season)
    if case:
        w = w + load_dw(season, case)
    w_p = (w - w.mean('distance')) * ds.maskprof
    t_p = (ds.temperature - ds.temperature.mean('distance')) * ds.maskprof
    layer = dict(depth=slice(-top, -bottom))
    w_p, t_p = w_p.sel(**layer).values.ravel() * DAY, t_p.sel(**layer).values.ravel()
    ok = np.isfinite(w_p) & np.isfinite(t_p)
    return w_p[ok], t_p[ok]


def decomposition(w_p, t_p):
    '''<VHT> = rho cp r rms(w') rms(CT'), and the frequency/strength split
    <VHT> = P+ V+ + P- V- = (V+ + V-)/2 [strength] + (P+ - P-)(V+ - V-)/2 [frequency], exact since P+ + P- = 1.'''
    rms_w, rms_t = np.sqrt(np.mean(w_p**2)), np.sqrt(np.mean(t_p**2))
    cov = np.mean(w_p * t_p)
    vht = RCP * w_p * t_p / DAY
    p_up, v_up, v_down = np.mean(vht > 0), vht[vht > 0].mean(), vht[vht < 0].mean()
    return {'rms_w': rms_w, 'rms_t': rms_t, 'r': cov / (rms_w * rms_t), 'vht': RCP * cov / DAY,
            'P+': p_up, 'V+': v_up, 'V-': v_down,
            'strength': 0.5 * (v_up + v_down), 'frequency': 0.5 * (2 * p_up - 1) * (v_up - v_down)}


def seasonal_decomposition(bc='wm', case=None):
    '''Decomposition for every layer and season (printed); returns it and the winter anomalies of PDF_LAYER.'''
    stats = {}
    for name, (top, bottom) in LAYERS.items():
        print(f'\n{name}: <VHT> = rho cp r rms(w\') rms(CT\')')
        for s in SEASONS:
            w_p, t_p = anomalies(s, top, bottom, bc, case)
            d = decomposition(w_p, t_p)
            stats[(name, s)] = d
            if s == 'winter' and name == PDF_LAYER:
                w_win, t_win = w_p, t_p
            print(f"  {s:7s} <VHT> {d['vht']:6.1f} W/m2 | r {d['r']:.3f} | rms w' {d['rms_w']:5.2f} m/d | "
                  f"rms CT' {d['rms_t']:.3f} degC | n {w_p.size:,} | P+ {d['P+']:.3f}, V+ {d['V+']:.0f}, "
                  f"V- {d['V-']:.0f}: strength {d['strength']:.1f}, frequency {d['frequency']:.1f}")
        for s in ['spring', 'summer', 'fall']:
            a, b = stats[(name, 'winter')], stats[(name, s)]
            print(f"  winter/{s:6s}: <VHT> x{a['vht'] / b['vht']:.2f} = r x{a['r'] / b['r']:.2f} "
                  f"* rms w' x{a['rms_w'] / b['rms_w']:.2f} * rms CT' x{a['rms_t'] / b['rms_t']:.2f}")
    return stats, w_win, t_win


def joint_pdf(w_p, t_p, vht_all):
    '''Joint PDF of (w', CT') and the contribution of each bin to <VHT> [W/m2 per bin], with quadrant sums printed.
    vht_all: <VHT> from all data, printed for comparison with the binned estimate.'''
    pdf, _, _ = np.histogram2d(w_p, t_p, bins=[W_EDGES, T_EDGES], density=True)
    wc, tc = 0.5 * (W_EDGES[1:] + W_EDGES[:-1]), 0.5 * (T_EDGES[1:] + T_EDGES[:-1])
    W, T = np.meshgrid(wc, tc, indexing='ij')
    dA = np.diff(W_EDGES)[0] * np.diff(T_EDGES)[0]
    contrib = RCP * (W / DAY) * T * pdf * dA                 # sums to <VHT> within the plotted range
    quad = {'w\'>0, CT\'>0': (W > 0) & (T > 0), 'w\'<0, CT\'<0': (W < 0) & (T < 0),
            'w\'>0, CT\'<0': (W > 0) & (T < 0), 'w\'<0, CT\'>0': (W < 0) & (T > 0)}
    print(f"\nwinter {PDF_LAYER}: <VHT> from binned PDF {contrib.sum():.1f} W/m2 (all data {vht_all:.1f})")
    for k, sel in quad.items():
        print(f'  {k}: {contrib[sel].sum():7.1f} W/m2, probability {np.sum(pdf[sel]) * dA:.2f}')
    return pdf, wc, tc, contrib, quad


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--bc', choices=['wm', 'zero'], default='wm', help='upper boundary condition w(15 m)')
    parser.add_argument('--case', choices=['A', 'B', 'C'], default=None, help='depth-dependent strain case')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(__doc__.splitlines()[0])
    stats, w_win, t_win = seasonal_decomposition(args.bc, args.case)
    joint_pdf(w_win, t_win, stats[(PDF_LAYER, 'winter')]['vht'])
