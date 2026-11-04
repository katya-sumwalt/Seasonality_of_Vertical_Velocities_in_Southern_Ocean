'''Figure S1: L-kurtosis of w and VHT vs depth with w(15 m) = w_m, and a block-bootstrap 95% interval.

L-kurtosis (analysis.statistics.l_kurtosis) at each depth over all masked profiles of a season. Bootstrap as in
analysis/bootstrap.py (no downsampling): non-overlapping blocks of BLOCK contiguous profiles (same tag, 1 km step) are
drawn with replacement until the season has its original number of blocks; VHT anomalies are relative to the mean of
each resample. Shading: 2.5-97.5 percentiles over the resamples. Thick lines: w(15 m) = w_m (all profiles); thin faded
lines: w(15 m) = 0.

usage: python -m plotting_figures.figS1_lkurtosis [--n-iter N]
'''

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
from analysis.statistics import RCP, DAY, load_dw, l_kurtosis
from analysis.bootstrap import blocks, BLOCK, N_ITER, SEED
from plotting_figures.plot_utils import COLORS, LABELS

SEASONS = config.SEASONS
LAYERS = {'15-100 m': (15, 100), '200-500 m': (200, 500)}

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 22})


def fields(season):
    '''numpy w (stored convention), T and validity for w_m ('new') and w = 0 ('old').'''
    ds = xr.open_dataset(config.season_file(season))[['temperature', 'vertical_velocity', 'maskprof', 'name']].load()
    dw = load_dw(season).transpose(*ds.vertical_velocity.dims).values
    valid = np.isfinite(ds.maskprof.values)
    return {'new': ds.vertical_velocity.values - dw, 'old': ds.vertical_velocity.values, 'T': ds.temperature.values,
            'valid': valid, 'n': valid.sum(0), 'name': ds.name.values, 'dist': ds.distance.values}


def kurtosis_w_vht(a, ver, idx):
    '''L-kurtosis profiles of masked w and VHT for profiles idx (anomalies relative to the mean of these profiles).'''
    w, T, valid = a[ver][:, idx], a['T'][:, idx], a['valid'][:, idx]
    w_up = -w
    vht = RCP * (T - np.nanmean(T, 1, keepdims=True)) * (w_up - np.nanmean(w_up, 1, keepdims=True))
    return l_kurtosis(np.where(valid, w * DAY, np.nan)), l_kurtosis(np.where(valid, vht, np.nan))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure S1')
    parser.add_argument('--n-iter', type=int, default=N_ITER, help='number of bootstrap resamples')
    n_iter = parser.parse_args().n_iter
    out = config.out_path('figS1_lkurtosis')

    depth = -xr.open_dataset(config.season_file('summer')).depth.values
    rng = np.random.default_rng(SEED)
    res = {}
    for s in SEASONS:
        a = fields(s)
        all_idx = np.where(a['n'] > 0)[0]
        block_idx = blocks(a)
        res[s] = {'new': kurtosis_w_vht(a, 'new', all_idx), 'old': kurtosis_w_vht(a, 'old', all_idx)}
        boot_w, boot_vht = [], []
        for _ in range(n_iter):
            idx = np.concatenate([block_idx[i] for i in rng.integers(0, len(block_idx), len(block_idx))])
            kw, kv = kurtosis_w_vht(a, 'new', idx)
            boot_w.append(kw)
            boot_vht.append(kv)
        res[s]['boot'] = (np.array(boot_w), np.array(boot_vht))
        print(f'{s:7s}: {len(block_idx)} blocks of {BLOCK} profiles, {n_iter} resamples', flush=True)

    for j, var in enumerate(['w', 'VHT']):
        print(f'\nL-kurtosis {var}, layer mean: w=0 | w_m [2.5, 97.5 bootstrap]')
        for name, (top, bottom) in LAYERS.items():
            sel = (depth >= top) & (depth <= bottom)
            layer = {s: np.nanmean(res[s]['boot'][j][:, sel], 1) for s in SEASONS}
            for s in SEASONS:
                lo, hi = np.percentile(layer[s], [2.5, 97.5])
                print(f"  {name:9s} {s:7s} {np.nanmean(res[s]['old'][j][sel]):.3f} | "
                      f"{np.nanmean(res[s]['new'][j][sel]):.3f} [{lo:.3f}, {hi:.3f}]")
            for s in ['summer', 'spring', 'fall']:
                print(f'  {name}: fraction of resamples with winter < {s}: {np.mean(layer["winter"] < layer[s]):.3f}')

    fig = plt.figure(figsize=(20, 10))
    for j, (xlabel, xlim) in enumerate([('L-Kurtosis w', [0.03, 0.55]), ('L-Kurtosis VHT', [0.02, 0.8])]):
        ax = plt.subplot(1, 2, j + 1)
        ax.text(0.0, 1.06, 'ab'[j], transform=ax.transAxes, fontsize=26, fontweight='bold', va='top')
        for s in SEASONS:
            lo, hi = np.nanpercentile(res[s]['boot'][j], [2.5, 97.5], axis=0)
            ax.fill_betweenx(depth, lo, hi, alpha=0.3, color=COLORS[s], lw=0)
            ax.plot(res[s]['new'][j], depth, color=COLORS[s], lw=3, label=LABELS[s])
            ax.plot(res[s]['old'][j], depth, color=COLORS[s], lw=1.5, alpha=0.5)
        ax.set_ylim([500, 15])
        ax.set_xlim(xlim)
        ax.set_yticks([100, 200, 300, 400, 500])
        ax.set_xlabel(xlabel, fontsize=28)
        ax.set_ylabel('Depth [m]', fontsize=28)
        ax.grid()
        if j == 0:
            ax.legend(loc='upper left', fontsize=20)
        else:
            ax.legend(handles=[Line2D([0], [0], color='k', lw=3, label=r'w(15 m) = w$_m$'),
                               Line2D([0], [0], color='k', lw=1.5, alpha=0.5, label='w(15 m) = 0')],
                      loc='upper right', fontsize=18)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    print('saved', out)
