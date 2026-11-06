'''Block bootstrap of the Figure 3 statistics with summer, winter and spring downsampled to the fall sampling
(number of observations vs depth, Fig. 1d) (Figures S6, S7 and Table S2).

Each iteration (i) block-bootstraps every season: blocks of BLOCK contiguous profiles (same tag, 1 km step) are drawn with
replacement until the season has its original number of blocks (w' and w'CT' decorrelate over ~5 km along track, so
single profiles are not independent); (ii) downsamples summer, winter and spring from that bootstrap sample with the
whole-profile matching of select_profiles (from 500 m upward, profiles ending at each level are drawn until the number
reaching that depth equals the fall count). Fall is only block-bootstrapped. The spread over iterations thus contains
both the sampling uncertainty and the choice of retained profiles. Anomalies are relative to the mean of each resample.

w = w_stored(w(15 m) = 0) + dw_wm [+ dw_case] (linear superposition), positive upward.

usage: python -m analysis.bootstrap [--bc wm|zero] [--case A|B|C] [--n-iter N]
output: config.OUT_DIR/tableS2_bootstrap_layers.npz and the Table S2 numbers printed
'''

import argparse
import warnings
import numpy as np
import pandas as pd
import config
from analysis.statistics import RCP, DAY, load_season, load_dw, climatology

SEASONS = config.SEASONS
N_ITER = 500
BLOCK = 5
SEED = 0
W_BINS = np.linspace(-150, 150, 101)
VHT_BINS = np.linspace(-1.4e4, 1.4e4, 101)
LAYERS = {'15-100 m': (15, 100), '200-500 m': (200, 500)}
DEPTH = np.arange(15, 501)

warnings.filterwarnings('ignore', category=RuntimeWarning)


###### DOWNSAMPLING TO FALL ######################################################################
def profile_length(ds):
    '''Number of valid levels per profile, and whether the valid levels are contiguous from the top.'''
    m = np.isfinite(ds.maskprof.values)
    n = m.sum(0)
    contiguous = np.all(m == (np.arange(m.shape[0])[:, None] < n[None, :]), axis=0)
    return n, contiguous


def select_profiles(n, contiguous, n_target, rng):
    '''Indices of profiles whose cumulative count with depth matches the target (fall) counts.'''
    target_reaching = np.array([np.sum(n_target >= k) for k in range(1, n.max() + 2)])   # profiles reaching level k
    chosen = []
    for k in range(n.max(), 0, -1):
        need = target_reaching[k - 1] - len(chosen)
        pool = np.where((n == k) & contiguous)[0]
        if need > 0 and pool.size:
            chosen += list(rng.choice(pool, size=min(need, pool.size), replace=False))
    return np.sort(np.array(chosen))


###### BOOTSTRAP #################################################################################
def w_up(ds, season, bc='wm', case=None):
    '''w positive upward for the chosen boundary condition and strain case (linear superposition).'''
    w = -ds.vertical_velocity
    if bc == 'wm':
        w = w + load_dw(season)
    if case:
        w = w + load_dw(season, case)
    return w.transpose(*ds.vertical_velocity.dims).values


def arrays(ds, season, bc='wm', case=None):
    '''numpy fields (depth x profile), w positive upward, and profile lengths.'''
    valid = np.isfinite(ds.maskprof.values)
    n = valid.sum(0)
    return {'w': w_up(ds, season, bc, case), 'T': ds.temperature.values, 'b': ds.buoyancy.values,
            'strain': ds.strain_neuro.values, 'valid': valid, 'n': n, 'name': ds.name.values, 'dist': ds.distance.values,
            'time': pd.to_datetime(ds.time.values)}


def blocks(a):
    '''Non-overlapping blocks of <= BLOCK contiguous profiles with at least one valid level.'''
    name, dist = a['name'], a['dist']
    breaks = np.where((name[1:] != name[:-1]) | (np.diff(dist) != 1000))[0] + 1
    edges = np.concatenate([[0], breaks, [name.size]])
    idx = [np.arange(j0, min(j0 + BLOCK, i1)) for i0, i1 in zip(edges[:-1], edges[1:]) for j0 in range(i0, i1, BLOCK)]
    idx = [i[a['n'][i] > 0] for i in idx]
    return [i for i in idx if i.size]


def draw(a, block_idx, n_target, rng):
    '''Profile indices of one block bootstrap, downsampled to the fall counts if n_target is given.'''
    idx = np.concatenate([block_idx[i] for i in rng.integers(0, len(block_idx), len(block_idx))])
    if n_target is not None:
        n = a['n'][idx]
        idx = idx[select_profiles(n, np.ones(n.size, bool), n_target, rng)]
    return idx, a['n'][idx]


def statistics(a, idx, length):
    '''Figure 3 statistics for the profiles idx, each truncated to its first `length` levels.'''
    valid = a['valid'][:, idx] & (np.arange(a['valid'].shape[0])[:, None] < length[None, :])   # whole profiles here
    # anomalies are relative to the mean of all finite values (also outside maskprof), then masked
    w_p, T_p, b_p = (np.where(valid, x - np.nanmean(x, 1, keepdims=True), np.nan)
                     for x in (a[k][:, idx] for k in ('w', 'T', 'b')))
    w = np.where(valid, a['w'][:, idx], np.nan)
    vht = RCP * T_p * w_p
    wb = w_p * b_p
    pos, neg = np.where(vht > 0, vht, np.nan), np.where(vht < 0, vht, np.nan)
    out = {'w_rms': np.sqrt(np.nanmean((w * DAY) ** 2, 1)), 'vht_rms': np.sqrt(np.nanmean(vht ** 2, 1)),
           'vht_mean': np.nanmean(vht, 1), 'vht_pos': np.nanmean(pos, 1), 'vht_neg': np.nanmean(neg, 1),
           'wb_pos': np.nanmean(np.where(wb > 0, wb, np.nan), 1), 'wb_neg': np.nanmean(np.where(wb < 0, wb, np.nan), 1),
           'wb_mean': np.nanmean(wb, 1), 'n_pos': np.sum(vht > 0), 'n_neg': np.sum(vht < 0),
           'n_wb_pos': np.sum(wb > 0), 'n_wb_neg': np.sum(wb < 0), 'counts': valid.sum(1)}
    s = a['strain'][idx]   # one value per profile
    out['strain_rms'], out['strain_mean'] = np.sqrt(np.nanmean(s ** 2)), np.nanmean(s)
    for key, x, bins in [('w_pdf', w[valid] * DAY, W_BINS), ('vht_pdf', vht[valid], VHT_BINS)]:
        x = x[np.isfinite(x)]
        out[key] = np.histogram(x, bins)[0] / (x.size * np.diff(bins))
    for name, (top, bottom) in LAYERS.items():
        v = vht[top - 15:bottom - 15 + 1]
        v = v[np.isfinite(v)]
        p_up = np.mean(v > 0)
        out[f'layer {name}'] = v.mean()
        out[f'P+ {name}'], out[f'V+ {name}'], out[f'V- {name}'] = p_up, v[v > 0].mean(), v[v < 0].mean()
        # Fig. 3g: per-profile depth mean of VHT over the layer (as layer_series)
        out[f'ts {name}'] = pd.Series(np.nanmean(vht[top - 15:bottom - 15 + 1], 0), index=a['time'][idx])
    return out


def bootstrap(a, n_target, rng, n_iter=N_ITER):
    block_idx = blocks(a)
    runs = [statistics(a, *draw(a, block_idx, n_target, rng)) for _ in range(n_iter)]
    return {k: [r[k] for r in runs] if k.startswith('ts ') else np.array([r[k] for r in runs]) for k in runs[0]}


def climatologies(boot, full, n_iter=N_ITER):
    '''Fig. 3g calendar climatology of the layer <VHT> (all seasons together) for every resample and for all profiles.'''
    def clim(stats):
        return climatology({name: pd.concat([stats[s][f'ts {name}'] for s in SEASONS]) for name in LAYERS})
    full_clim = clim(full)
    runs = [clim({s: {f'ts {n}': boot[s][f'ts {n}'][i] for n in LAYERS} for s in SEASONS}) for i in range(n_iter)]
    return full_clim, {name: np.array([r[name].values for r in runs]) for name in LAYERS}


def counts_by_depth(a, idx, length):
    return (a['valid'][:, idx] & (np.arange(a['valid'].shape[0])[:, None] < length[None, :])).sum(1)


def suffix(bc='wm', case=None):
    return ('' if bc == 'wm' else '_w0') + (f'_strain{case}' if case else '')


###### RUN #######################################################################################
def run(bc='wm', case=None, n_iter=N_ITER):
    '''Bootstrap all seasons, print the Table S2 numbers and save them; returns (boot, full, climatologies).'''
    rng = np.random.default_rng(SEED)
    data = {s: arrays(load_season(s), s, bc, case) for s in SEASONS}
    print('w: BC', bc, '| strain case', case or 'F = 1', '|', n_iter, 'resamples', flush=True)
    n_fall = data['fall']['n'][data['fall']['n'] > 0]
    fall_counts = data['fall']['valid'].sum(1)

    full, boot = {}, {}
    for s in SEASONS:
        a = data[s]
        all_idx = np.where(a['n'] > 0)[0]
        full[s] = statistics(a, all_idx, a['n'][all_idx])
        target = None if s == 'fall' else n_fall
        diff = counts_by_depth(a, *draw(a, blocks(a), target, np.random.default_rng(1))) - fall_counts
        print(f'{s:7s}: {len(blocks(a))} blocks; example draw obs/depth minus fall: max {diff.max()}, min {diff.min()}',
              flush=True)
        boot[s] = bootstrap(a, target, rng, n_iter)

    print(f'\nlayer-mean <VHT> [W/m2]: all profiles | resampled median [2.5, 97.5]')
    for name in LAYERS:
        for s in SEASONS:
            lo, med, hi = np.percentile(boot[s][f'layer {name}'], [2.5, 50, 97.5])
            print(f"  {name:9s} {s:7s} {full[s][f'layer {name}']:7.1f} | {med:7.1f} [{lo:6.1f}, {hi:6.1f}]")
        x = {s: boot[s][f'layer {name}'] for s in SEASONS}
        for s in ['spring', 'summer', 'fall']:
            print(f'  {name}: fraction of resamples with winter > {s}: {np.mean(x["winter"] > x[s]):.3f}')
    for s in ['spring', 'summer', 'fall']:
        p = np.mean(boot['winter']['vht_mean'] > boot[s]['vht_mean'], axis=0)   # per depth, over resamples
        deep = DEPTH >= 200
        print(f'  winter > {s}: P > 0.95 at {100 * np.mean(p > 0.95):.0f}% of depths 15-500 m, '
              f'{100 * np.mean(p[deep] > 0.95):.0f}% of depths 200-500 m (median P {np.median(p[deep]):.2f})')
    name = '200-500 m'
    print(f'\nfrequency/strength, {name}: median [2.5, 97.5]')
    for s in SEASONS:
        V_p, V_m, P = boot[s][f'V+ {name}'], boot[s][f'V- {name}'], boot[s][f'P+ {name}']
        for label, x in [('P+', P), ('strength', 0.5 * (V_p + V_m)), ('frequency', 0.5 * (2 * P - 1) * (V_p - V_m))]:
            lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
            print(f'  {s:7s} {label:9s} {med:7.3f} [{lo:7.3f}, {hi:7.3f}]')

    keys = [f'{k} {name}' for name in LAYERS for k in ('layer', 'P+', 'V+', 'V-')]
    np.savez(config.out_path('tableS2_bootstrap_layers' + suffix(bc, case) + '.npz'),
             **{f'{s}|boot|{k}': np.asarray(boot[s][k]) for s in SEASONS for k in keys + ['vht_mean']},
             **{f'{s}|full|{k}': np.asarray(full[s][k]) for s in SEASONS for k in keys + ['vht_mean']})
    for s in SEASONS:
        d = boot[s]['counts'] - full['fall']['counts']
        print(f'{s:7s} obs/depth minus fall over resamples: min {d.min()}, max {d.max()}, '
              f'median |diff| {np.median(np.abs(d)):.0f}; deepest-level shortfall depths (median) '
              f'{np.sum(np.median(d, 0) < -10)} levels < -10')
    clim = climatologies(boot, full, n_iter)
    for name in LAYERS:
        x = clim[1][name]
        for s, (start, end) in {'winter': ('2000-06-01', '2000-08-31'), 'spring': ('2000-09-01', '2000-11-30'),
                                'summer': ('2000-12-01', '2001-02-28'), 'fall': ('2001-03-01', '2001-05-31')}.items():
            sel = (clim[0].index >= start) & (clim[0].index <= end)
            m = np.nanmean(x[:, sel], 1)
            print(f'  3g {name} {s:7s} seasonal mean of climatology: all {np.nanmean(clim[0][name].values[sel]):6.1f} | '
                  f'resampled {np.median(m):6.1f} [{np.percentile(m, 2.5):6.1f}, {np.percentile(m, 97.5):6.1f}]')
    return boot, full, clim


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--bc', choices=['wm', 'zero'], default='wm', help='upper boundary condition w(15 m)')
    parser.add_argument('--case', choices=['A', 'B', 'C'], default=None, help='depth-dependent strain case')
    parser.add_argument('--n-iter', type=int, default=N_ITER, help='number of bootstrap resamples')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(__doc__.splitlines()[0])
    run(args.bc, args.case, args.n_iter)
