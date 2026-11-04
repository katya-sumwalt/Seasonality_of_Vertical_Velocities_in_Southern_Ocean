'''Seasonal statistics of w and vertical heat transport (VHT): 95% confidence intervals, PDFs, conditional means,
layer time series and their calendar climatology, and L-kurtosis.

w is used with the upper boundary condition w(15 m) = w_m (Ponte et al. 2013, eq. 5): w_stored_new = vertical_velocity - dw,
with dw from computing_vertical_velocities/compute_dw_mixing_bc.py (panel-e convention, w positive upward).
'''

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t
import config

RCP = 4e6     # rho * cp [J m-3 K-1]
DAY = 86400


def load_season(season):
    ds = xr.open_dataset(config.season_file(season))
    return ds[['temperature', 'buoyancy', 'vertical_velocity', 'maskprof', 'strain_neuro']].load()


def load_dw(season, case=None):
    '''dw for w(15 m) = w_m (case None) or for the depth-dependent strain case A, B or C (panel-e convention).'''
    if case is None:
        return xr.open_dataset(os.path.join(config.DW_WM_DIR, f'dw_{season}.nc')).dw
    return xr.open_dataset(os.path.join(config.DW_STRAIN_DIR, f'dw_{case}_{season}.nc')).dw


def calculate_vht(ds):
    '''Vertical heat transport rho*cp*T'*w' [W m-2], anomalies relative to the mean over distance.'''
    w = -1 * ds.vertical_velocity
    T_prime = ds.temperature - ds.temperature.mean(dim='distance')
    w_prime = w - w.mean(dim='distance')
    return RCP * T_prime * w_prime


###### STATISTICS ################################################################################
def tcrit(n):
    return xr.apply_ufunc(lambda x: t.ppf(0.975, x - 1), n)


def rms_ci(da, dim='distance'):
    sq = da ** 2
    n = sq.count(dim)
    rms = np.sqrt(sq.mean(dim))
    return rms.values, (0.5 * tcrit(n) * sq.std(dim) / np.sqrt(n) / rms).values


def mean_ci(da, dim='distance'):
    n = da.count(dim)
    return da.mean(dim).values, (tcrit(n) * da.std(dim) / np.sqrt(n)).values


def rms_over_all_dims(da):
    sq = da ** 2
    n = sq.count()
    rms = np.sqrt(sq.mean())
    return float(rms), float(0.5 * tcrit(n) * sq.std() / np.sqrt(n) / rms)


def mean_over_all_dims(da):
    n = da.count()
    return float(da.mean()), float(tcrit(n) * da.std() / np.sqrt(n))


def pdf(da):
    x = da.values.ravel()
    return np.histogram(x[np.isfinite(x)], bins=100, density=True)


def conditional(da):
    '''Mean and 95% CI over distance of the positive and negative values, and their counts.'''
    pos, neg = da.where(da > 0), da.where(da < 0)
    return {'pos': mean_ci(pos), 'neg': mean_ci(neg), 'n_pos': int(pos.count()), 'n_neg': int(neg.count())}


def layer_series(vht, top, bottom):
    '''Per-profile depth mean of VHT over a layer and its 95% CI across depth.'''
    layer = vht.sel(depth=slice(-top, -bottom))
    mean, ci = mean_ci(layer, dim='depth')
    time = pd.to_datetime(vht.time.values)
    return pd.Series(mean, index=time), pd.Series(ci, index=time)


def version_stats(ds, w_stored):
    '''Statistics of Figures 3, S3 and S9 from one w field (stored sign convention) on dataset ds.'''
    mask = ds.maskprof
    w_fig = DAY * w_stored * mask                      # stored sign convention, as in the first submission
    w_up = -1 * w_stored
    w_prime = w_up - w_up.mean('distance')
    vht = RCP * (ds.temperature - ds.temperature.mean('distance')) * w_prime
    vht_m = vht * mask
    wb = w_prime * (ds.buoyancy - ds.buoyancy.mean('distance')) * mask
    vht_ts = vht.where(mask.notnull()).assign_coords(time=ds.time)
    return {'w_pdf': pdf(w_fig), 'w_rms': rms_ci(w_fig),
            'vht_pdf': pdf(vht_m), 'vht_rms': rms_ci(vht_m), 'vht_mean': mean_ci(vht_m),
            'vht_cond': conditional(vht_m), 'wb_cond': conditional(wb), 'wb_mean': mean_ci(wb),
            'ts_upper': layer_series(vht_ts, 15, 100), 'ts_deep': layer_series(vht_ts, 200, 500)}


def season_stats(season):
    '''Statistics for one season with w(15 m) = w_m ('new') and w(15 m) = 0 ('old').'''
    ds = load_season(season)
    dw = load_dw(season).load()
    return {'strain_rms': rms_over_all_dims(ds.strain_neuro), 'strain_mean': mean_over_all_dims(ds.strain_neuro),
            'old': version_stats(ds, ds.vertical_velocity), 'new': version_stats(ds, ds.vertical_velocity - dw)}


def climatology(series):
    '''Daily calendar climatology (Jun-May), 30-day centred running mean.'''
    df = pd.DataFrame(series).sort_index()
    daily = df.groupby(df.index.normalize()).mean()
    clim = daily.groupby(daily.index.strftime('%m-%d')).mean().drop(index='02-29', errors='ignore')
    clim.index = pd.to_datetime('2001-' + clim.index, format='%Y-%m-%d')
    clim = clim.reindex(pd.date_range('2001-01-01', '2001-12-31', freq='D'))
    jun_dec = clim.loc['2001-06-01':'2001-12-31'].set_axis(pd.date_range('2000-06-01', '2000-12-31', freq='D'))
    jan_may = clim.loc['2001-01-01':'2001-05-31']
    clim = pd.concat([jun_dec, jan_may])
    return clim.fillna(0).rolling(window=30, center=True, min_periods=1).mean().mask(clim.isna())


def l_kurtosis(x):
    '''L-kurtosis tau_4 = l4 / l2 along axis 1 of x (depth x samples) from unbiased probability-weighted moments
    (Hosking 1990; identical to lmoments3.lmom_ratios), ignoring NaNs; NaN where fewer than 4 values.'''
    x = np.sort(x, axis=1)                       # NaNs last
    n = np.sum(np.isfinite(x), 1, keepdims=True).astype(float)
    i = np.arange(x.shape[1])[None, :].astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = [np.sum(x0 * w, 1) / n[:, 0] for w in (np.ones_like(i),
                                                     i / (n - 1),
                                                     i * (i - 1) / ((n - 1) * (n - 2)),
                                                     i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)))]
        l2 = 2 * b[1] - b[0]
        l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
        return np.where(n[:, 0] >= 4, l4 / l2, np.nan)
