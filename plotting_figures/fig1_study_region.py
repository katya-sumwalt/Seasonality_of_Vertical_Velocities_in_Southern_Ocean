'''Figure 1: study region and sampling.
  a: time-mean NeurOST strain |sigma| = sqrt(sn^2 + ss^2) over the period of the profiles (cached in config.STRAIN_CACHE),
     RTopo-2 bathymetry contours, seal tracks by season and profile counts;
  b: number of profiles per bin (config.ALL_PROFILES_FILE), with the region of interest on an orthographic Earth;
  c: profiles per month; d: observations vs depth.

usage: python -m plotting_figures.fig1_study_region
'''

import os
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import cmocean
import config

EXTENT = [67, 117, -56.7, -45.7]
BAD_SEALS = ['ct131-035BAT2-15', 'ct139-624-17']   # vertical velocity not computed
SEASONS = config.SEASONS
COLORS = {'summer': 'tab:blue', 'fall': 'tab:orange', 'winter': 'tab:green', 'spring': 'tab:red'}
LABELS = {'summer': 'Summer', 'winter': 'Winter', 'spring': 'Spring', 'fall': 'Fall'}

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def load_season(season):
    ds = xr.open_dataset(config.season_file(season))[['lon', 'lat', 'time', 'name', 'maskprof']]
    return ds.isel(distance=~ds.name.isin(BAD_SEALS).values).load()


def strain_mean(start, end):
    '''Time mean of |sigma| over the map domain from the daily NeurOST files between start and end (cached).'''
    if os.path.exists(config.STRAIN_CACHE):
        return xr.open_dataarray(config.STRAIN_CACHE)
    files = [f for f in sorted(glob.glob(config.NEUROST_GLOB)) if start <= os.path.basename(f).split('_')[2] <= end]
    total, count = 0, 0
    for f in files:
        d = xr.open_dataset(f)[['sn', 'ss']].sel(latitude=slice(EXTENT[2] - 0.5, EXTENT[3] + 0.5),
                                                 longitude=slice(EXTENT[0] - 0.5, EXTENT[1] + 0.5)).isel(time=0)
        sigma = np.sqrt(d.sn ** 2 + d.ss ** 2).load()
        total = total + sigma.fillna(0)
        count = count + sigma.notnull()
    n = len(files)
    mean = (total / count.where(count > 0)).rename('strain')   # mean over the days with data at each point
    mean.attrs.update(description=f'time mean of sqrt(sn^2 + ss^2), NeurOST, {n} days {start}-{end}', units='s-1')
    mean.to_netcdf(config.STRAIN_CACHE)
    return mean


def bathymetry():
    t = xr.open_dataset(config.RTOPO_FILE)
    lat, lon = t.lat.values, t.lon.values
    j = np.where((lat >= EXTENT[2] - 0.5) & (lat <= EXTENT[3] + 0.5))[0]
    i = np.where((lon >= EXTENT[0] - 0.5) & (lon <= EXTENT[1] + 0.5))[0]
    z = t.bedrock_topography.isel(latdim=slice(j[0], j[-1] + 1, 2), londim=slice(i[0], i[-1] + 1, 2)).values
    return lon[i[0]:i[-1] + 1:2], lat[j[0]:j[-1] + 1:2], z


def map_axes(ax, bathy):
    ax.contour(*bathy, levels=[-1500, -1000, -500], linestyles='-', colors='black', linewidths=0.5,
               transform=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.set_aspect(2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def count_box(ax, data):
    '''Profile counts per season in a framed box (upper right of the panel).'''
    n = {s: data[s].sizes['distance'] for s in SEASONS}
    total = sum(n.values())
    lines = [TextArea(f'{LABELS[s]} Profiles: {n[s]:,} / {total:,} - {100 * n[s] / total:.1f} % of total profiles',
                      textprops=dict(color=COLORS[s], fontsize=8.5)) for s in ['summer', 'winter', 'spring', 'fall']]
    box = AnchoredOffsetbox(loc='upper right', child=VPacker(children=lines, align='center', pad=0, sep=3),
                            pad=0.4, borderpad=0.6, frameon=True)
    box.patch.set(edgecolor='black', linewidth=1.5, facecolor='white')
    ax.add_artist(box)


def globe_inset(ax):
    '''Region of interest (the map domain) as a polygon on an orthographic Earth.'''
    lon0, lat0 = np.mean(EXTENT[:2]), np.mean(EXTENT[2:])
    # centred in the empty (no profiles) north-east part of panel b
    sub = inset_axes(ax, width='100%', height='100%', loc='center', bbox_to_anchor=(0.65, 0.29, 0.30, 0.70),
                     bbox_transform=ax.transAxes, borderpad=0, axes_class=GeoAxes,
                     axes_kwargs=dict(projection=ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0 + 10)))
    sub.set_global()
    sub.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    sub.add_feature(cfeature.LAND, facecolor='lightgrey')
    sub.coastlines(linewidth=0.4)
    sub.gridlines(linewidth=0.3, color='gray', alpha=0.6)
    lon = np.concatenate([np.linspace(EXTENT[0], EXTENT[1], 50), np.full(50, EXTENT[1]),
                          np.linspace(EXTENT[1], EXTENT[0], 50), np.full(50, EXTENT[0])])
    lat = np.concatenate([np.full(50, EXTENT[2]), np.linspace(EXTENT[2], EXTENT[3], 50),
                          np.full(50, EXTENT[3]), np.linspace(EXTENT[3], EXTENT[2], 50)])
    sub.fill(lon, lat, facecolor='tab:green', alpha=0.35, edgecolor='tab:green', linewidth=1.2, transform=ccrs.PlateCarree())
    sub.spines['geo'].set_linewidth(0.8)
    return sub


if __name__ == '__main__':
    out = config.out_path('fig1_study_region')
    data = {s: load_season(s) for s in SEASONS}
    t0 = min(data[s].time.min().values for s in SEASONS)
    t1 = max(data[s].time.max().values for s in SEASONS)
    strain = strain_mean(str(t0)[:10].replace('-', ''), str(t1)[:10].replace('-', ''))
    print(strain.attrs['description'])
    bathy = bathymetry()
    raw = xr.open_dataset(config.ALL_PROFILES_FILE)[['lon', 'lat']].load()

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.25, hspace=0.06, rect=(0, 0, 1, 0.98))
    gs = gspec.GridSpec(ncols=2, nrows=3, width_ratios=[4, 2.2], height_ratios=[6, 6, 6], figure=fig)

    # a: mean strain, bathymetry and tracks, with the profile counts
    ax0 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    label_a = ax0.text(0, 1.02, 'a', transform=ax0.transAxes, fontsize=15, fontweight='bold', clip_on=False, va='bottom')
    pcm = ax0.pcolormesh(strain.longitude, strain.latitude, strain, transform=ccrs.PlateCarree(), cmap=cmocean.cm.dense,
                         shading='auto', vmin=0, vmax=1.5e-5, alpha=0.9)
    cb = plt.colorbar(pcm, ax=ax0, shrink=0.9, ticks=np.arange(0, 1.51e-5, 0.25e-5))
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.ax.text(0.5, 1.01, r'$\times 10^{-5}$', transform=cb.ax.transAxes, fontsize=8, va='bottom', ha='center')
    cb.set_label(r' $\sigma$ [s$^{-1}$]', fontsize=15)
    for s in ['summer', 'spring', 'winter', 'fall']:
        ax0.scatter(data[s].lon, data[s].lat, s=0.2, color=COLORS[s], transform=ccrs.PlateCarree())
    map_axes(ax0, bathy)
    count_box(ax0, data)

    # b: number of profiles per bin, with the region of interest on a sphere
    ax1 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
    label_b = ax1.text(0, 1.02, 'b', transform=ax1.transAxes, fontsize=15, fontweight='bold', clip_on=False, va='bottom')
    counts, lon_edges, lat_edges = np.histogram2d(raw.lon, raw.lat, bins=[np.linspace(EXTENT[0], EXTENT[1], 18),
                                                                          np.linspace(EXTENT[2], EXTENT[3], 10)])
    pcm = ax1.pcolormesh(lon_edges, lat_edges, counts.T, transform=ccrs.PlateCarree(), cmap=cmocean.cm.tempo,
                         norm=plt_colors.LogNorm(vmin=10, vmax=10000))
    cb = plt.colorbar(pcm, ax=ax1, shrink=0.9)
    cb.set_label('# of Profiles', fontsize=15)
    map_axes(ax1, bathy)
    globe_inset(ax1)

    # c: profiles per month
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('c', loc='left', fontsize=15, fontweight='bold')
    months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    season_of = {12: 'summer', 1: 'summer', 2: 'summer', 3: 'fall', 4: 'fall', 5: 'fall',
                 6: 'winter', 7: 'winter', 8: 'winter', 9: 'spring', 10: 'spring', 11: 'spring'}
    per_month = [int((data[season_of[m]].time.dt.month == m).sum()) for m in months]
    ax3.bar(np.arange(1, 13), per_month, color=[COLORS[season_of[m]] for m in months], edgecolor='black', width=0.8)
    ax3.set_xticks(np.arange(1, 13), ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'])
    ax3.set_xlim([0, 13])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_visible(False)
    ax3.text(-0.06, 1.02, r'$\times 10^{4}$', transform=ax3.transAxes, fontsize=8, va='top')
    ax3.grid()
    ax3.set_ylabel('# of Profiles', fontsize=15)

    # d: observations vs depth
    ax2 = fig.add_subplot(gs[2, 1])
    ax2.set_title('d', loc='left', fontsize=15, fontweight='bold')
    for s in ['summer', 'spring', 'winter', 'fall']:
        mask = data[s].maskprof
        ax2.plot(mask.count('distance'), -mask.depth, color=COLORS[s])
    ax2.set_xscale('log')
    ax2.set_xlabel('# of Observations', fontsize=15)
    ax2.set_ylabel('Depth [m]', fontsize=15)
    ax2.set_xlim([1e2, 2.5e4])
    ax2.set_ylim([500, 15])
    ax2.grid(True)

    # align panels c and d with the map frames above (left edge of c, right edge of d), keeping their gap and ratio
    fig.canvas.draw()
    fig.set_layout_engine('none')
    p0, p3, p2 = ax0.get_position(), ax3.get_position(), ax2.get_position()
    gap = p2.x0 - p3.x1
    w3 = (p0.width - gap) * p3.width / (p3.width + p2.width)
    ax3.set_position([p0.x0, p3.y0, w3, p3.height])
    ax2.set_position([p0.x0 + w3 + gap, p2.y0, p0.x1 - (p0.x0 + w3 + gap), p2.height])

    for ext in ('png', 'pdf'):
        # cartopy axes leave their texts out of the tight bbox, so the panel labels are passed explicitly
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight', pad_inches=0.05,
                    bbox_extra_artists=[label_a, label_b])
    print('profiles per season:', {s: data[s].sizes['distance'] for s in SEASONS})
    print('saved', out)
