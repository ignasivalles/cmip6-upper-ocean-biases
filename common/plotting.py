"""Map helpers shared by the figure scripts."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.util import add_cyclic_point
from scipy import stats

ROBINSON = ccrs.Robinson(central_longitude=-150)
PC = ccrs.PlateCarree()


def shift_lon(da):
    """0-360 to -180-180."""
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")


def global_map(ax, data, vmin, vmax, cmap):
    da = shift_lon(data)
    im = ax.pcolormesh(da.lon, da.lat, da, transform=PC, vmin=vmin, vmax=vmax,
                       cmap=cmap, rasterized=True)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.72", zorder=5)
    ax.coastlines(linewidth=0.35, zorder=6, color="0.3")
    return im


def _axes_frac(ax, proj, lon, lat):
    xy = proj.transform_point(lon, lat, PC)
    return ax.transAxes.inverted().transform(ax.transData.transform(xy))


def geo_labels(ax, proj=ROBINSON, lat=True, lon=True,
               lats=(-60, 0, 60), lons=(-120, 0, 120), fontsize=8):
    """Latitude labels on the left edge and longitude labels below a global map.
    Call after fig.canvas.draw()."""
    lon_0 = proj.proj4_params.get("lon_0", 0)
    if lat:
        for la in lats:
            text = "0°" if la == 0 else f"{abs(la)}°{'N' if la > 0 else 'S'}"
            y = _axes_frac(ax, proj, lon_0, la)[1]
            x_left = 1.0 - _axes_frac(ax, proj, lon_0 + 180, la)[0]
            ax.text(x_left - 0.02, y, text, transform=ax.transAxes,
                    ha="right", va="center", fontsize=fontsize)
    if lon:
        for lo in lons:
            text = "0°" if lo == 0 else f"{abs(lo)}°{'E' if lo > 0 else 'W'}"
            x = _axes_frac(ax, proj, lo, -88)[0]
            ax.text(x, -0.03, text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=fontsize)


def polar_map(ax, field, levels, cmap="RdBu_r", lat_max=-35, pvalue=None, p_threshold=0.05,
              stride=5):
    """Filled contours on a South Polar Stereographic axis, with optional
    stippling where pvalue < p_threshold."""
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax.set_extent([-180, 180, -90, lat_max], PC)
    ax.set_boundary(circle, transform=ax.transAxes)

    values, lon = add_cyclic_point(field.values, coord=field.lon.values)
    im = ax.contourf(lon, field.lat.values, values, levels=levels, cmap=cmap,
                     extend="both", antialiased=False, transform=PC)

    if pvalue is not None:
        lo, la = np.meshgrid(field.lon.values[::stride], field.lat.values[::stride])
        sig = pvalue.values[::stride, ::stride] < p_threshold
        ax.plot(lo[sig], la[sig], "k.", ms=2, transform=PC, zorder=6, linestyle="none")

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="darkgray", zorder=99)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.xlocator = mticker.FixedLocator([])
    gl.ylocator = mticker.FixedLocator([-60, -40])
    for la in (-60, -40):
        ax.text(0, la, f"{abs(la)}°S", transform=PC, ha="left", va="center", fontsize=8,
                color="0.2", clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
    return im


def corr_triangle(ax, df, labels, fontsize=10):
    """Lower-triangle Pearson correlation matrix of the columns of df.
    Asterisks mark p < 0.05."""
    n = df.shape[1]
    r = np.eye(n)
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            r[i, j], p[i, j] = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_where(np.triu(np.ones((n, n), bool), k=1), r),
                   cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            star = "*" if i == j or p[i, j] < 0.05 else ""
            ax.text(j, i, f"{r[i, j]:.2f}{star}", ha="center", va="center", fontsize=fontsize,
                    color="white" if abs(r[i, j]) > 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im

