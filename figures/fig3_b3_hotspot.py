"""Figure 3: ensemble-mean b3 bias with the Indo-Pacific hotspot, zonal-mean b3 bias
by model, and partial correlations of Southern Ocean biases with PC1."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from scipy.ndimage import uniform_filter1d

from common.data import SO_EXCLUDE, area_mean, basin_masks, load_ensemble, samw_mask
from common.eof import ensemble_mean_b3, hotspot_masks, inter_model_eof, partial_corr_with_field
from common.paths import FIG_DIR
from common.plotting import PC, ROBINSON, geo_labels, polar_map

P_THRESHOLD = 0.10
# models whose wind-stress output is unreliable in the Southern Ocean
TAUU_EXCLUDE = ["CMCC-CM2-SR5", "FIO-ESM-2-0"]
LEVELS = np.linspace(-0.8, 0.8, 17)


def jet_path(tauu):
    """Latitude of maximum tau_x at each longitude (45-70S), smoothed."""
    lat = tauu.where((tauu.lat >= -70) & (tauu.lat <= -45)).idxmax("lat")
    return lat.lon.values, uniform_filter1d(lat.values, size=15, mode="wrap")


def compute(ds):
    masks = basin_masks(ds)
    pac, ind = hotspot_masks(ds, masks)
    b3 = ds["bias_b3"].mean("month")
    pcs, _, var = inter_model_eof(b3, pac | ind)
    pc1 = pcs[0]
    print(f"EOF1: {var[0] * 100:.1f}% of inter-model variance")

    sst = ds["bias_sst"].mean("month")
    sss = ds["bias_sos"].mean("month")
    tauu = ds["bias_tauu"].mean("month")

    # drop models with Southern Ocean wind-stress bias beyond 2 sigma
    so_tau = tauu.where((ds.lat >= -70) & (ds.lat <= -30)).mean(("lat", "lon"))
    ok = [m for m in so_tau.model.values
          if abs(float(so_tau.sel(model=m) - so_tau.mean())) <= 2 * float(so_tau.std())
          and m not in TAUU_EXCLUDE]
    print(f"tau_x panel: {len(ok)} of {ds.model.size} models")

    # control: each model's area-mean bias north of 35S
    north = ds.lat > -35
    out = {
        "sst": partial_corr_with_field(sst, pc1, area_mean(sst, north)),
        "sss": partial_corr_with_field(sss, pc1, area_mean(sss, north)),
        "tauu": partial_corr_with_field(tauu.sel(model=ok), pc1.sel(model=ok),
                                        area_mean(tauu.sel(model=ok), north)),
        "jet_obs": jet_path(ds["tauu_obs"].mean("month")),
        "jet_model": jet_path(ds["tauu"].sel(model=ok).median(("month", "model"))),
        "b3_mean": ensemble_mean_b3(ds), "pac": pac, "ind": ind,
    }
    return out


def make_figure(ds, res, samw):
    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.3, 0.9], hspace=0.25)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1, 1.5],
                                           wspace=0.2)

    # (a) ensemble-mean b3 bias and hotspot contours
    ax1 = fig.add_subplot(top[0], projection=ROBINSON)
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, facecolor="darkgray", zorder=99)
    ax1.coastlines(zorder=99)
    b3 = res["b3_mean"]
    im_a = ax1.pcolormesh(b3.lon, b3.lat, b3 * 1e3, transform=PC, cmap="coolwarm",
                          vmin=-2, vmax=2, rasterized=True)
    for mask, color in [(res["pac"], "b"), (res["ind"], "g")]:
        m = mask.reindex_like(b3, fill_value=False)
        ax1.contour(m.lon, m.lat, m.astype(float), levels=[0.5], transform=PC, colors=color,
                    linewidths=1.2)
    ax1.legend(handles=[mlines.Line2D([], [], color="b", label="Pacific"),
                        mlines.Line2D([], [], color="g", label="Indian")],
               loc="lower right", bbox_to_anchor=(1.1, -0.40), frameon=True, fontsize=12)
    ax1.set_title(rf"$\bf{{a)}}$ Global $b_3$ bias (n={ds.model.size})", fontsize=14)

    # (b) zonal-median b3 bias by model
    ax2 = fig.add_subplot(top[1])
    zonal = ds["bias_b3"].median("month").median("lon")
    models = zonal.model.values
    ax2.pcolormesh(np.arange(len(models)), zonal.lat, zonal.T * 1e3, cmap="coolwarm",
                   vmin=-2, vmax=2, rasterized=True)
    peak = zonal.where(zonal.lat < -10).argmax("lat")
    ax2.plot(np.arange(len(models)), zonal.lat[peak], "--k", lw=2)
    ax2.set_ylim(-70, 70)
    ax2.set_yticks(np.arange(-60, 70, 20))
    ax2.set_ylabel("Latitude")
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_xticklabels(models, rotation=90, fontsize=8)
    ax2.grid(alpha=0.3, zorder=0)
    ax2.set_title(r"$\bf{b)}$ Zonal-mean $b_3$ bias by model", fontsize=14)
    fig.canvas.draw()
    p = ax2.get_position()
    ax2.set_position([p.x0, p.y0 + 0.1, p.width, p.height * 0.5])

    # (c-e) partial correlations with PC1
    bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=-0.05)
    polar = ccrs.SouthPolarStereo()
    axes = []
    for k, (key, title) in enumerate([("sst", "c) SST"), ("sss", "d) SSS"),
                                      ("tauu", r"e) $\tau_x$")]):
        ax = fig.add_subplot(bottom[k], projection=polar)
        r, pval = res[key]
        im = polar_map(ax, r, LEVELS, pvalue=pval, p_threshold=P_THRESHOLD)
        letter, name = title.split(" ", 1)
        ax.set_title(rf"$\bf{{{letter}}}$ {name}", pad=8, fontsize=14)
        axes.append(ax)
        if key != "tauu":
            vals, lon = add_cyclic_point(samw.values.astype(float), coord=samw.lon.values)
            ax.contour(lon, samw.lat.values, vals, levels=[0.5], colors="green", linewidths=1.8,
                       transform=PC, zorder=11)
        if k == 0:
            im_c = im

    for (lon, lat), color, ls in [(res["jet_obs"], "k", "-"), (res["jet_model"], "darkorange", "--")]:
        axes[2].plot(lon, lat, color=color, lw=1.8, ls=ls, transform=PC, zorder=10)
    legend = axes[2].legend(
        handles=[mlines.Line2D([], [], color="green", lw=1.8, label="SAMW"),
                 mlines.Line2D([], [], color="k", lw=2, label=r"Obs $\phi(\tau_x^{\max})$"),
                 mlines.Line2D([], [], color="darkorange", lw=2, ls="--",
                               label=r"CMIP6 $\phi(\tau_x^{\max})$")],
        loc="lower left", fontsize=12, frameon=True, framealpha=0.8,
        bbox_to_anchor=(0.98, 0.0), borderaxespad=0)

    fig.canvas.draw()
    geo_labels(ax1)

    pa = ax1.get_position()
    cax = fig.add_axes([pa.x0 + (pa.width - 0.15) / 2, pa.y0 - 0.04, 0.15, 0.010])
    cb = fig.colorbar(im_a, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(r"$b_3$ bias  [$\times10^{-3}$ kg m$^{-4}$]", fontsize=12)
    cb.set_ticks([-2, -1, 0, 1, 2])

    lb = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    w = lb.width * 0.85
    cax = fig.add_axes([lb.x0 + (lb.width - w) / 2 + 0.004, lb.y1 + 0.025, w, 0.012])
    cb = fig.colorbar(im_c, cax=cax, orientation="horizontal", extend="both", ticks=[-0.8, 0, 0.8])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.set_xticklabels(["-0.8", "0", "0.8"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Partial $r$", fontsize=12)
    return fig


if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(ds, compute(ds), samw_mask(ds))
    fig.savefig(FIG_DIR / "fig3_b3_hotspot.pdf", dpi=300, bbox_inches="tight")
