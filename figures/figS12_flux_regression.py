"""Figure S12: regression of net heat flux and E-P biases onto PC1, and the
inter-model relation between SST and net heat flux biases in the subpolar band
and in the SAMW formation belt."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats

from common.data import SO_EXCLUDE, area_mean, basin_masks, load_ensemble, samw_mask
from common.eof import hotspot_masks, inter_model_eof, regress_on_index
from common.paths import FIG_DIR
from common.plotting import polar_map
from fig1_bias_maps import _fmt

PANELS = [("bias_hflux", 1, 20, r"$\bf{a)}$ $Q_{net}$ regression onto PC$_1$", r"W m$^{-2}$"),
          ("bias_ep", 1e6, 4, r"$\bf{b)}$ E$-$P regression onto PC$_1$",
           r"$\times10^{-6}$ kg m$^{-2}$ s$^{-1}$")]


def sst_qnet_scatter(ax, ds, mask, letter, title):
    """Area-mean SST bias against area-mean Qnet bias, one dot per model."""
    x = area_mean(ds["bias_sst"].mean("month"), mask).values
    y = area_mean(ds["bias_hflux"].mean("month"), mask).values
    r, p = stats.pearsonr(x, y)
    fit = stats.linregress(x, y)
    xl = np.array([x.min(), x.max()])
    ax.scatter(x, y, color="#2E5C8A", s=40)
    ax.plot(xl, fit.intercept + fit.slope * xl, "k--", lw=1.5, zorder=4)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("SST bias (°C)", fontsize=12)
    ax.set_ylabel(r"$Q_{net}$ bias (W m$^{-2}$)", fontsize=12)
    ax.set_title(rf"$\bf{{{letter})}}$ {title}", loc="left", fontsize=14)
    ax.text(0.04, 0.94, f"r = {r:.2f} (p = {p:.3f})", transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    print(f"{title}: n={len(x)} r={r:.2f} p={p:.3f}")

if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    pac, ind = hotspot_masks(ds, basin_masks(ds))
    pc1 = inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)[0][0]

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.42, height_ratios=[1.15, 0.85])
    for k, (var, scale, vmax, title, unit) in enumerate(PANELS):
        slope, p = regress_on_index(ds[var].mean("month") * scale, pc1)
        ax = fig.add_subplot(gs[0, k], projection=ccrs.SouthPolarStereo())
        im = polar_map(ax, slope, np.linspace(-vmax, vmax, 21), lat_max=-30, pvalue=p,
                       p_threshold=0.01)
        ax.set_title(title, pad=8, fontsize=14)
        pos = ax.get_position()
        w = pos.width * 0.75
        cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + (pos.width - w) / 2, pos.y0 - 0.07, w, 0.022]),
                          orientation="horizontal", extend="both")
        cb.set_label(unit, fontsize=12)
        cb.ax.tick_params(labelsize=10)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    subpolar = (ds.lat >= -70) & (ds.lat <= -50)
    sst_qnet_scatter(fig.add_subplot(gs[1, 0]), ds, subpolar, "c", "Subpolar band (50–70°S)")
    sst_qnet_scatter(fig.add_subplot(gs[1, 1]), ds, samw_mask(ds), "d", "SAMW formation belt")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS12_flux_regression.pdf", dpi=300, bbox_inches="tight")
