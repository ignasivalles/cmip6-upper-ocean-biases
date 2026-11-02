"""Figure S8: b3 bias referenced to EN4 against IAPv4 (top) and CORA (bottom):
cell-by-cell in two open-ocean subtropical boxes, and PC1 of the hotspot."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from common.data import basin_masks, load_ensemble
from common.eof import hotspot_masks, inter_model_eof
from common.gradient import BOXES, box_mask
from common.paths import FIG_DIR

AXIS_LIMITS = {("IAP", "Indian"): (-2, 5), ("CORA", "Indian"): (-5, 4)}
RED = "#B33A3A"


def box_values(bias, box):
    return bias.mean("month").where(box_mask(bias, box)).values.ravel()


def pc1(ds):
    pac, ind = hotspot_masks(ds, basin_masks(ds))
    return inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)[0][0]


def box_panel(ax, fig, x, y, box, product, letter):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok] * 1e3, y[ok] * 1e3
    r = np.corrcoef(x, y)[0, 1]
    lims = AXIS_LIMITS.get((product, box["name"]))
    if lims is None:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        lims = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.hexbin(x, y, gridsize=40, extent=[*lims, *lims], cmap="Blues", mincnt=1, linewidths=0.2,
              rasterized=True, zorder=1)
    ax.plot(lims, lims, color="gray", ls="--", lw=1.2, zorder=5)
    ax.axhline(0, color="k", lw=0.6, zorder=0)
    ax.axvline(0, color="k", lw=0.6, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"EN4 bias $b_3$ [$\times10^{-3}$ kg m$^{-4}$]", fontsize=11)
    ax.set_ylabel(rf"{product} bias $b_3$ [$\times10^{{-3}}$ kg m$^{{-4}}$]", fontsize=11)
    ax.set_title(rf"$\bf{{{letter})}}$  {box['name']} (n={len(x)}, r={r:.2f})", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    # inset map of the box
    p = ax.get_position()
    inset = fig.add_axes([p.x0 + 0.02 * p.width, p.y1 - 0.32 * p.height, 0.34 * p.width, 0.30 * p.height],
                         projection=ccrs.PlateCarree(central_longitude=np.mean(box["lon"])))
    inset.set_extent(box["extent"], ccrs.PlateCarree())
    inset.add_feature(cfeature.LAND, facecolor="darkgray", zorder=9)
    inset.coastlines(linewidth=0.4, zorder=10)
    (la0, la1), (lo0, lo1) = box["lat"], box["lon"]
    inset.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], color=RED, lw=1.5,
               transform=ccrs.PlateCarree(), zorder=11)
    return r


def pc_panel(ax, a, b, product, letter):
    a, b = xr.align(a, b)
    r = np.corrcoef(a, b)[0, 1]
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    lims = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.plot(lims, lims, color="gray", ls="--", lw=1.2, zorder=1)
    ax.scatter(a, b, s=34, color=RED, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("PC1 (EN4)", fontsize=11)
    ax.set_ylabel(f"PC1 ({product})", fontsize=11)
    ax.set_title(rf"$\bf{{{letter})}}$  PC1 (n={a.size}, r={r:.2f})", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    return r


if __name__ == "__main__":
    en4 = load_ensemble()
    pc_en4 = pc1(en4)

    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.94, bottom=0.07, wspace=0.38, hspace=0.35)
    letters = iter("abcdef")
    for row, product in enumerate(["IAP", "CORA"]):
        other = load_ensemble(obs=product)
        for col, box in enumerate(BOXES):
            r = box_panel(fig.add_subplot(gs[row, col]), fig, box_values(en4["bias_b3"], box),
                          box_values(other["bias_b3"], box), box, product, next(letters))
            print(f"{product} {box['name']} box: r={r:.2f}")
        r = pc_panel(fig.add_subplot(gs[row, 2]), pc_en4, pc1(other), product, next(letters))
        print(f"{product} PC1: r={r:.2f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS08_obs_products.pdf", dpi=300, bbox_inches="tight")
