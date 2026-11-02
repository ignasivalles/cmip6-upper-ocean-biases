"""Figure S15: subpolar North Atlantic SST bias as a proxy for AMOC-related errors.
(a) ensemble-mean SST bias with the subpolar gyre (SPG) box, (b) PC1 against the
SPG SST bias relative to each model's global-mean SST bias, (c-e) dominance
analysis with the SPG SST bias as a fourth predictor, for three choices of the
Southern Ocean SST/SSS averaging region."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common.data import SO_EXCLUDE, area_mean, basin_masks, load_ensemble, samw_mask
from common.eof import hotspot_masks, inter_model_eof
from common.paths import FIG_DIR
from common.stats import dominance
from fig4_southern_ocean import COLORS, LABELS, TARGETS, predictor_table, south_of_samw

SPG_LAT, SPG_LON = (45, 60), (310, 340)  # 45-60N, 20-50W
PREDICTORS = ["SST", "SSS", "JetLat", "SPG"]
COLORS = {**COLORS, "SPG": "#1ABC9C"}
LABELS = {**LABELS, "SPG": "SPG SST"}


def spg_index(ds):
    sst = ds["bias_sst"].mean("month")
    box = (sst.lat >= SPG_LAT[0]) & (sst.lat <= SPG_LAT[1]) & (sst.lon >= SPG_LON[0]) & (sst.lon <= SPG_LON[1])
    return (area_mean(sst, box) - area_mean(sst, np.isfinite(sst.lat))).to_pandas()


if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    masks = basin_masks(ds)
    pac, ind = hotspot_masks(ds, masks)
    pc1 = inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)[0][0].to_pandas()
    spg = spg_index(ds).dropna()

    samw = samw_mask(ds)
    regions = {"45-70S band": (ds.lat >= -70) & (ds.lat <= -45), "SAMW zone": samw,
               "South of SAMW": south_of_samw(ds, samw)}
    tables = {k: predictor_table(ds, m, masks).join(spg.rename("SPG"), how="inner")
              for k, m in regions.items()}

    fig = plt.figure(figsize=(13, 10))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.35)
    top = outer[0].subgridspec(1, 2, wspace=0.3)
    bottom = outer[1].subgridspec(1, 3, wspace=0.15)

    ax = fig.add_subplot(top[0], projection=ccrs.PlateCarree(central_longitude=-30))
    ax.set_extent([-80, 10, 20, 70], ccrs.PlateCarree())
    sst = ds["bias_sst"].mean(("month", "model"))
    im = ax.pcolormesh(sst.lon, sst.lat, sst, transform=ccrs.PlateCarree(), cmap="RdBu_r",
                       vmin=-3, vmax=3, rasterized=True)
    ax.add_feature(cfeature.LAND, facecolor="darkgray", zorder=9)
    ax.coastlines(linewidth=0.6, zorder=10)
    lo0, lo1 = SPG_LON[0] - 360, SPG_LON[1] - 360
    ax.plot([lo0, lo1, lo1, lo0, lo0], [SPG_LAT[0]] * 2 + [SPG_LAT[1]] * 2 + [SPG_LAT[0]],
            color="black", lw=2, transform=ccrs.PlateCarree(), zorder=11)
    ax.set_title(r"$\bf{a)}$  SPG SST bias", loc="left", fontsize=12)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.8, extend="both")
    cb.set_label("SST bias (°C)", fontsize=9)

    ax = fig.add_subplot(top[1])
    x, y = spg.align(pc1, join="inner")
    r, p = stats.pearsonr(x, y)
    fit = stats.linregress(x, y)
    xl = np.array([x.min(), x.max()])
    ax.scatter(x, y, color="#2E5C8A", s=45)
    ax.plot(xl, fit.intercept + fit.slope * xl, "k--", lw=1.8, zorder=4)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("SPG SST bias, relative to global mean (°C)")
    ax.set_ylabel("PC1 ($b_3$-bias hotspot)")
    ax.set_title(rf"$\bf{{b)}}$  PC1 vs SPG SST bias (n={len(x)}, r={r:.2f}, p={p:.2f})",
                 loc="left", fontsize=12)
    print(f"PC1 vs SPG: n={len(x)} r={r:.2f} p={p:.2f}")

    for k, ((target, title), letter) in enumerate(zip(TARGETS, "cde")):
        ax = fig.add_subplot(bottom[k])
        base = np.zeros(len(tables))
        weights = [dominance(df, target, PREDICTORS) for df in tables.values()]
        for pred in PREDICTORS:
            vals = np.array([w[pred] * 100 for w, _ in weights])
            ax.bar(range(len(tables)), vals, 0.55, bottom=base, color=COLORS[pred],
                   label=LABELS[pred], edgecolor="white", linewidth=0.8)
            for xi, (b, v) in enumerate(zip(base, vals)):
                if v >= 4:
                    ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")
            base += vals
        for xi, (region, (w, total)) in enumerate(zip(tables, weights)):
            ax.text(xi, total * 100 + 1.5, rf"$R^2$={total * 100:.0f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#2C3E50")
            print(f"{title}, {region}: R2={total * 100:.1f}%, SPG={w['SPG'] * 100:.1f} points")
        ax.set_xticks(range(len(tables)))
        ax.set_xticklabels(list(tables), fontsize=9)
        ax.set_ylim(0, 95)
        if k == 0:
            ax.set_ylabel("Absolute contribution (%)", fontsize=11)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(rf"$\bf{{{letter})}}$  {title}", loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.03))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS15_amoc_proxy.pdf", dpi=300, bbox_inches="tight")
