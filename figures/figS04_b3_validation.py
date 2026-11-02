"""Figure S4: b3 against the main-pycnocline density gradient computed directly
from the model temperature and salinity profiles (preprocessing step 09).

(a) example profile, (b) one point per model, (c, d) all cells of the two
subtropical boxes, pooled over models.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from common.data import load_ensemble
from common.gradient import BOXES, Z_BASE, background_gradient, box_mask
from common.paths import DATA_DIR, FIG_DIR

RED, BLUE = "#B33A3A", "#2E5C8A"
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def example_panel(ax, raw, b3):
    sig, z = raw["example_sigma0"].values, raw["depth"].values
    slope, icpt, z_peak, z_base = background_gradient(sig, z)
    ex = {k.replace("example_", ""): v for k, v in raw.attrs.items()}
    b3_val = float(b3.sel(model=ex["model"], month=int(ex["month"]))
                   .sel(lat=float(ex["lat"]), lon=float(ex["lon"]), method="nearest"))

    ax.plot(sig, -z, "o-", color=BLUE, ms=3, lw=1.3, zorder=6, label="raw profile")
    ax.plot(icpt + slope * np.array([z_base, Z_BASE]), [-z_base, -Z_BASE], color=RED, lw=2.4,
            zorder=5, label=r"$d\rho/dz$")
    ax.plot(icpt + slope * np.array([z[0], z_base]), [-z[0], -z_base], color=RED, lw=1.4, ls="--",
            zorder=4)
    s900 = icpt + slope * Z_BASE
    ax.plot(s900 + b3_val * (np.array([z[0], Z_BASE]) - Z_BASE), [-z[0], -Z_BASE], color="k",
            lw=1.8, ls="-.", zorder=7, label=r"$b_3$")
    ax.axhline(-z_peak, color="gray", ls=":", lw=1.2)
    ax.axhline(-z_base, color="k", ls="--", lw=1.2)
    x_text = ax.get_xlim()[1]
    ax.text(x_text, -z_peak, r"$d\rho/dz$ max", fontsize=8, va="bottom", ha="right", color="gray")
    ax.text(x_text, -z_base, r"$d\rho/dz$ < 5% max", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel(r"$\sigma_0$ [kg m$^{-3}$]", fontsize=11)
    ax.set_ylabel("Depth [m]", fontsize=11)
    ax.set_title(rf"$\bf{{a)}}$  {ex['model']}, {abs(float(ex['lat'])):.0f}°S "
                 rf"{float(ex['lon']):.0f}°E, {MONTHS[int(ex['month'])]}", loc="left", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)


def pairs(raw, b3_annual, mask):
    """Flattened (b3, raw gradient) over a mask, one array per model."""
    out = {}
    for model in raw.model.values:
        b = b3_annual.sel(model=model).where(mask).values.ravel()
        r = raw["raw_gradient"].sel(model=model).where(mask).values.ravel()
        ok = np.isfinite(b) & np.isfinite(r)
        if ok.sum() >= 10:
            out[model] = (b[ok], r[ok])
    return out


def one_to_one(ax, x, y, label_x, label_y, title):
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color="gray", ls="--", lw=1.2, zorder=5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_title(title, loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    return lim


def inset_map(fig, ax, box):
    p = ax.get_position()
    m = fig.add_axes([p.x0 + 0.02 * p.width, p.y1 - 0.32 * p.height, 0.34 * p.width, 0.30 * p.height],
                     projection=ccrs.PlateCarree(central_longitude=np.mean(box["lon"])))
    m.set_extent(box["extent"], ccrs.PlateCarree())
    m.add_feature(cfeature.LAND, facecolor="darkgray", zorder=9)
    m.coastlines(linewidth=0.4, zorder=10)
    (la0, la1), (lo0, lo1) = box["lat"], box["lon"]
    m.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], color=RED, lw=1.5,
           transform=ccrs.PlateCarree(), zorder=11)


if __name__ == "__main__":
    raw = xr.open_dataset(DATA_DIR / "b3_raw_gradient.nc")
    b3 = load_ensemble()["b3"].sel(lat=raw.lat, lon=raw.lon)
    b3_annual = b3.mean("month")
    b3_annual = b3_annual.where((b3_annual > -1e-3) & (b3_annual < 1e-2))

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    example_panel(fig.add_subplot(gs[0, 0]), raw, b3)

    # (b) model medians over both boxes
    both = box_mask(raw, BOXES[0]) | box_mask(raw, BOXES[1])
    med = {m: (np.median(b), np.median(r)) for m, (b, r) in pairs(raw, b3_annual, both).items()}
    x, y = (np.array([v[k] for v in med.values()]) * 1e3 for k in (0, 1))
    r = np.corrcoef(x, y)[0, 1]
    ax = fig.add_subplot(gs[0, 1])
    one_to_one(ax, x, y, r"model-median $b_3$ [$\times10^{-3}$ kg m$^{-4}$]",
               r"model-median raw $d\sigma_0/dz$ [$\times10^{-3}$ kg m$^{-4}$]",
               rf"$\bf{{b)}}$  Across-model (n={len(med)}, r={r:.2f})")
    ax.scatter(x, y, s=32, color=RED, edgecolor="white", linewidth=0.5, zorder=3)
    print(f"across models: n={len(med)} r={r:.2f}")

    # (c, d) all cells of each box
    for k, (box, letter) in enumerate(zip(BOXES, "cd")):
        cells = pairs(raw, b3_annual, box_mask(raw, box))
        x = np.concatenate([b for b, _ in cells.values()]) * 1e3
        y = np.concatenate([r for _, r in cells.values()]) * 1e3
        r = np.corrcoef(x, y)[0, 1]
        ax = fig.add_subplot(gs[1, k])
        lim = one_to_one(ax, x, y, r"$b_3$ [$\times10^{-3}$ kg m$^{-4}$]",
                         r"raw $d\sigma_0/dz$ [$\times10^{-3}$ kg m$^{-4}$]",
                         rf"$\bf{{{letter})}}$  {box['name']} (n={len(x)}, r={r:.2f})")
        ax.hexbin(x, y, gridsize=45, extent=[*lim, *lim], cmap="Blues", mincnt=1, linewidths=0.2,
                  rasterized=True, zorder=1)
        inset_map(fig, ax, box)
        print(f"{box['name']} box: n={len(x)} r={r:.2f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS04_b3_validation.pdf", dpi=300, bbox_inches="tight")
