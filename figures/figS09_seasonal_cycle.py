"""Figure S9: multi-model median bias of the SHDR parameters by month and latitude
in each basin, with the annual-mean median and interquartile range."""

import matplotlib.pyplot as plt

from common.data import load_ensemble
from common.paths import FIG_DIR
from common.zones import BASINS, extended_basins, quality_mask

# variable, title, unit, colour limit, scale
VARIABLES = [
    ("bias_a1", r"$a_1$ bias", r"kg m$^{-3}$", 1.0, 1),
    ("bias_D1", r"$D_1$ bias", "m", 40.0, 1),
    ("bias_a2", r"$a_2$ bias", r"kg m$^{-3}$", 0.8, 1),
    ("bias_b3", r"$b_3$ bias", r"$\times10^{-4}$ kg m$^{-4}$", 6e-4, 1e4),
]
LAT_RANGE = {"Atlantic": (-65, 65), "Pacific": (-65, 65), "Indian": (-65, 20)}

if __name__ == "__main__":
    ds = quality_mask(load_ensemble())
    masks = extended_basins(ds)

    ratios = []
    for k in range(len(VARIABLES)):
        ratios += [4, 1] + ([0.4] if k < len(VARIABLES) - 1 else [])
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(len(BASINS), len(ratios), width_ratios=ratios, hspace=0.30, wspace=0.05)

    for r, basin in enumerate(BASINS):
        sub = ds.where(masks[basin]).sel(lat=slice(*LAT_RANGE[basin]))
        for c, (var, title, unit, vmax, scale) in enumerate(VARIABLES):
            zonal = sub[var].mean("lon")
            med = zonal.median("model")
            q25, q75 = (zonal.quantile(q, "model").mean("month") for q in (0.25, 0.75))
            lat = med.lat.values

            ax = fig.add_subplot(gs[r, 3 * c])
            im = ax.pcolormesh(med.month, lat, med.T * scale, cmap="RdBu_r", vmin=-vmax * scale,
                               vmax=vmax * scale, shading="auto")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([1, 3, 6, 9, 12])
            ax.set_xticklabels(["J", "M", "J", "S", "D"], fontsize=11)
            if c == 0:
                ax.set_ylabel("Latitude (°N)", fontsize=12)
                ax.text(-0.28, 0.5, basin, transform=ax.transAxes, fontsize=13, fontweight="bold",
                        rotation=90, va="center", ha="center")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
            if r == len(BASINS) - 1:
                p = ax.get_position()
                cb = fig.colorbar(im, cax=fig.add_axes([p.x0, p.y0 - 0.045, p.width, 0.012]),
                                  orientation="horizontal")
                cb.ax.tick_params(labelsize=10)
                cb.set_label(unit, fontsize=11)

            axp = fig.add_subplot(gs[r, 3 * c + 1], sharey=ax)
            axp.fill_betweenx(lat, q25 * scale, q75 * scale, alpha=0.3, color="steelblue")
            axp.plot(med.mean("month") * scale, lat, color="darkblue", lw=1.8)
            axp.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            axp.set_xlim(-vmax * scale, vmax * scale)
            axp.tick_params(axis="y", labelleft=False)
            axp.spines[["top", "right"]].set_visible(False)
            axp.grid(True, axis="x", alpha=0.3)
            if r < len(BASINS) - 1:
                axp.tick_params(axis="x", labelbottom=False)
            else:
                axp.set_xlabel("Annual\nmean", fontsize=10)
                axp.tick_params(axis="x", labelsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS09_seasonal_cycle.pdf", dpi=300, bbox_inches="tight")
