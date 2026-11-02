"""Figure S11: Figure 2 recomputed with NCEP/NCAR Reanalysis-1 as the reference for
surface fluxes and wind stress, and comparison of total R^2 with the ERA5/ORAS5
reference."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from common.data import load_ensemble
from common.paths import FIG_DIR, NCEP_FILE
from common.stats import dominance
from common.zones import BASINS, PREDICTORS, TARGETS, zone_tables
from fig2_local_forcing import ROW_LABELS, dominance_panel, legend_handles

BASIN_COLORS = {"Atlantic": "#4C72B0", "Pacific": "#55A868", "Indian": "#C44E52"}


def ncep_referenced(ds):
    """Replace the ERA5/ORAS5-referenced Ft, Fs and curl biases with NCEP-referenced
    ones. The observed Ft and Fs are rescaled by the NCEP/ERA5 ratio of net heat flux
    and E-P (cells with |Qnet| <= 5 W m-2 or |E-P| <= 1e-6 kg m-2 s-1 are masked)."""
    ncep = xr.open_dataset(NCEP_FILE).reindex(lat=ds.lat, lon=ds.lon, method="nearest",
                                              tolerance=0.01)
    ratio_q = (ncep["hflux"] / ds["hflux_obs"]).where(np.abs(ds["hflux_obs"]) > 5.0)
    ratio_e = (ncep["ep"] / ds["ep_obs"]).where(np.abs(ds["ep_obs"]) > 1e-6)
    out = ds.copy()
    out["bias_Ft"] = ds["Ft"] - ds["Ft_obs"] * ratio_q
    out["bias_Fs"] = ds["Fs"] - ds["Fs_obs"] * ratio_e
    out["bias_curl"] = ds["curl"] - ncep["curl"]
    return out


def point_estimates(ds):
    rows = []
    for (basin, zone), df in zone_tables(ds).items():
        for target in TARGETS:
            sub = df[PREDICTORS + [target]].dropna()
            if len(sub) < 6:
                continue
            w, total = dominance(sub, target, PREDICTORS)
            rows.append(dict(basin=basin, zone=zone, bias_var=target, n_models=len(sub),
                             total_r2=total, **w))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ds = load_ensemble()
    era5 = point_estimates(ds)
    ncep = point_estimates(ncep_referenced(ds))
    comp = era5.merge(ncep, on=["basin", "zone", "bias_var"], suffixes=("_era5", "_ncep"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(FIG_DIR / "figS11_ncep_vs_era5.csv", index=False)

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.3], hspace=0.65, wspace=0.28,
                          left=0.09, right=0.97, top=0.97, bottom=0.04)
    letters = iter("abcdefghijklm")
    for i, (target, label) in enumerate(zip(TARGETS, ROW_LABELS)):
        for j, basin in enumerate(BASINS):
            ax = fig.add_subplot(gs[i, j])
            dominance_panel(ax, ncep[ncep.bias_var == target], basin, show_ci=False)
            if i == 0:
                ax.set_title(basin, fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{label}\nPartial %$R^2$", fontsize=10)
            if i == 3 and j == 1:
                ax.set_xlabel("Region", fontsize=11)
            ax.text(-0.12, 1.08, rf"$\bf{{{next(letters)})}}$", transform=ax.transAxes,
                    fontsize=13, va="bottom")
    ax.legend(handles=legend_handles(show_ci=False), fontsize=8, frameon=False, loc="upper right")

    # (m) total R^2 under both references
    ax = fig.add_subplot(gs[4, :])
    x, y = comp["total_r2_era5"] * 100, comp["total_r2_ncep"] * 100
    r, mad = np.corrcoef(x, y)[0, 1], (y - x).abs().mean()
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color="gray", ls="--", lw=1.2, zorder=1)
    so = comp.zone == "SO"
    for basin, color in BASIN_COLORS.items():
        sel = (comp.basin == basin) & ~so
        ax.scatter(x[sel], y[sel], s=42, color=color, alpha=0.8, edgecolor="white", linewidth=0.5,
                   zorder=3, label=basin)
    ax.scatter(x[so], y[so], s=110, facecolor=[BASIN_COLORS[b] for b in comp.basin[so]],
               edgecolor="black", linewidth=1.4, zorder=5, marker="D",
               label="Southern Ocean (SO)\nzone cells")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Total $R^2$, ERA5+ORAS5 [%]", fontsize=11)
    ax.set_ylabel("Total $R^2$, NCEP/NCAR R1 [%]", fontsize=11)
    ax.set_title(f"All {len(comp)} cells (4 bias variables x zone x basin): r={r:.2f}, "
                 f"mean |diff|={mad:.1f} pts", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.05, 1.05, rf"$\bf{{{next(letters)})}}$", transform=ax.transAxes, fontsize=13,
            va="bottom")
    print(f"total R2, ERA5 vs NCEP: r={r:.3f}, mean |diff|={mad:.2f} points")

    fig.savefig(FIG_DIR / "figS11_ncep.pdf", dpi=300, bbox_inches="tight")
