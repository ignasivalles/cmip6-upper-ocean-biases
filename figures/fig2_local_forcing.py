"""Figure 2: dominance analysis of local surface-forcing biases (Ft, Fs, wind-stress
curl) for the SHDR parameter biases, by basin and latitude zone.

Also writes the table of weights and bootstrap CIs (fig2_dominance.csv), reused
by Figure S11.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_ensemble
from common.paths import FIG_DIR
from common.stats import bootstrap_dominance, critical_r2, dominance
from common.zones import BASINS, INDIAN_ZONES, PREDICTORS, TARGETS, ZONES, zone_tables

ROW_LABELS = ["SSD", "MLD", r"$a_2$", r"$b_3$"]
COLORS = {"bias_Ft": "#E67E22", "bias_Fs": "#3498DB", "bias_curl": "#2ECC71"}
LEGEND = {"bias_Ft": "Ft", "bias_Fs": "Fs", "bias_curl": "curl"}


def dominance_table(ds, n_boot=1000):
    rows = []
    for (basin, zone), df in zone_tables(ds).items():
        for target in TARGETS:
            sub = df[PREDICTORS + [target]].dropna()
            if len(sub) < 6:
                continue
            w, total = dominance(sub, target, PREDICTORS)
            ci = bootstrap_dominance(sub, target, PREDICTORS, n_boot=n_boot)
            rows.append(dict(basin=basin, zone=zone, bias_var=target, n_models=len(sub),
                             total_r2=total, total_r2_lo=ci["total_r2"][0],
                             total_r2_hi=ci["total_r2"][1], **w))
    return pd.DataFrame(rows)


def dominance_panel(ax, df, basin, show_ci=True):
    zones = INDIAN_ZONES if basin == "Indian" else list(ZONES)
    sub = df[df.basin == basin].set_index("zone").reindex(zones)
    x = np.array([list(ZONES).index(z) for z in zones])
    ax.set_xlim(-0.5, 4.5 if basin == "Indian" else 6.5)

    width = 0.5 if basin == "Indian" else 0.7
    bottom = np.zeros(len(zones))
    for p in PREDICTORS:
        h = sub[p].fillna(0).values * 100
        ax.bar(x, h, bottom=bottom, width=width, color=COLORS[p], alpha=0.85)
        bottom += h

    total = sub["total_r2"].fillna(0).values * 100
    if show_ci:
        err = np.clip([total - sub["total_r2_lo"].fillna(0) * 100,
                       sub["total_r2_hi"].fillna(0) * 100 - total], 0, None)
        ax.errorbar(x, total, yerr=err, fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=2.5, capthick=1.0, zorder=5, alpha=0.7)
    ax.plot(x, total, color="black", lw=1.3, ls="--", zorder=4)
    sig = total > np.array([critical_r2(n) * 100 for n in sub["n_models"].fillna(0)])
    ax.scatter(x[sig], total[sig], color="black", s=18, zorder=6)

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles(show_ci=True):
    h = [plt.Line2D([0], [0], color=COLORS[p], lw=6, label=LEGEND[p]) for p in PREDICTORS]
    h.append(plt.Line2D([0], [0], color="black", marker="o", lw=2.0, ls="--", label="Total %$R^2$"))
    if show_ci:
        h.append(plt.Line2D([0], [0], color="black", lw=1.0, label="95% bootstrap CI"))
    return h


def make_figure(df):
    fig = plt.figure(figsize=(12, 12.8))
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.28, left=0.09, right=0.97, top=0.95, bottom=0.06)
    letters = iter("abcdefghijkl")
    for i, (target, label) in enumerate(zip(TARGETS, ROW_LABELS)):
        for j, basin in enumerate(BASINS):
            ax = fig.add_subplot(gs[i, j])
            dominance_panel(ax, df[df.bias_var == target], basin)
            if i == 0:
                ax.set_title(basin, fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{label}\nPartial %$R^2$", fontsize=10)
            if i == 3 and j == 1:
                ax.set_xlabel("Region", fontsize=11)
            ax.text(-0.12, 1.08, rf"$\bf{{{next(letters)})}}$", transform=ax.transAxes,
                    fontsize=13, va="bottom")
    ax.legend(handles=legend_handles(), loc="upper right", fontsize=7.5, frameon=False, ncol=2,
              columnspacing=0.8, handlelength=1.5)
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = dominance_table(load_ensemble(), n_boot=args.n_boot)
    df.to_csv(FIG_DIR / "fig2_dominance.csv", index=False)
    make_figure(df).savefig(FIG_DIR / "fig2_local_forcing.pdf", dpi=300, bbox_inches="tight")
