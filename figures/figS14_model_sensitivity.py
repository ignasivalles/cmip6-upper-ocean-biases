"""Figure S14: Figure 4 dominance analysis (Pacific and Indian hotspots combined)
after leaving out (a) one model at a time and (b) one model family at a time."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import SO_EXCLUDE, load_ensemble, samw_mask
from common.paths import FIG_DIR
from common.stats import dominance
from fig4_southern_ocean import COLORS, PREDICTORS, predictor_table

# models sharing ocean and/or atmosphere code
FAMILIES = {
    "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
    "BCC": ["BCC-CSM2-MR", "BCC-ESM1"],
    "NCAR/POP2": ["CESM2", "CESM2-WACCM", "CESM2-FV2", "SAM0-UNICON", "FIO-ESM-2-0"],
    "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
    "E3SM": ["E3SM-1-0", "E3SM-1-1-ECA"],
    "EC-Earth": ["EC-Earth3", "EC-Earth3-Veg-LR"],
    "GFDL": ["GFDL-CM4", "GFDL-ESM4"],
    "GISS": ["GISS-E2-1-G", "GISS-E2-2-G"],
    "MPI/ECHAM": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "NESM3"],
    "NorESM": ["NorESM2-LM", "NorESM2-MM"],
    "UK Met Office": ["UKESM1-1-LL", "HadGEM3-GC31-LL", "ACCESS-CM2"],
    "IPSL/CNRM": ["IPSL-CM6A-LR", "CNRM-ESM2-1"],
}
SERIES = [("SST", COLORS["SST"], "o", "SST"), ("SSS", COLORS["SSS"], "o", "SSS"),
          ("JetLat", COLORS["JetLat"], "o", r"$\phi(\tau_x^{\max})$"),
          ("total_r2", "black", "D", "Total $R^2$")]


def run(df, label):
    w, total = dominance(df, "b3_both", PREDICTORS)
    return dict(label=label, n_models=len(df), total_r2=total * 100,
                **{p: w[p] / total * 100 for p in PREDICTORS})


def panel(ax, res, full, title):
    y = np.arange(len(res))[::-1]
    for col, color, marker, _ in SERIES:
        ax.axvline(full[col], color=color, lw=1.0, ls="--", alpha=0.6, zorder=1)
        ax.scatter(res[col], y, s=22, color=color, marker=marker, edgecolor="white",
                   linewidth=0.4, zorder=3)
    for yy in y[::2]:
        ax.axhspan(yy - 0.5, yy + 0.5, color="0.95", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(res["label"], fontsize=8.5)
    ax.set_ylim(-0.7, len(res) - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%", fontsize=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)


if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    df = predictor_table(ds, samw_mask(ds))
    full = run(df, "all")
    loo = pd.DataFrame([run(df.drop(index=m), m) for m in sorted(df.index, key=str.lower)])
    lofo = pd.DataFrame([run(df.drop(index=[m for m in ms if m in df.index]),
                             f"{fam} ({sum(m in df.index for m in ms)})")
                         for fam, ms in FAMILIES.items()])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame([full]), loo, lofo]).to_csv(FIG_DIR / "figS14_model_sensitivity.csv",
                                                        index=False)
    for name, res in [("one model", loo), ("one family", lofo)]:
        shift = (res[["total_r2"] + PREDICTORS] - pd.Series(full)[["total_r2"] + PREDICTORS]).abs()
        print(f"leave {name} out, max shift:", shift.max().round(1).to_dict())

    fig = plt.figure(figsize=(9.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[len(lofo), len(loo) - len(lofo)], wspace=0.55,
                          hspace=0.35, left=0.17, right=0.98, top=0.95, bottom=0.07)
    panel(fig.add_subplot(gs[:, 0]), loo, full, rf"$\bf{{a)}}$  Leave one model out (n={len(df) - 1})")
    panel(fig.add_subplot(gs[0, 1]), lofo, full, r"$\bf{b)}$  Leave one family out")
    ax_leg = fig.add_subplot(gs[1, 1])
    ax_leg.axis("off")
    handles = [plt.Line2D([0], [0], marker=m, color="none", markerfacecolor=c, markeredgecolor="white",
                          markersize=7, label=lab) for _, c, m, lab in SERIES]
    handles.append(plt.Line2D([0], [0], color="0.4", ls="--", lw=1.0,
                              label=f"All models (n={len(df)})"))
    ax_leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5)
    fig.savefig(FIG_DIR / "figS14_model_sensitivity.pdf", dpi=300, bbox_inches="tight")
