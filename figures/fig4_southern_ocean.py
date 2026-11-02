"""Figure 4: Southern Ocean SST, SSS and jet-position biases against PC1 of the
subtropical b3 bias, and their dominance-analysis contributions.

Also writes the dominance inputs (fig4_predictors.csv) used by Figures S6,
S7 and S14.
"""

import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from common.data import (SO_EXCLUDE, basin_masks, jet_latitude, load_ensemble, region_mean,
                         samw_mask)
from common.eof import hotspot_masks, inter_model_eof
from common.paths import FIG_DIR
from common.stats import bootstrap_dominance, dominance

PREDICTORS = ["SST", "SSS", "JetLat"]
LABELS = {"SST": "SST", "SSS": "SSS", "JetLat": r"$\phi(\tau_x^{\max})$"}
COLORS = {"SST": "#E8503A", "SSS": "#3498DB", "JetLat": "#A855D8"}
TARGETS = [("b3_pac", "Pacific"), ("b3_ind", "Indian"), ("b3_both", "Pacific & Indian")]
MARKERS = ["o", "s", "D", "d", "^", "v", "<", ">", "p", "P", "*", "X", "h", "H", "8"]
FS = 12


def south_of_samw(ds, samw, south_limit=-75):
    """Cells between the southern edge of the SAMW belt and south_limit."""
    edge = samw.lat.where(samw).min("lat")
    return (ds.lat >= south_limit) & (ds.lat < edge)


def predictor_table(ds, region, masks=None, threshold=None):
    """Per-model SO predictors (SST, SSS averaged over region; jet latitude) and
    hotspot-mean b3 bias (Pacific, Indian and their mean)."""
    masks = masks or basin_masks(ds)
    kw = {} if threshold is None else {"threshold_low": threshold}
    pac, ind = hotspot_masks(ds, masks, **kw)
    b3 = ds["bias_b3"].mean("month")
    bp = b3.where(pac).mean(("lat", "lon"))
    bi = b3.where(ind).mean(("lat", "lon"))
    cols = {
        "SST": region_mean(ds["bias_sst"].mean("month"), region),
        "SSS": region_mean(ds["bias_sos"].mean("month"), region),
        "JetLat": jet_latitude(ds["tauu"].mean("month")),
        "b3_pac": bp, "b3_ind": bi, "b3_both": (bp + bi) / 2,
        "b3_hotspot": b3.where(pac | ind).mean(("lat", "lon")),
    }
    return pd.DataFrame({k: v.to_pandas() for k, v in cols.items()}).dropna()


def scatter_panel(ax, x, y, color, ylabel, title, stat_pos):
    for i, model in enumerate(x.index):
        sc = ax.scatter(x[model], y[model], c=color[model] * 1e4, cmap=cmocean.cm.amp,
                        marker=MARKERS[i % len(MARKERS)], s=100, edgecolor="k", linewidths=0.6,
                        vmin=4, vmax=16, label=model, zorder=5)
    slope, icpt, r, p, _ = stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, icpt + slope * xl, "k--", lw=1.8, zorder=4)
    tx, ty, ha, va = stat_pos
    ax.text(tx, ty, rf"$r={r:.2f},\ p={p:.2f}$", transform=ax.transAxes, fontsize=FS - 1,
            ha=ha, va=va)
    ax.set_xlabel("PC1", fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_title(title, loc="left", fontsize=FS + 2)
    ax.set_xlim(-3, 3)
    ax.set_xticks(range(-3, 4))
    ax.tick_params(labelsize=FS - 2)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"PC1 vs {y.name}: r={r:.3f} p={p:.4f}")
    return sc


def bar_panel(ax, df, title, name):
    x = np.arange(len(TARGETS))
    width = 0.25
    for k, pred in enumerate(PREDICTORS):
        vals, lo, hi = [], [], []
        for target, _ in TARGETS:
            w, total = dominance(df, target, PREDICTORS)
            ci = bootstrap_dominance(df, target, PREDICTORS)
            vals.append(w[pred] * 100)
            lo.append(ci[pred][0] * 100)
            hi.append(ci[pred][1] * 100)
            if k == 0:
                ax.text(x[len(vals) - 1], 83.5, rf"$R^2$={total * 100:.0f}%", ha="center",
                        va="top", fontsize=9.5, fontweight="bold", color="#2C3E50")
                print(f"{name} {target}: R2={total * 100:.1f}% " +
                      " ".join(f"{p}={v / total * 100:.1f}%" for p, v in w.items()))
        vals, lo, hi = map(np.array, (vals, lo, hi))
        xs = x + (k - 1) * width
        ax.bar(xs, vals, width * 0.9, color=COLORS[pred], label=LABELS[pred], edgecolor="white",
               linewidth=0.6, zorder=5)
        ax.errorbar(xs, vals, yerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)],
                    fmt="none", ecolor="#2C3E50", elinewidth=1.1, capsize=2.5, zorder=10)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in TARGETS], fontsize=10)
    ax.set_title(title, loc="left", fontsize=FS)
    ax.set_ylim(0, 85)
    ax.set_yticks(range(0, 81, 20))
    ax.set_xlim(-0.5, len(TARGETS) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)


def make_figure(pc1, df, df_south):
    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(2, 3, left=0.07, right=0.97, top=0.93, bottom=0.10,
                           height_ratios=[0.65, 0.8], hspace=0.35, wspace=0.30)
    pc = pc1.sel(model=df.index).to_pandas()

    ax_a = fig.add_subplot(gs[0, 0])
    scatter_panel(ax_a, pc, df["SST"], df["b3_hotspot"], "SST bias  [°C]",
                  r"$\mathbf{a)}$ SST", (0.95, 0.05, "right", "bottom"))
    ax_b = fig.add_subplot(gs[0, 1])
    sc = scatter_panel(ax_b, pc, df["SSS"], df["b3_hotspot"], "SSS bias  [psu]",
                       r"$\mathbf{b)}$ SSS", (0.05, 0.05, "left", "bottom"))
    ax_c = fig.add_subplot(gs[0, 2])
    scatter_panel(ax_c, pc, df["JetLat"], df["b3_hotspot"], r"$\phi(\tau_x^{\max})$ [$^{\circ}$ Lat]",
                  r"$\mathbf{c)}$ $\phi(\tau_x^{\max})$", (0.05, 0.95, "left", "top"))
    ax_c.set_ylim(-61, -40)
    ax_c.set_yticks([-60, -55, -50, -45, -40])
    ax_c.set_yticklabels(["60°S", "55°S", "50°S", "45°S", "40°S"])

    ax_d = fig.add_subplot(gs[1, 0])
    bar_panel(ax_d, df, r"$\bf{d)}$  SAMW zone", "SAMW")
    ax_d.set_ylabel("Absolute contribution (%)", fontsize=11)
    ax_e = fig.add_subplot(gs[1, 1])
    bar_panel(ax_e, df_south, r"$\bf{e)}$  South of SAMW", "south of SAMW")
    ax_e.tick_params(labelleft=False)

    # legend cell: model markers, colorbar, predictor colours
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    p = ax_leg.get_position()
    ax_leg.set_position([p.x0 - 0.05, p.y0 - 0.03, p.width, p.height])
    leg = ax_leg.legend(*ax_b.get_legend_handles_labels(), fontsize=8.5, ncol=3, loc="upper left",
                        bbox_to_anchor=(0, 1), frameon=False, handletextpad=0.3,
                        columnspacing=0.8, labelspacing=0.45)
    ax_leg.add_artist(leg)
    fig.canvas.draw()
    y0 = leg.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax_leg.transAxes.inverted()).y0 - 0.06
    cax = ax_leg.inset_axes([0.05, y0 - 0.05, 0.55, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[4, 8, 12, 16], extend="both")
    cb.ax.tick_params(labelsize=FS - 3)
    cb.set_label(r"$b_3$ [$\times10^{-4}$ kg m$^{-4}$]", fontsize=FS - 3, labelpad=3)
    ax_leg.legend(*ax_d.get_legend_handles_labels(), loc="upper left", fontsize=FS - 1,
                  bbox_to_anchor=(0.72, y0 + 0.09), frameon=False)
    return fig


if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    masks = basin_masks(ds)
    pac, ind = hotspot_masks(ds, masks)
    pcs, _, _ = inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)

    samw = samw_mask(ds)
    df = predictor_table(ds, samw, masks)
    df_south = predictor_table(ds, south_of_samw(ds, samw), masks)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.assign(PC1=pcs[0].sel(model=df.index).values).to_csv(FIG_DIR / "fig4_predictors.csv",
                                                           index_label="model")
    fig = make_figure(pcs[0], df, df_south)
    fig.savefig(FIG_DIR / "fig4_southern_ocean.pdf", dpi=300, bbox_inches="tight", pad_inches=0.25)
