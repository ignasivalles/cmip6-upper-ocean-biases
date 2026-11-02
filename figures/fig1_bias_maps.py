"""Figure 1: observed, CMIP6 ensemble-mean and bias maps of the SHDR parameters."""

import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common.data import load_ensemble
from common.paths import FIG_DIR
from common.plotting import ROBINSON, geo_labels, global_map

# variable, label, scale, (field range), (bias range), unit
ROWS = [
    ("a1", "SSD", 1, (20, 30), (-1, 1), r"kg m$^{-3}$"),
    ("D1", "MLD", 1, (0, 200), (-100, 100), "m"),
    ("a2", r"$a_2$ (upp. pyc. $\Delta\rho$)", 1, (0, 10), (-1, 1), r"kg m$^{-3}$"),
    ("b3", r"$b_3$ (main pyc. $d\rho/dz$)", 1e3, (0, 4), (-2, 2), r"kg m$^{-4}$ $\times\,10^{-3}$"),
]


def _fmt(x, _pos):
    return f"{int(round(x))}" if abs(x - round(x)) < 1e-8 else f"{x:.1f}"


def make_figure(ds, obs_name):
    fig = plt.figure(figsize=(16, 10))
    # map | gap | map | gap | map
    gs = gridspec.GridSpec(len(ROWS), 5, figure=fig, width_ratios=[1, -0.1, 1, 0.08, 1],
                           wspace=0, hspace=0.28)
    letters = iter("abcdefghijkl")
    axes, images = [], []

    for r, (var, label, scale, rng, brng, _) in enumerate(ROWS):
        obs = ds[f"{var}_obs"].mean("month") * scale
        mod = ds[var].mean(("month", "model")) * scale
        bias = ds[f"bias_{var}"].mean(("month", "model")) * scale

        row = [fig.add_subplot(gs[r, c], projection=ROBINSON) for c in (0, 2, 4)]
        for ax, name in zip(row, [obs_name, "CMIP6", "Bias"]):
            ax.set_title(rf"$\bf{{{next(letters)})}}\,${name} {label}", loc="left", fontsize=14, pad=3)
        im_obs = global_map(row[0], obs, *rng, cmocean.cm.dense)
        global_map(row[1], mod, *rng, cmocean.cm.dense)
        im_bias = global_map(row[2], bias, *brng, cmocean.cm.balance)
        axes.append(row)
        images.append((im_obs, im_bias))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_size_inches()[0] * fig.dpi

    for r, row in enumerate(axes):
        bottom = r == len(ROWS) - 1
        geo_labels(row[0], lat=True, lon=bottom)
        geo_labels(row[1], lat=False, lon=bottom)
        geo_labels(row[2], lat=False, lon=bottom)

        unit = ROWS[r][5]
        for ax, im, symmetric in [(row[1], images[r][0], False), (row[2], images[r][1], True)]:
            x_right = ax.spines["geo"].get_window_extent(renderer).x1 / fig_w
            p = ax.get_position()
            h = p.height * 0.65
            cax = fig.add_axes([x_right + 0.008, p.y0 + (p.height - h) / 2, 0.010, h])
            cb = fig.colorbar(im, cax=cax, extend="both")
            cb.set_label(unit, fontsize=12, labelpad=5)
            cb.ax.tick_params(labelsize=10)
            cb.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
            cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=symmetric))
    return fig


if __name__ == "__main__":
    ds = load_ensemble()
    w = np.cos(np.deg2rad(ds.lat))
    for var, label, *_ in ROWS:
        m = ds[f"bias_{var}"].mean(("month", "model")).weighted(w).mean(("lat", "lon"))
        print(f"global-mean bias {var}: {float(m):.4g}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(ds, "EN4").savefig(FIG_DIR / "fig1_bias_maps.pdf", dpi=300, bbox_inches="tight")
