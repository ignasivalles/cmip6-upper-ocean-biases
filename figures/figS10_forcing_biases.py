"""Figure S10: ensemble-mean biases of the thermal and haline surface density fluxes
and of the wind-stress curl (relative to ERA5 and ORAS5)."""

import cmocean
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common.data import load_ensemble
from common.paths import FIG_DIR
from common.plotting import ROBINSON, geo_labels, global_map
from fig1_bias_maps import _fmt

PANELS = [
    ("bias_Ft", r"$\bf{a)}$ Thermal flux bias $F_t$", 1e6, 2, r"kg m$^{-2}$ s$^{-1}$ $\times\,10^{-6}$"),
    ("bias_Fs", r"$\bf{b)}$ Haline flux bias $F_s$", 1e7, 5, r"kg m$^{-2}$ s$^{-1}$ $\times\,10^{-7}$"),
    ("bias_curl", r"$\bf{c)}$ Wind stress curl bias", 1e8, 1, r"N m$^{-3}$ $\times\,10^{-8}$"),
]

if __name__ == "__main__":
    ds = load_ensemble()
    fig = plt.figure(figsize=(15, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    axes, images = [], []
    for k, (var, title, scale, vmax, _) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, k], projection=ROBINSON)
        ax.set_title(title, loc="left", fontsize=14, pad=3)
        images.append(global_map(ax, ds[var].mean(("month", "model")) * scale, -vmax, vmax,
                                 cmocean.cm.balance))
        axes.append(ax)

    fig.canvas.draw()
    for k, (ax, im, panel) in enumerate(zip(axes, images, PANELS)):
        geo_labels(ax, lat=k == 0)
        p = ax.get_position()
        w = p.width * 0.75
        cb = fig.colorbar(im, cax=fig.add_axes([p.x0 + (p.width - w) / 2, p.y0 - 0.12, w, 0.055]),
                          orientation="horizontal", extend="both")
        cb.set_label(panel[4], fontsize=12)
        cb.ax.tick_params(labelsize=10)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS10_forcing_biases.pdf", dpi=300, bbox_inches="tight")
