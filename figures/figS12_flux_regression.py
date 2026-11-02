"""Figure S12: regression of net heat flux and E-P biases onto PC1."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from common.data import SO_EXCLUDE, basin_masks, load_ensemble
from common.eof import hotspot_masks, inter_model_eof, regress_on_index
from common.paths import FIG_DIR
from common.plotting import polar_map
from fig1_bias_maps import _fmt

PANELS = [("bias_hflux", 1, 20, r"$\bf{a)}$ $Q_{net}$ regression onto PC$_1$", r"W m$^{-2}$"),
          ("bias_ep", 1e6, 4, r"$\bf{b)}$ E$-$P regression onto PC$_1$",
           r"$\times10^{-6}$ kg m$^{-2}$ s$^{-1}$")]

if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    pac, ind = hotspot_masks(ds, basin_masks(ds))
    pc1 = inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)[0][0]

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    for k, (var, scale, vmax, title, unit) in enumerate(PANELS):
        slope, p = regress_on_index(ds[var].mean("month") * scale, pc1)
        ax = fig.add_subplot(gs[0, k], projection=ccrs.SouthPolarStereo())
        im = polar_map(ax, slope, np.linspace(-vmax, vmax, 21), lat_max=-30, pvalue=p,
                       p_threshold=0.01)
        ax.set_title(title, pad=8, fontsize=14)
        pos = ax.get_position()
        w = pos.width * 0.75
        cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + (pos.width - w) / 2, pos.y0 - 0.07, w, 0.022]),
                          orientation="horizontal", extend="both")
        cb.set_label(unit, fontsize=12)
        cb.ax.tick_params(labelsize=10)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS12_flux_regression.pdf", dpi=300, bbox_inches="tight")
