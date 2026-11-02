"""Figure S13: correlations of Southern Ocean surface biases with PC2 of the
subtropical b3 bias."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from common.data import SO_EXCLUDE, basin_masks, load_ensemble
from common.eof import corr_with_field, hotspot_masks, inter_model_eof
from common.paths import FIG_DIR
from common.plotting import polar_map

PANELS = [("bias_sst", "a) SST"), ("bias_sos", "b) SSS"), ("bias_tauu", r"c) $\tau_x$"),
          ("bias_hflux", r"d) $Q_{\mathrm{net}}$"), ("bias_ep", r"e) $E-P$")]

if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    pac, ind = hotspot_masks(ds, basin_masks(ds))
    pcs, _, var = inter_model_eof(ds["bias_b3"].mean("month"), pac | ind)
    print(f"EOF1 {var[0] * 100:.1f}%, EOF2 {var[1] * 100:.1f}%")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.08, top=0.93, bottom=0.10)
    axes = []
    for k, (var_name, title) in enumerate(PANELS):
        r, p = corr_with_field(ds[var_name].mean("month"), pcs[1])
        ax = fig.add_subplot(gs[divmod(k, 3)], projection=ccrs.SouthPolarStereo())
        im = polar_map(ax, r, np.linspace(-0.8, 0.8, 17), pvalue=p)
        letter, name = title.split(" ", 1)
        ax.set_title(rf"$\bf{{{letter}}}$ {name}", fontsize=10, loc="left")
        axes.append(ax)

    fig.canvas.draw()
    pos = axes[4].get_position()
    h = pos.height * 0.65
    cb = fig.colorbar(im, cax=fig.add_axes([pos.x1 + 0.01, pos.y0 + (pos.height - h) / 2, 0.012, h]),
                      extend="both")
    cb.set_label("Pearson $r$ with PC$_2$", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS13_pc2.pdf", dpi=300, bbox_inches="tight")
