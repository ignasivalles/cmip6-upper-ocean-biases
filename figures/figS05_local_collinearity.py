"""Figure S5: correlations across models between the local forcing predictors
(Ft, Fs, wind-stress curl) in each basin and zone, as used in Figure 2."""

import matplotlib.pyplot as plt

from common.data import load_ensemble
from common.paths import FIG_DIR
from common.plotting import corr_triangle
from common.zones import BASINS, INDIAN_ZONES, PREDICTORS, ZONES, zone_tables

TITLES = {"SO": "SO", "S_ST": "S_SubT", "S_T": "S_Trop", "Eq": "Eq", "N_T": "N_Trop",
          "N_ST": "N_SubT", "N_HL": "N_HiLat"}

if __name__ == "__main__":
    tables = zone_tables(load_ensemble(), PREDICTORS)
    fig, axes = plt.subplots(3, len(ZONES), figsize=(16, 7.2))
    for i, basin in enumerate(BASINS):
        for j, zone in enumerate(ZONES):
            ax = axes[i, j]
            if (basin, zone) not in tables:
                ax.axis("off")
                continue
            im = corr_triangle(ax, tables[basin, zone].dropna(), ["Ft", "Fs", "curl"])
            if not (i == 2 or (basin == "Pacific" and zone not in INDIAN_ZONES)):
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(TITLES[zone], fontsize=12)
            if j == 0:
                ax.set_ylabel(basin, fontsize=12)
    cb = fig.colorbar(im, cax=fig.add_axes([0.76, 0.17, 0.17, 0.025]), orientation="horizontal")
    cb.set_label("Pearson r", fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS05_local_collinearity.pdf", dpi=300, bbox_inches="tight")
