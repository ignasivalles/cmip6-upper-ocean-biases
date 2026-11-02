"""Figure S6: sensitivity of the Figure 4 dominance analysis (Pacific and Indian
hotspots combined) to the percentile threshold that defines the b3 hotspot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import SO_EXCLUDE, basin_masks, load_ensemble, samw_mask
from common.eof import ensemble_mean_b3, hotspot_masks
from common.paths import FIG_DIR
from common.stats import dominance
from fig4_southern_ocean import COLORS, PREDICTORS, predictor_table

PERCENTILES = [50, 60, 70, 75, 80, 85, 90]


def sweep(ds):
    masks = basin_masks(ds)
    samw = samw_mask(ds)
    domain = ensemble_mean_b3(ds).where(masks["pacific"] | masks["indian"]).sel(lat=slice(-40, -10))
    values = domain.values[np.isfinite(domain.values)]
    rows = []
    for pct in PERCENTILES:
        thr = np.percentile(values, pct)
        pac, ind = hotspot_masks(ds, masks, threshold_low=thr)
        df = predictor_table(ds, samw, masks, threshold=thr)
        w, total = dominance(df, "b3_both", PREDICTORS)
        rows.append(dict(percentile=pct, threshold=thr, n_cells=int(pac.sum() + ind.sum()),
                         total_r2=total * 100, **{p: w[p] / total * 100 for p in PREDICTORS}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = sweep(load_ensemble(exclude=SO_EXCLUDE))
    print(df.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(df.percentile, df.n_cells, "o-", color="black", lw=1.5, ms=5)
    axes[0].set_ylabel("N hotspot cells", fontsize=11)
    axes[0].set_title(r"$\bf{a)}$  Domain size", loc="left", fontsize=12)
    axes[1].plot(df.percentile, df.total_r2, "o-", color="black", lw=1.5, ms=5)
    axes[1].set_ylabel("Total $R^2$ [%]", fontsize=11)
    axes[1].set_ylim(0, 100)
    axes[1].set_title(r"$\bf{b)}$  Total explained variance", loc="left", fontsize=12)
    for p in PREDICTORS:
        axes[2].plot(df.percentile, df[p], "o-", color=COLORS[p], lw=1.5, ms=5,
                     label=p if p != "JetLat" else r"$\phi(\tau_x^{\max})$")
    axes[2].set_ylabel("Partial %$R^2$", fontsize=11)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=9, frameon=False, loc="upper left")
    axes[2].set_title(r"$\bf{c)}$  Predictor dominance shares", loc="left", fontsize=12)
    for ax in axes:
        ax.axvline(75, color="grey", ls="--", lw=1)
        ax.set_xlabel("Hotspot percentile threshold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS06_hotspot_threshold.pdf", dpi=300, bbox_inches="tight")
