"""Figure S7: correlations across models between the Southern Ocean predictors of
Figure 4 (SAMW-belt SST and SSS biases, jet latitude)."""

import matplotlib.pyplot as plt

from common.data import SO_EXCLUDE, load_ensemble, samw_mask
from common.paths import FIG_DIR
from common.plotting import corr_triangle
from fig4_southern_ocean import PREDICTORS, predictor_table

if __name__ == "__main__":
    ds = load_ensemble(exclude=SO_EXCLUDE)
    df = predictor_table(ds, samw_mask(ds))[PREDICTORS]
    print(f"n = {len(df)}\n{df.corr().round(2)}")

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    im = corr_triangle(ax, df, ["SST", "SSS", r"$\phi(\tau_x^{\max})$"], fontsize=12)
    ax.tick_params(labelsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.set_label("Pearson r", fontsize=10)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "figS07_remote_collinearity.pdf", dpi=300, bbox_inches="tight")
