"""Figures S1-S2: as Figure 1, with IAPv4 and CORA as the observational reference."""

from common.data import load_ensemble
from common.paths import FIG_DIR
from fig1_bias_maps import make_figure

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, obs in [("figS01_iap", "IAP"), ("figS02_cora", "CORA")]:
        make_figure(load_ensemble(obs=obs), obs).savefig(FIG_DIR / f"{name}.pdf", dpi=300,
                                                         bbox_inches="tight")
