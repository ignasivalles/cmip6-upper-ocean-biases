"""Loading the processed ensemble, basin masks and the SAMW mask."""

import numpy as np
import xarray as xr

from .paths import ENSEMBLE_FILE, OBS_PRODUCTS_FILE, REGION_FILE

FIELDS = ["a1", "a2", "D1", "b3", "Ft", "Fs", "curl", "sst", "sos", "tauu", "hflux", "ep"]

# FGOALS-f3-L has no valid Southern Ocean SST; excluded from the Section 5 analyses
SO_EXCLUDE = ["FGOALS-f3-L"]

# The curl fields in the processed file were computed with grid spacing in
# degrees instead of radians, i.e. too small by a factor 180/pi (same factor
# for models and ORAS5). Corrected here.
CURL_FACTOR = 180 / np.pi


def load_ensemble(exclude=(), obs="EN4"):
    """Model fields, observed fields and biases (model - obs), monthly climatology.

    obs="IAP" or "CORA" replaces the EN4 SHDR reference (a1, a2, D1, b3).
    """
    ds = xr.open_dataset(ENSEMBLE_FILE).load()
    if exclude:
        ds = ds.drop_sel(model=[m for m in exclude if m in ds.model.values])
    if obs != "EN4":
        prod = xr.open_dataset(OBS_PRODUCTS_FILE).load()
        for v in ["a1", "a2", "D1", "b3"]:
            ds[f"{v}_obs"] = prod[f"{v}_obs_{obs}"]
    ds["curl"] = ds["curl"] * CURL_FACTOR
    ds["curl_obs"] = ds["curl_obs"] * CURL_FACTOR
    for v in FIELDS:
        ds[f"bias_{v}"] = ds[v] - ds[f"{v}_obs"]
    return ds


def basin_masks(ds):
    """RECCAP2 Atlantic, Pacific and Indian masks on the dataset grid."""
    regions = xr.open_dataset(REGION_FILE).interp_like(ds, method="nearest")
    return {b: regions[b] > 0 for b in ["atlantic", "pacific", "indian"]}


def samw_mask(ds, months=(8, 9), sigma=(26.6, 27.0), lat_max=-20):
    """SAMW formation belt: late-winter surface density in the SAMW class
    (Sallée et al., 2010), from the observed (EN4) SHDR surface density."""
    a1 = ds["a1_obs"].sel(month=list(months)).mean("month")
    return (a1 >= sigma[0]) & (a1 <= sigma[1]) & (a1.lat <= lat_max)


def region_mean(field, mask):
    """Plain mean over the grid cells of a mask (1-degree cells)."""
    return field.where(mask).mean(("lat", "lon"))


def area_mean(field, mask):
    """Area-weighted mean over a mask."""
    w = np.cos(np.deg2rad(field.lat))
    return field.where(mask).weighted(w).mean(("lat", "lon"))


def jet_latitude(tauu):
    """Latitude of the zonal-mean maximum westerly wind stress, 45-70S."""
    so = tauu.where((tauu.lat >= -70) & (tauu.lat <= -45), drop=True)
    return so.max("lon").idxmax("lat")
