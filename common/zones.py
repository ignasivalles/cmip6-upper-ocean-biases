"""Basin x latitude-zone means used by the local-forcing analysis (Figure 2)."""

import numpy as np
import pandas as pd
import xarray as xr

from .paths import REGION_FILE

ZONES = {
    "SO":   (-65, -40),
    "S_ST": (-40, -20),
    "S_T":  (-20, -5),
    "Eq":   (-5, 5),
    "N_T":  (5, 20),
    "N_ST": (20, 40),
    "N_HL": (40, 65),
}
BASINS = ["Atlantic", "Pacific", "Indian"]
INDIAN_ZONES = ["SO", "S_ST", "S_T", "Eq", "N_T"]

PREDICTORS = ["bias_Ft", "bias_Fs", "bias_curl"]
TARGETS = ["bias_sst", "bias_D1", "bias_a2", "bias_b3"]


def _lon_mask(lon, basin):
    if basin == "Atlantic":
        return (lon >= 290) | (lon <= 20)
    lo, hi = {"Pacific": (120, 290), "Indian": (20, 120)}[basin]
    return (lon >= lo) & (lon <= hi)


def quality_mask(ds):
    """Mask cells with unrealistic b3 bias (|bias| > 1e-3 kg m-4)."""
    return ds.where(np.isfinite(ds)).where((ds.bias_b3 > -1e-3) & (ds.bias_b3 < 1e-3))


def extended_basins(ds):
    """RECCAP2 basin masks extended 5 degrees into the Southern Ocean and 20
    degrees poleward in the north."""
    regions = xr.open_dataset(REGION_FILE).interp_like(ds, method="nearest")
    lat2d, lon2d = xr.broadcast(ds.lat, ds.lon)
    out = {}
    for basin in BASINS:
        mask = regions[basin.lower()] > 0
        lat_min = float(ds.lat.where(mask.any("lon"), drop=True).min())
        lat_max = float(ds.lat.where(mask.any("lon"), drop=True).max())
        lons = _lon_mask(lon2d, basin)
        south = (lat2d <= lat_min + 5) & lons
        north = (lat2d >= lat_max - 5) & (lat2d <= lat_max + 20) & lons
        out[basin] = mask | south | north
    return out


def basin_subsets(ds):
    ds = quality_mask(ds)
    return {b: ds.where(m, drop=True) for b, m in extended_basins(ds).items()}


def standardize_within_model(ds, variables):
    """Standardise each variable per model over space and time."""
    ds = ds.copy()
    for v in variables:
        dims = ("lat", "lon", "month")
        ds[v] = ((ds[v] - ds[v].mean(dims)) / ds[v].std(dims)).where(np.isfinite(ds[v]))
    return ds


def zone_tables(ds, variables=PREDICTORS + TARGETS):
    """{(basin, zone): DataFrame (model x variables)} of zone-mean values."""
    tables = {}
    for basin, sub in basin_subsets(ds).items():
        sub = standardize_within_model(sub, variables)
        for zone, (lat_min, lat_max) in ZONES.items():
            if basin == "Indian" and zone not in INDIAN_ZONES:
                continue
            box = sub.where(_lon_mask(sub.lon, basin) & (sub.lat >= lat_min) & (sub.lat <= lat_max),
                            drop=True)
            means = box[variables].mean(("lat", "lon", "month"))
            tables[basin, zone] = pd.DataFrame({v: means[v].to_pandas() for v in variables})
    return tables
