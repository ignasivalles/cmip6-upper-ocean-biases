"""
Compute E-P (evaporation minus precipitation) bias for all 29 CMIP6 models
and append bias_ep, ep (model), and obs_ep (ERA5) to allbias_29CMIP6_EN4.nc.

Units throughout: kg m-2 s-1  (positive = net freshwater flux to atmosphere)

Sign conventions
----------------
CMIP6:
    evspsbl  kg m-2 s-1   positive = upward evaporation   ✓ use as-is
    pr       kg m-2 s-1   positive = downward precip       ✓ use as-is
    E-P = evspsbl - pr

ERA5 (m of water equivalent, monthly means):
    e        m/day  NEGATIVE = evaporation (upward)  → E = -e × 1000/86400
    tp       m/day  POSITIVE = precipitation          → P =  tp × 1000/86400
    E-P = (-e - tp) × 1000/86400

Input:
    RAW_DIR/CMIP6_forcings_RG_360/<model>_atm_forcing.nc
    RAW_DIR/ERA5_data/ERA5_fluxes_rg/ERA5_fluxes_*.nc
    INTERIM_DIR/allbias_29CMIP6_EN4.nc

Output:
    INTERIM_DIR/allbias_29CMIP6_EN4.nc  (variables appended in-place)

Usage:
    python preprocessing/07_add_ep_bias.py
"""

from pathlib import Path
import numpy as np
import xarray as xr

from common.paths import INTERIM_DIR, RAW_DIR

CMIP6_FORC_DIR = RAW_DIR / "CMIP6_forcings_RG_360"
ERA5_FLUX_DIR  = RAW_DIR / "ERA5_data" / "ERA5_fluxes_rg"
ALLBIAS_FILE   = INTERIM_DIR / "allbias_29CMIP6_EN4.nc"

# Conversion: m/day  →  kg m-2 s-1  (1 m water = 1000 kg/m²; 1 day = 86400 s)
M_DAY_TO_KG   = 1000.0 / 86400.0

PERIOD_START  = "1970"
PERIOD_END    = "2014"


# ── ERA5 obs E-P climatology ───────────────────────────────────────────────────

def compute_era5_ep():
    """
    Load ERA5 flux files, select 1970-2014, compute monthly climatology (12 months).
    Returns DataArray (month=1-12, lat, lon) in kg m-2 s-1.
    """
    files = sorted(ERA5_FLUX_DIR.glob("ERA5_fluxes_*_rg.nc"))
    print(f"  ERA5 files found: {[f.name for f in files]}")

    # Accumulate monthly sums per calendar month to avoid loading all at once
    month_sums  = {}   # {1..12: running sum array}
    month_count = {}

    for f in files:
        ds_f = xr.open_dataset(f)
        # Drop ERA5-specific coords that prevent concatenation
        drop = [v for v in ["number", "expver"] if v in ds_f.coords]
        if drop:
            ds_f = ds_f.drop_vars(drop)

        # ERA5 newer files use 'valid_time' instead of 'time'
        time_dim = "time" if "time" in ds_f.dims else "valid_time"
        if time_dim == "valid_time":
            ds_f = ds_f.rename({"valid_time": "time"})

        # Select 1970-2014
        ds_f = ds_f.sel(time=slice(PERIOD_START, PERIOD_END)).load()
        print(f"    {f.name}: {len(ds_f.time)} months")

        ep_f = (-ds_f["e"] - ds_f["tp"]) * M_DAY_TO_KG   # (time, lat, lon)

        for mo in range(1, 13):
            sel = ep_f.sel(time=ep_f.time.dt.month == mo).values
            if sel.shape[0] == 0:
                continue
            if mo not in month_sums:
                month_sums[mo]  = sel.sum(axis=0)
                month_count[mo] = sel.shape[0]
            else:
                month_sums[mo]  += sel.sum(axis=0)
                month_count[mo] += sel.shape[0]

        ds_f.close()

    # Build climatology DataArray
    lat  = xr.open_dataset(files[0]).lat.values
    lon  = xr.open_dataset(files[0]).lon.values
    clim = np.stack([month_sums[mo] / month_count[mo] for mo in range(1, 13)], axis=0)
    ep_clim = xr.DataArray(clim, dims=["month", "lat", "lon"],
                           coords={"month": np.arange(1, 13), "lat": lat, "lon": lon})
    ep_clim = ep_clim.rename("obs_ep")
    ep_clim.attrs["units"]     = "kg m-2 s-1"
    ep_clim.attrs["long_name"] = "E-P bias (ERA5 climatology 1970-2014)"
    ep_clim.attrs["note"]      = "E-P = (-e_ERA5 - tp_ERA5) * 1000/86400; positive = net evaporation"

    print(f"  ERA5 E-P range: [{float(ep_clim.min()):.2e}, {float(ep_clim.max()):.2e}] kg/m2/s")
    return ep_clim.load()


# ── CMIP6 E-P climatologies ───────────────────────────────────────────────────

# Name aliases: allbias model name → forcing file stem prefix
MODEL_FILE_ALIASES = {
    "EC-Earth3-Veg-LR": "EC-Earth3-Veg",
    "GISS-E2-1-G":       "GISS-E2",
    "MPI-ESM1-2-LR":     "MPI-ESM2",
    "MRI-ESM2-0":        "MRI-ESM2",
    "UKESM1-1-LL":       "UKESM1-1",
}


def compute_cmip6_ep(models):
    """
    For each model, load evspsbl and pr from the forcing file and compute
    E-P = evspsbl - pr.  The forcing files are already 12-month climatologies.
    Returns dict: model_name → DataArray (month=1-12, lat, lon).
    """
    ep_dict = {}
    missing = []

    for m in models:
        stem  = MODEL_FILE_ALIASES.get(m, m)
        fpath = CMIP6_FORC_DIR / f"{stem}_atm_forcing.nc"
        if not fpath.exists():
            print(f"  WARNING: forcing file not found for {m} (tried {fpath.name})")
            missing.append(m)
            continue

        ds = xr.open_dataset(fpath).load()

        if "evspsbl" not in ds or "pr" not in ds:
            print(f"  WARNING: evspsbl or pr missing in {m}")
            missing.append(m)
            continue

        ep = ds["evspsbl"] - ds["pr"]   # kg m-2 s-1, positive = net evap
        # Normalise time dimension name to 'month'
        if "time" in ep.dims:
            ep = ep.assign_coords(time=np.arange(1, len(ep.time) + 1))
            ep = ep.rename({"time": "month"})
        ep.attrs["units"]     = "kg m-2 s-1"
        ep.attrs["long_name"] = f"E-P climatology {m}"
        ep_dict[m] = ep

    print(f"  CMIP6 E-P computed for {len(ep_dict)} models; missing: {missing}")
    return ep_dict, missing


# ── Build bias and append to allbias ─────────────────────────────────────────

def main():
    print("Loading allbias...")
    ab = xr.open_dataset(ALLBIAS_FILE).load()
    ab = (ab.rename({"time": "month"})
          if "time" in ab.dims and "month" not in ab.dims else ab)

    models = list(ab.model.values)
    print(f"  Models in allbias: {len(models)}")

    # Check if already computed
    if "bias_ep" in ab:
        print("  bias_ep already present in allbias — re-computing to overwrite.")

    # ── ERA5 obs ──────────────────────────────────────────────────────────────
    print("\nComputing ERA5 E-P climatology...")
    obs_ep = compute_era5_ep()   # (month, lat, lon)

    # Ensure lat/lon match allbias grid (both should be 1°×1° regridded)
    obs_ep = obs_ep.reindex(lat=ab.lat, lon=ab.lon, method="nearest", tolerance=0.01)

    # ── CMIP6 per model ───────────────────────────────────────────────────────
    print("\nComputing CMIP6 E-P per model...")
    ep_dict, missing = compute_cmip6_ep(models)

    # Stack into (model, month, lat, lon)
    ep_list = []
    for m in models:
        if m in ep_dict:
            da = ep_dict[m]
            da = da.reindex(lat=ab.lat, lon=ab.lon, method="nearest", tolerance=0.01)
            ep_list.append(da.assign_coords(model=m))
        else:
            # Fill with NaN for missing models
            blank = xr.full_like(obs_ep, fill_value=np.nan).assign_coords(model=m)
            ep_list.append(blank)

    ep_all = xr.concat(ep_list, dim="model")   # (model, month, lat, lon)
    ep_all.attrs["units"]     = "kg m-2 s-1"
    ep_all.attrs["long_name"] = "Evaporation minus precipitation (CMIP6)"

    bias_ep = ep_all - obs_ep   # (model, month, lat, lon)
    bias_ep.attrs["units"]     = "kg m-2 s-1"
    bias_ep.attrs["long_name"] = "E-P bias (CMIP6 - ERA5)"

    print(f"\n  bias_ep range: [{float(bias_ep.min()):.2e}, {float(bias_ep.max()):.2e}]")

    # ── Append to allbias and save ────────────────────────────────────────────
    print("\nAppending to allbias and saving...")
    ab = ab.rename({"month": "time"})   # restore original dim name

    # Rename month dim in new variables to match allbias
    obs_ep_out = obs_ep.rename({"month": "time"})
    ep_out     = ep_all.rename({"month": "time"})
    bias_ep_out = bias_ep.rename({"month": "time"})

    ab["obs_ep"]  = obs_ep_out
    ab["ep"]      = ep_out
    ab["bias_ep"] = bias_ep_out

    ab.to_netcdf(ALLBIAS_FILE)
    print(f"  Saved: {ALLBIAS_FILE}")
    print("  New variables: obs_ep, ep, bias_ep")


if __name__ == "__main__":
    main()
