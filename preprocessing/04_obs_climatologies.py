"""
Compute monthly climatologies (1970–2014) for all observational and
reanalysis datasets used as the bias reference in 06_build_bias_dataset.py.

Inputs (under RAW_DIR) — regridded monthly files from 03_prepare_observations.py:
    ERA5_data/ERA5_sst_rg/          — ERA5 SST        (valid_time, lat, lon)
    ORAS5/ORAS5_hflux_rg/           — ORAS5 heat flux (time_counter, lat, lon)
    ORAS5/ORAS5_tau_rg/taux_rg/     — ORAS5 taux      (time_counter, lat, lon)
    ORAS5/ORAS5_sosaline_rg/        — ORAS5 SSS       (time_counter, lat, lon)

Outputs (same directories, _climatology_1970_2014.nc):
    ERA5_data/ERA5_sst_rg/ERA5_sst_climatology_1970_2014.nc
    ORAS5/ORAS5_hflux_rg/ORAS5_hflux_climatology_1970_2014.nc
    ORAS5/ORAS5_tau_rg/taux_rg/ORAS5_taux_climatology_1970_2014.nc
    ORAS5/ORAS5_sosaline_rg/ORAS5_sos_climatology_1970_2014.nc

Run from the project root:
    python preprocessing/04_obs_climatologies.py
"""

from glob import glob
from pathlib import Path

import xarray as xr

from common.paths import RAW_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


TIME_START = "1970-01"
TIME_END   = "2014-12"

DATASETS = [
    {
        "name":     "ERA5 SST",
        "pattern":  str(RAW_DIR / "ERA5_data" / "ERA5_sst_rg" / "ERA5_sst_*_rg.nc"),
        "time_dim": "valid_time",
        "output":   RAW_DIR / "ERA5_data" / "ERA5_sst_rg" / "ERA5_sst_climatology_1970_2014.nc",
    },
    {
        "name":     "ORAS5 heat flux",
        "pattern":  str(RAW_DIR / "ORAS5" / "ORAS5_hflux_rg" / "ORAS5_hflux_*_rg.nc"),
        "time_dim": "time_counter",
        "output":   RAW_DIR / "ORAS5" / "ORAS5_hflux_rg" / "ORAS5_hflux_climatology_1970_2014.nc",
    },
    {
        "name":     "ORAS5 taux",
        "pattern":  str(RAW_DIR / "ORAS5" / "ORAS5_tau_rg" / "taux_rg" / "ORAS5_taux_*_rg.nc"),
        "time_dim": "time_counter",
        "output":   RAW_DIR / "ORAS5" / "ORAS5_tau_rg" / "taux_rg" / "ORAS5_taux_climatology_1970_2014.nc",
    },
    {
        "name":     "ORAS5 SSS",
        "pattern":  str(RAW_DIR / "ORAS5" / "ORAS5_sosaline_rg" / "ORAS5_sosaline_*_rg.nc"),
        "time_dim": "time_counter",
        "output":   RAW_DIR / "ORAS5" / "ORAS5_sosaline_rg" / "ORAS5_sos_climatology_1970_2014.nc",
    },
]

# ---------------------------------------------------------------------------


def compute_climatology(name, pattern, time_dim, output):
    print(f"--- {name} climatology ---")

    if output.exists():
        print(f"  skip (exists): {output}\n")
        return

    files = sorted(glob(pattern))
    if not files:
        print(f"  WARNING: no files found matching {pattern}, skipping.\n")
        return

    print(f"  Found {len(files)} files. Loading {TIME_START} – {TIME_END}...")
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sel({time_dim: slice(TIME_START, TIME_END)})

    clim = (ds
            .groupby(f"{time_dim}.month")
            .mean(time_dim)
            .rename({"month": "time"}))

    clim.to_netcdf(str(output))
    print(f"  Saved: {output}\n")


def main():
    print(f"=== Computing observational climatologies ({TIME_START} – {TIME_END}) ===\n")
    for ds_cfg in DATASETS:
        compute_climatology(**ds_cfg)
    print("=== All climatologies complete ===")


if __name__ == "__main__":
    main()
