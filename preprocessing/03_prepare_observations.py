"""
Prepare all observational and reanalysis climatologies needed by the
climatologies (04_obs_climatologies.py) and bias dataset (06_build_bias_dataset.py).

Steps:
  1. Regrid ERA5 SST monthly files to 1°×1°
  2. Regrid ERA5 10-m wind monthly files to 1°×1°
  3. Compute wind stress curl from ERA5 winds
  4. Regrid ORAS5 heat-flux monthly files to 1°×1°
  5. Regrid CORA/EN4 SHDR monthly files to 1°×1°
  6. Compute wind stress curl from ORAS5 taux/tauy
  7. Compute wind stress magnitude (WSM) from ORAS5 taux/tauy
  8. Compute sea-surface density flux (SSDflux) from ERA5 + ORAS5

All inputs and outputs live under RAW_DIR, in the corresponding *_rg/ directories.

Run from the project root:
    python preprocessing/03_prepare_observations.py
"""

import os
from glob import glob
from pathlib import Path

import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds
import numpy as np

from cmip6_tools import regrid_models_simple
from common.paths import RAW_DIR
from flux_tools import (
    dll_dist,
    calculate_and_append_curl,
    calculate_and_append_wsm,
    calculate_SSDflux_ERA5_ORAS5_mfdataset,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


ERA5_DIR         = RAW_DIR / "ERA5_data"
ERA5_SST_DIR     = ERA5_DIR / "ERA5_sst_rg"
ERA5_WIND_DIR    = ERA5_DIR             # ERA5_10mWind*.nc files are here
ERA5_FLUXES_DIR  = ERA5_DIR / "ERA5_fluxes_rg"
ERA5_SSDFLUX_DIR = ERA5_DIR / "SSDflux_rg"

ORAS5_DIR         = RAW_DIR / "ORAS5"
ORAS5_HFLUX_IN    = ORAS5_DIR / "ORAS5_original_data" / "ORAS5_hflux"
ORAS5_HFLUX_OUT   = ORAS5_DIR / "ORAS5_hflux_rg"
ORAS5_TAUX_RG     = ORAS5_DIR / "ORAS5_original_data" / "ORAS5_wind_stress" / "taux_rg"
ORAS5_TAUY_RG     = ORAS5_DIR / "ORAS5_original_data" / "ORAS5_wind_stress" / "tauy_rg"
ORAS5_CURL_OUT    = ORAS5_DIR / "ORAS5_wind_stress_curl"
ORAS5_WSM_OUT     = ORAS5_DIR / "ORAS5_wind_stress_magnitude_rg"
ORAS5_SOS_DIR     = ORAS5_DIR / "ORAS5_sosaline_rg"

CORA_IN  = RAW_DIR / "CORA_SHDR"
CORA_OUT = RAW_DIR / "CORA_SHDR_rg"

# ---------------------------------------------------------------------------


def regrid_era5_sst():
    """Regrid ERA5 SST files to 1°×1°."""
    print("--- ERA5 SST regridding ---")
    files = glob(str(ERA5_DIR / "ERA5_sst*.nc"))
    for f in files:
        out = f.replace(".nc", "_rg.nc")
        if os.path.exists(out):
            print(f"  skip (exists): {out}")
            continue
        print(f"  {f}")
        xr.open_dataset(f).pipe(regrid_models_simple).to_netcdf(out)
    print("  ERA5 SST done.\n")


def regrid_era5_wind():
    """Regrid ERA5 10-m wind files to 1°×1°."""
    print("--- ERA5 wind regridding ---")
    files = glob(str(ERA5_DIR / "ERA5_10mWind*.nc"))
    for f in files:
        out = f.replace(".nc", "_rg.nc")
        if os.path.exists(out):
            print(f"  skip (exists): {out}")
            continue
        print(f"  {f}")
        xr.open_dataset(f).pipe(regrid_models_simple).to_netcdf(out)
    print("  ERA5 wind done.\n")


def compute_era5_curl():
    """Compute wind stress curl from regrided ERA5 10-m wind files."""
    print("--- ERA5 wind stress curl ---")
    files = glob(str(ERA5_DIR / "ERA5_10mWind*_rg.nc"))
    for f in files:
        ds = xr.open_dataset(f)
        ds_full = generate_grid_ds(ds, {"Y": "lat", "X": "lon"})
        grid = Grid(ds_full, periodic=["X"])

        dlong = grid.diff(ds_full.lon, "X", boundary="extend", fill_value=np.nan)
        dlatg = grid.diff(ds_full.lat, "Y", boundary="extrapolate", fill_value=np.nan)
        dlong[0] = dlong[1]

        dx, dy = dll_dist(dlong, dlatg, ds.lon, ds.lat)

        du_dy = grid.diff(ds.v10, "Y", boundary="fill", fill_value=np.nan) / dy
        dv_dx = grid.diff(ds.u10, "X") / dx

        du_dy["lat_left"] = dv_dx["lat"].values
        dv_dx["lon_left"] = du_dy["lon"].values
        du_dy = np.squeeze(du_dy.isel(lat=0).drop("lat"))
        du_dy = du_dy.rename({"lat_left": "lat"})
        dv_dx = dv_dx.rename({"lon_left": "lon"})

        curl = dv_dx - du_dy
        years = f[-19:-3]
        out = str(ERA5_DIR / f"ERA5_curl_{years}.nc")
        print(f"  {years}")
        curl.rename("curl").to_netcdf(out)
    print("  ERA5 curl done.\n")


def regrid_oras5_hflux():
    """Regrid ORAS5 monthly heat-flux files to 1°×1°."""
    print("--- ORAS5 hflux regridding ---")
    ORAS5_HFLUX_OUT.mkdir(exist_ok=True)
    files = glob(str(ORAS5_HFLUX_IN / "*.nc"))
    for f in files:
        ds = xr.open_dataset(f)
        t = ds.time_counter.values
        t_str = str(np.datetime_as_string(t, unit="M")[0])
        ym = t_str[:4] + t_str[5:7]
        out = str(ORAS5_HFLUX_OUT / f"ORAS5_hflux_{ym}_rg.nc")
        if os.path.exists(out):
            print(f"  skip (exists): {out}")
            continue
        print(f"  {os.path.basename(f)}")
        regrid_models_simple(ds).to_netcdf(out)
    print("  ORAS5 hflux done.\n")


def regrid_cora_shdr():
    """Regrid CORA/EN4 SHDR monthly files to 1°×1°."""
    print("--- CORA SHDR regridding ---")
    CORA_OUT.mkdir(exist_ok=True)
    files = glob(str(CORA_IN / "*.nc"))
    for f in files:
        ds = xr.open_dataset(f)
        t = ds.time.values
        t_str = str(np.datetime_as_string(t, unit="M")[0])
        ym = t_str[:4] + t_str[5:7]
        out = str(CORA_OUT / f"CORA_SHDR_{ym}_rg.nc")
        if os.path.exists(out):
            print(f"  skip (exists): {out}")
            continue
        print(f"  {os.path.basename(f)}")
        regrid_models_simple(ds).to_netcdf(out)
    print("  CORA SHDR done.\n")


def compute_oras5_curl():
    """Compute wind stress curl from ORAS5 taux/tauy."""
    print("--- ORAS5 wind stress curl ---")
    ORAS5_CURL_OUT.mkdir(exist_ok=True)
    files_u = sorted(glob(str(ORAS5_TAUX_RG / "*.nc")))
    files_v = sorted(glob(str(ORAS5_TAUY_RG / "*.nc")))
    calculate_and_append_curl(files_u, files_v, str(ORAS5_CURL_OUT))
    print("  ORAS5 curl done.\n")


def compute_oras5_wsm():
    """Compute wind stress magnitude from ORAS5 taux/tauy."""
    print("--- ORAS5 wind stress magnitude ---")
    ORAS5_WSM_OUT.mkdir(exist_ok=True)
    files_u = sorted(glob(str(ORAS5_TAUX_RG / "*.nc")))
    files_v = sorted(glob(str(ORAS5_TAUY_RG / "*.nc")))
    calculate_and_append_wsm(files_u, files_v, str(ORAS5_WSM_OUT))
    print("  ORAS5 WSM done.\n")


def compute_ssdflux():
    """Compute sea-surface density flux from ERA5 fluxes and ORAS5 SSS."""
    print("--- SSDflux computation ---")
    ERA5_SSDFLUX_DIR.mkdir(exist_ok=True)
    calculate_SSDflux_ERA5_ORAS5_mfdataset(
        str(ERA5_FLUXES_DIR),
        str(ORAS5_HFLUX_OUT),
        str(ERA5_SST_DIR),
        str(ORAS5_SOS_DIR),
        str(ERA5_SSDFLUX_DIR),
    )
    print("  SSDflux done.\n")


if __name__ == "__main__":
    regrid_era5_sst()
    regrid_era5_wind()
    compute_era5_curl()
    regrid_oras5_hflux()
    regrid_cora_shdr()
    compute_oras5_curl()
    compute_oras5_wsm()
    compute_ssdflux()
    print("=== All observation preparation complete ===")
