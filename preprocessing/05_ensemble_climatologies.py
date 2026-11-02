"""Monthly 1970-2014 climatologies of all models and observational references,
in the form read by 06_build_bias_dataset.py.

Models (one file per field, models along a `model` dimension):
    RAW_DIR/cmip6_clim_ens/shdr.nc      SHDR parameters (00_shdr_climatology.py)
    RAW_DIR/cmip6_clim_ens/{curl,wsm,ssdflux}.nc   (02_model_forcing_fluxes.py)
    RAW_DIR/CMIP6_forcings_RG_360/clim/CMIP6_atm_forcing_climatology.nc
                                         surface fields (01_regrid_forcing.py)
Observations:
    RAW_DIR/obs_clim_ens/SHDR_{EN4,CORA}_clim.nc   from the monthly SHDR fits
    RAW_DIR/obs_clim_ens/SHDR_IAP_clim.nc          IAPv4 climatology (provided as such)
"""

import re

import xarray as xr

from common.paths import RAW_DIR

PERIOD = slice("1970", "2014")
SHDR_DIR = RAW_DIR / "SHDR_clim" / "RG"
FORCING_DIR = RAW_DIR / "CMIP6_forcings_RG_360"
OUT_MODELS = RAW_DIR / "cmip6_clim_ens"
OUT_OBS = RAW_DIR / "obs_clim_ens"

# file prefixes of the surface-forcing files -> CMIP6 source_id
FORCING_NAMES = {
    "CMCC-CM2": "CMCC-CM2-SR5", "EC-Earth3-Veg": "EC-Earth3-Veg-LR", "FIO-ESM2": "FIO-ESM-2-0",
    "GISS-E2": "GISS-E2-1-G", "MPI-ESM2": "MPI-ESM1-2-LR", "MRI-ESM2": "MRI-ESM2-0",
    "UKESM1-1": "UKESM1-1-LL",
}


def monthly(ds):
    return ds.sel(time=PERIOD).groupby("time.month").mean("time")


def model_name(prefix):
    return FORCING_NAMES.get(prefix, prefix)


def model_climatologies():
    shdr_files = {re.match(r"SHDR_(.+?)_(historical|climatology)", f.name).group(1): f
                  for f in sorted(SHDR_DIR.glob("SHDR_*.nc"))}
    forcing = {model_name(f.name.split("_atm_forcing")[0]): f
               for f in sorted(FORCING_DIR.glob("*_atm_forcing.nc"))}
    models = sorted(set(shdr_files) & set(forcing))

    shdr = xr.concat([xr.open_dataset(shdr_files[m])["shdr"] for m in models], dim="model")
    out = {"shdr": shdr.assign_coords(model=models).to_dataset()}
    for field, suffix in [("curl", "curl"), ("wsm", "wsm"), ("ssdflux", "SSDflux")]:
        parts = []
        for m in models:
            prefix = forcing[m].name.split("_atm_forcing")[0]
            f = FORCING_DIR / f"{suffix}_output" / f"{prefix}_atm_forcing_{suffix}.nc"
            parts.append(monthly(xr.open_dataset(f)))
        out[field] = xr.concat(parts, dim="model").assign_coords(model=models).rename(month="time")

    atm = xr.concat([monthly(xr.open_dataset(forcing[m])) for m in models], dim="model")
    atm = atm.assign_coords(model=models)

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for name, ds in out.items():
        ds.to_netcdf(OUT_MODELS / f"{name}.nc")
    (FORCING_DIR / "clim").mkdir(exist_ok=True)
    atm.to_netcdf(FORCING_DIR / "clim" / "CMIP6_atm_forcing_climatology.nc")


def obs_climatologies():
    OUT_OBS.mkdir(parents=True, exist_ok=True)
    for name, pattern in [("EN4", "EN4_SHDR/RG/SHDR_EN4_gridded_*_RG.nc"),
                          ("CORA", "CORA_SHDR_rg/CORA_SHDR_*_rg.nc")]:
        ds = xr.open_mfdataset(sorted(RAW_DIR.glob(pattern)), combine="by_coords")[["shdr"]]
        monthly(ds).rename(month="time").to_netcdf(OUT_OBS / f"SHDR_{name}_clim.nc")
    iap = xr.open_dataset(RAW_DIR / "OBS" / "RG" / "SHDR_IAP_gridded_climatology_RG.nc")
    iap[["shdr"]].to_netcdf(OUT_OBS / "SHDR_IAP_clim.nc")


if __name__ == "__main__":
    model_climatologies()
    obs_climatologies()
