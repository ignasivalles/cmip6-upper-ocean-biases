"""Write the compact datasets used by all figure scripts (the Zenodo archive).

    cmip6_upper_ocean_n34.nc   model and observed fields, 34 models, monthly climatology
    obs_shdr_iap_cora.nc       SHDR parameters for the IAPv4 and CORA references
    ncep_r1_fluxes.nc          NCEP/NCAR R1 net heat flux, E-P and wind-stress curl

Biases are not stored: they are model minus observation and are computed
on loading (common.data.load_ensemble).
"""

import argparse
import shutil

import xarray as xr

from common.paths import DATA_DIR

MODEL_VARS = ["a1", "a2", "D1", "b3", "Ft", "Fs", "curl", "sst", "sos", "tauu", "hflux", "ep"]
OBS_VARS = {  # name in the merged file -> name in the release file
    "a1_obs": "a1_obs", "a2_obs": "a2_obs", "D1_obs": "D1_obs", "b3_obs": "b3_obs",
    "Ft_obs": "Ft_obs", "Fs_obs": "Fs_obs", "curl_obs": "curl_obs",
    "obs_sst": "sst_obs", "obs_sos": "sos_obs", "obs_tauu": "tauu_obs",
    "obs_hflux": "hflux_obs", "obs_ep": "ep_obs",
}
SHDR_OBS = ["a1_obs", "a2_obs", "D1_obs", "b3_obs"]


def _encode(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def export_ensemble(merged_file, out):
    ds = xr.open_dataset(merged_file).rename({"time": "month"})
    ds = ds[MODEL_VARS + list(OBS_VARS)].rename(OBS_VARS)
    ds.attrs = {
        "title": "CMIP6 upper-ocean structure (SHDR) and surface forcing, 1970-2014 climatology",
        "references": "EN4 (SHDR), ERA5 (fluxes, SST), ORAS5 (wind stress, SSS)",
        "note": "bias = model - obs; a1 is potential density minus 1000 kg m-3",
    }
    ds.to_netcdf(out, encoding=_encode(ds))


def export_obs_products(iap_file, cora_file, out):
    parts = []
    for name, f in [("IAP", iap_file), ("CORA", cora_file)]:
        ds = xr.open_dataset(f)[SHDR_OBS].reset_coords(drop=True).rename({"time": "month"})
        parts.append(ds.rename({v: f"{v}_{name}" for v in SHDR_OBS}))
    ds = xr.merge(parts)
    ds.attrs = {"title": "SHDR parameters for IAPv4 and CORA, 1970-2014 climatology"}
    ds.to_netcdf(out, encoding=_encode(ds))


def export_ncep(ncep_file, out):
    ds = xr.open_dataset(ncep_file)[["obs_hflux_ncep", "obs_ep_ncep", "obs_curl_ncep"]]
    ds = ds.rename({"obs_hflux_ncep": "hflux", "obs_ep_ncep": "ep", "obs_curl_ncep": "curl"})
    ds.attrs = {"title": "NCEP/NCAR Reanalysis-1 surface fluxes, 1970-2014 climatology"}
    ds.to_netcdf(out, encoding=_encode(ds))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", required=True, help="output of 11_merge_models.py")
    p.add_argument("--iap", required=True, help="output of 06_build_bias_dataset.py --obs IAP")
    p.add_argument("--cora", required=True, help="output of 06_build_bias_dataset.py --obs CORA")
    p.add_argument("--ncep", required=True, help="output of 12_ncep_climatology.py")
    p.add_argument("--raw-gradient", help="output of 13_raw_density_gradient.py (copied as is)")
    a = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_ensemble(a.merged, DATA_DIR / "cmip6_upper_ocean_n34.nc")
    export_obs_products(a.iap, a.cora, DATA_DIR / "obs_shdr_iap_cora.nc")
    export_ncep(a.ncep, DATA_DIR / "ncep_r1_fluxes.nc")
    if a.raw_gradient:
        shutil.copy(a.raw_gradient, DATA_DIR / "b3_raw_gradient.nc")
