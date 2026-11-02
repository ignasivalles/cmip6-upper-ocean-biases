"""
Build the main bias dataset used by all figure scripts.

Merges CMIP6 ensemble climatologies with an observational SHDR climatology,
computes model biases (model − obs) and inter-model spread (std), and
adds atmospheric forcing biases (SST, wind stress, heat flux).

Supported observation datasets:
    EN4   — RAW_DIR/obs_clim_ens/SHDR_EN4_clim.nc   (shdr only)
    CORA  — RAW_DIR/obs_clim_ens/SHDR_CORA_clim.nc  (shdr only)
    IAP   — RAW_DIR/obs_clim_ens/SHDR_IAP_clim.nc   (shdr only)

Output:
    INTERIM_DIR/allbias_29CMIP6_{OBS}.nc

Usage:
    python preprocessing/06_build_bias_dataset.py --obs EN4
    python preprocessing/06_build_bias_dataset.py --obs CORA
    python preprocessing/06_build_bias_dataset.py --obs IAP
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr

from common.paths import INTERIM_DIR, RAW_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


CMIP6_CLIM   = RAW_DIR / "cmip6_clim_ens"
OBS_CLIM     = RAW_DIR / "obs_clim_ens"
OBS_SSDFLUX  = RAW_DIR / "ERA5_data" / "SSDflux_rg"
OBS_CURL     = RAW_DIR / "ORAS5" / "ORAS5_wind_stress_curl_rg"
OBS_WSM      = RAW_DIR / "ORAS5" / "ORAS5_wind_stress_magnitude_rg"

CMIP6_ATM_CLIM = RAW_DIR / "CMIP6_forcings_RG_360" / "clim" / "CMIP6_atm_forcing_climatology.nc"
OBS_SST_FILE   = RAW_DIR / "ERA5_data" / "ERA5_sst_rg" / "ERA5_sst_climatology_1970_2014.nc"
OBS_TAUU_FILE  = RAW_DIR / "ORAS5" / "ORAS5_tau_rg" / "taux_rg" / "ORAS5_taux_climatology_1970_2014.nc"
OBS_HFLUX_FILE = RAW_DIR / "ORAS5" / "ORAS5_hflux_rg" / "ORAS5_hflux_climatology_1970_2014.nc"
OBS_SOS_FILE   = RAW_DIR / "ORAS5" / "ORAS5_sosaline_rg" / "ORAS5_sos_climatology_1970_2014.nc"

# Observation datasets and which variables they provide
OBS_REGISTRY = {
    "EN4":  {"file": OBS_CLIM / "SHDR_EN4_clim.nc"},
    "CORA": {"file": OBS_CLIM / "SHDR_CORA_clim.nc"},
    "IAP":  {"file": OBS_CLIM / "SHDR_IAP_clim.nc"},
}

# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Build allbias dataset for a given obs reference.")
    parser.add_argument(
        "--obs",
        choices=list(OBS_REGISTRY.keys()),
        default="EN4",
        help="Observational dataset to use as reference (default: EN4)",
    )
    return parser.parse_args()


def check_inputs(obs_name):
    """Verify all required input files exist before starting."""
    obs_file = OBS_REGISTRY[obs_name]["file"]
    required = {
        "CMIP6 SHDR":       CMIP6_CLIM / "shdr.nc",
        "CMIP6 curl":       CMIP6_CLIM / "curl.nc",
        "CMIP6 SSDflux":    CMIP6_CLIM / "ssdflux.nc",
        "CMIP6 WSM":        CMIP6_CLIM / "wsm.nc",
        f"OBS SHDR ({obs_name})": obs_file,
        "OBS SSDflux":      OBS_SSDFLUX / "SSDflux_ERA5_ORAS5.nc",
        "CMIP6 atm clim":   CMIP6_ATM_CLIM,
        "OBS SST":          OBS_SST_FILE,
        "OBS tauu":         OBS_TAUU_FILE,
        "OBS hflux":        OBS_HFLUX_FILE,
        "OBS SSS":          OBS_SOS_FILE,
    }
    missing = [f"  {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        print("ERROR — missing input files:")
        print("\n".join(missing))
        sys.exit(1)

    # Check that at least some curl / wsm files exist
    for label, pattern in [("OBS curl", str(OBS_CURL / "*.nc")),
                             ("OBS WSM",  str(OBS_WSM  / "*.nc"))]:
        if not glob(pattern):
            print(f"ERROR — no files found matching: {pattern}")
            sys.exit(1)

    print(f"All input files found. Using observation: {obs_name}\n")


def load_inputs(obs_name):
    """Load all climatology datasets."""
    print("Loading CMIP6 climatologies...")
    cmip6_shdr    = xr.open_dataset(CMIP6_CLIM / "shdr.nc")
    cmip6_curl    = xr.open_dataset(CMIP6_CLIM / "curl.nc")
    cmip6_ssdflux = xr.open_dataset(CMIP6_CLIM / "ssdflux.nc")
    cmip6_wsm     = xr.open_dataset(CMIP6_CLIM / "wsm.nc")

    print(f"Loading {obs_name} observational climatology...")
    obs_shdr = xr.open_dataset(OBS_REGISTRY[obs_name]["file"])

    print("Loading forcing climatologies (curl, WSM, SSDflux)...")
    obs_curl    = xr.open_mfdataset(sorted(glob(str(OBS_CURL / "*.nc")))).load()
    obs_wsm     = xr.open_mfdataset(sorted(glob(str(OBS_WSM / "*.nc")))).load()
    obs_ssdflux = xr.open_dataset(OBS_SSDFLUX / "SSDflux_ERA5_ORAS5.nc").load()

    obs_curl_clim    = obs_curl.groupby("time_counter.month").mean("time_counter").rename(month="time")
    obs_wsm_clim     = obs_wsm.groupby("time_counter.month").mean("time_counter").rename(month="time")
    obs_ssdflux_clim = obs_ssdflux.groupby("time.month").mean("time").rename(month="time")

    return (cmip6_shdr, cmip6_curl, cmip6_ssdflux, cmip6_wsm,
            obs_shdr, obs_curl_clim, obs_wsm_clim, obs_ssdflux_clim)


def compute_biases(obs_name, cmip6_shdr, cmip6_curl, cmip6_ssdflux, cmip6_wsm,
                   obs_shdr, obs_curl_clim, obs_wsm_clim, obs_ssdflux_clim):
    """Extract variables, compute model biases and inter-model spread."""
    print("Computing biases...")

    def _param(name):
        m = cmip6_shdr.sel(param=name).shdr.drop_vars("param")
        o = obs_shdr.sel(param=name).shdr.drop_vars("param")
        return m, o, m - o, m.std(dim="model")

    a1, a1_obs, diff_a1, std_a1 = _param("a1")
    a3, a3_obs, diff_a3, std_a3 = _param("a3")
    D1, D1_obs, diff_D1, std_D1 = _param("D1")
    b3, b3_obs, diff_b3, std_b3 = _param("b3")

    # a2 is stored as negative; take absolute value
    a2     = abs(cmip6_shdr.sel(param="a2").shdr.drop_vars("param"))
    a2_obs = abs(obs_shdr.sel(param="a2").shdr.drop_vars("param"))
    diff_a2 = a2 - a2_obs
    std_a2  = a2.std(dim="model")

    # Forcing fields
    Ft, Ft_obs = cmip6_ssdflux.Ft, obs_ssdflux_clim.Ft
    Fs, Fs_obs = cmip6_ssdflux.Fs, obs_ssdflux_clim.Fs
    curl, curl_obs = cmip6_curl.curl, obs_curl_clim.curl
    wsm, wsm_obs   = cmip6_wsm.wsm, obs_wsm_clim.wsm

    return dict(
        a1=a1, a2=a2, a3=a3, D1=D1, b3=b3,
        Fs=Fs, Ft=Ft, curl=curl, wsm=wsm,
        a1_obs=a1_obs, a2_obs=a2_obs, a3_obs=a3_obs,
        D1_obs=D1_obs, b3_obs=b3_obs,
        Fs_obs=Fs_obs, Ft_obs=Ft_obs, curl_obs=curl_obs, wsm_obs=wsm_obs,
        bias_a1=diff_a1, bias_a2=diff_a2, bias_a3=diff_a3,
        bias_D1=diff_D1, bias_b3=diff_b3,
        bias_Fs=Fs - Fs_obs, bias_Ft=Ft - Ft_obs,
        bias_curl=curl - curl_obs, bias_wsm=wsm - wsm_obs,
        std_a1=std_a1, std_a2=std_a2, std_a3=std_a3,
        std_D1=std_D1, std_b3=std_b3,
        std_Fs=Fs.std(dim="model"), std_Ft=Ft.std(dim="model"),
        std_curl=curl.std(dim="model"), std_wsm=wsm.std(dim="model"),
    )


def build_dataset(vars_dict):
    """Assemble all variables into a single xr.Dataset."""
    print("Merging into a single dataset...")
    ds = xr.merge([v.rename(k) for k, v in vars_dict.items()])
    for coord in ("number",):
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds


def add_atmospheric_biases(all_bias):
    """Add SST, tauu, and heat-flux biases from CMIP6 atm forcing vs ERA5/ORAS5."""
    print("Adding atmospheric forcing biases (SST, tauu, hflux)...")

    def _open(path):
        ds = xr.open_dataset(path)
        return ds.rename({"month": "time"}) if "month" in ds.dims else ds

    cmip_atm  = _open(CMIP6_ATM_CLIM)
    obs_sst   = _open(OBS_SST_FILE)
    obs_tauu  = _open(OBS_TAUU_FILE)
    obs_hflux = _open(OBS_HFLUX_FILE)
    obs_sos   = _open(OBS_SOS_FILE)

    obs_sst_c   = obs_sst.sst - 273.15   # K → °C
    obs_tauu_v  = obs_tauu.sozotaux
    obs_hflux_v = obs_hflux.sohefldo
    obs_sos_v   = obs_sos.sosaline.squeeze(drop=True)

    new = xr.Dataset({
        "sst":        cmip_atm.tos.rename("sst"),
        "tauu":       cmip_atm.tauu.rename("tauu"),
        "hflux":      cmip_atm.hfds.rename("hflux"),
        "sos":        cmip_atm.sos.squeeze(drop=True).rename("sos"),
        "obs_sst":    obs_sst_c.rename("obs_sst"),
        "obs_tauu":   obs_tauu_v.rename("obs_tauu"),
        "obs_hflux":  obs_hflux_v.rename("obs_hflux"),
        "obs_sos":    obs_sos_v.rename("obs_sos"),
        "bias_sst":   (cmip_atm.tos - obs_sst_c).rename("bias_sst"),
        "bias_tauu":  (cmip_atm.tauu - obs_tauu_v).rename("bias_tauu"),
        "bias_hflux": (cmip_atm.hfds - obs_hflux_v).rename("bias_hflux"),
        "bias_sos":   (cmip_atm.sos.squeeze(drop=True) - obs_sos_v).rename("bias_sos"),
    })

    return xr.merge([all_bias, new])


def main():
    args = parse_args()
    obs_name = args.obs

    output_file = INTERIM_DIR / f"allbias_29CMIP6_{obs_name}.nc"

    print(f"=== Building allbias dataset — obs: {obs_name} ===\n")
    check_inputs(obs_name)

    (cmip6_shdr, cmip6_curl, cmip6_ssdflux, cmip6_wsm,
     obs_shdr, obs_curl_clim, obs_wsm_clim, obs_ssdflux_clim) = load_inputs(obs_name)

    vars_dict = compute_biases(
        obs_name,
        cmip6_shdr, cmip6_curl, cmip6_ssdflux, cmip6_wsm,
        obs_shdr, obs_curl_clim, obs_wsm_clim, obs_ssdflux_clim,
    )

    all_bias = build_dataset(vars_dict)
    all_bias = add_atmospheric_biases(all_bias)

    print(f"\n{all_bias}\n")
    print(f"Saving to {output_file} ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_bias.to_netcdf(str(output_file))
    print("Done.")


if __name__ == "__main__":
    main()
