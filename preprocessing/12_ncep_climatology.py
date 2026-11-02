"""NCEP/NCAR Reanalysis-1 net heat flux, E-P and wind-stress curl, 1970-2014
monthly climatology on the 1x1 degree grid (used for Figure S11).

Downloads the monthly surface fields from NOAA PSL if not present.

Output: RAW_DIR/ncep/ncep_flux_climatology_1970_2014.nc
"""

import numpy as np
import requests
import xarray as xr

from common.paths import RAW_DIR

NCEP_DIR = RAW_DIR / "ncep"
OUT_FILE = NCEP_DIR / "ncep_flux_climatology_1970_2014.nc"
BASE_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface_gauss"
VARS = ["uflx", "vflx", "nswrs", "nlwrs", "lhtfl", "shtfl", "prate"]
LV = 2.5e6  # latent heat of vaporisation [J kg-1]
TARGET_LAT = np.arange(-89.5, 90, 1)
TARGET_LON = np.arange(0.5, 360, 1)


def download():
    NCEP_DIR.mkdir(parents=True, exist_ok=True)
    for v in VARS:
        out = NCEP_DIR / f"{v}.sfc.mon.mean.nc"
        if out.exists():
            continue
        r = requests.get(f"{BASE_URL}/{out.name}", stream=True, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def climatology(var):
    da = xr.open_dataset(NCEP_DIR / f"{var}.sfc.mon.mean.nc")[var]
    clim = da.sel(time=slice("1970-01-01", "2014-12-31")).groupby("time.month").mean("time")
    # bilinear to 1x1, closing the longitude wrap
    clim = clim.sortby("lat")
    wrap = clim.isel(lon=0).assign_coords(lon=clim.lon.values[0] + 360.0)
    clim = xr.concat([clim, wrap], dim="lon").sortby("lon")
    return clim.interp(lat=TARGET_LAT, lon=TARGET_LON).load()


if __name__ == "__main__":
    download()
    c = {v: climatology(v) for v in VARS}

    # NCEP fluxes are positive upward; Qnet is positive into the ocean
    qnet = -(c["nswrs"] + c["nlwrs"] + c["lhtfl"] + c["shtfl"])
    ep = c["lhtfl"] / LV - c["prate"]
    # uflx/vflx are the stress on the atmosphere; flip to stress on the ocean
    tauu, tauv = -c["uflx"], -c["vflx"]
    m_per_deg = 111320.0
    curl = (tauv.differentiate("lon") / (m_per_deg * np.cos(np.deg2rad(tauv.lat)))
            - tauu.differentiate("lat") / m_per_deg)

    out = xr.Dataset({"obs_hflux_ncep": qnet, "obs_ep_ncep": ep, "obs_tauu_ncep": tauu,
                      "obs_tauv_ncep": tauv, "obs_curl_ncep": curl})
    for v, units in [("obs_hflux_ncep", "W m-2"), ("obs_ep_ncep", "kg m-2 s-1"),
                     ("obs_tauu_ncep", "N m-2"), ("obs_tauv_ncep", "N m-2"),
                     ("obs_curl_ncep", "N m-3")]:
        out[v].attrs["units"] = units
    out.to_netcdf(OUT_FILE)
