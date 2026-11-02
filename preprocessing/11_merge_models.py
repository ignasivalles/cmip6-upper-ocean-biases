"""Add the models processed with 10_fit_shdr_model.py to the 29-model bias
dataset (06-08), computing their wind-stress curl and surface density fluxes
the same way as for the original models. For REFIT_MODELS, the SHDR parameters
are replaced by those of a 1970-2014 climatology refitted with 00, and for
REFIT_SST the SST-dependent fields by a 1970-2014 climatology from the archive.

Output: INTERIM_DIR/allbias_34CMIP6_EN4.nc
"""

import gsw
import numpy as np
import xarray as xr

from common.paths import INTERIM_DIR, RAW_DIR

BASE_FILE = INTERIM_DIR / "allbias_29CMIP6_EN4.nc"
OUT_FILE = INTERIM_DIR / "allbias_34CMIP6_EN4.nc"
NEW_MODELS = ["HadGEM3-GC31-LL", "CNRM-ESM2-1", "CMCC-ESM2", "CESM2-FV2", "MPI-ESM1-2-HR"]
REFIT_MODELS = ["ACCESS-CM2", "CESM2", "CESM2-WACCM"]
# models whose SST was rebuilt from the CMIP6 archive (RAW_DIR/models/<model>/forcing.nc);
# SST and the surface density fluxes that depend on it are replaced
REFIT_SST = ["ACCESS-CM2"]
SHDR_FIELDS = ["a1", "a2", "a3", "D1", "b3"]
OBS_NAME = dict(a1="a1_obs", a2="a2_obs", a3="a3_obs", D1="D1_obs", b3="b3_obs",
                sst="obs_sst", sos="obs_sos", tauu="obs_tauu", hflux="obs_hflux", ep="obs_ep",
                curl="curl_obs", wsm="wsm_obs", Ft="Ft_obs", Fs="Fs_obs")


def curl_as_original(tauu, tauv):
    """dtau_v/dx - dtau_u/dy with the WGS84 metric of flux_tools.dll_dist. As in
    the original fields, grid spacing enters in degrees, so the result is
    pi/180 times the physical curl (corrected in common.data.load_ensemble)."""
    a, f = 6378137.0, 1 / 298.257223563
    lat = np.deg2rad(tauu.lat.values)
    e2 = 2 * f - f**2
    r_lat = a * (1 - f) / (1 - e2 * np.sin(lat) ** 2) ** 1.5
    r_lon = a / (1 - e2 * np.sin(lat) ** 2) ** 0.5
    dx = float(tauu.lon[1] - tauu.lon[0]) * r_lon * np.cos(lat)
    dy = np.gradient(tauu.lat.values) * r_lat

    v, u = tauv.values, tauu.values
    dv_dx = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2 * dx[None, :, None])
    du_dy = np.gradient(u, axis=1) / dy[None, :, None]
    return xr.DataArray(dv_dx - du_dy, dims=tauu.dims, coords=tauu.coords)


def density_fluxes(sst, sss, hfds, evspsbl, pr):
    """Thermal and haline surface density fluxes (Eqs. 1-2), with E and P
    converted from kg m-2 s-1 to m s-1."""
    rho = gsw.rho(sss.values, sst.values, 0)
    alpha = gsw.alpha(sss.values, sst.values, 0)
    beta = gsw.beta(sss.values, sst.values, 0)
    ep = (evspsbl.values - pr.values) / 1000.0
    ft = -alpha * hfds.values / 3850.0
    fs = rho * beta * sss.values * ep / (1 - sss.values / 1000.0)
    wrap = lambda x: xr.DataArray(x, dims=sst.dims, coords=sst.coords)  # noqa: E731
    return wrap(ft), wrap(fs)


def shdr_fields(name, grid):
    """SHDR parameters of a model fitted with 10_fit_shdr_model.py, on the grid of
    the bias dataset (a1 as sigma, a2 positive, as for the original models)."""
    shdr = xr.open_dataset(RAW_DIR / "models" / name / "shdr.nc").rename(month="time")["shdr"]
    out = {p: shdr.sel(param=p, drop=True) for p in ["a1", "a3", "D1", "b3"]}
    out["a1"] = out["a1"] - 1000.0
    out["a2"] = abs(shdr.sel(param="a2", drop=True))
    return {v: da.reindex(lat=grid.lat, lon=grid.lon, method="nearest", tolerance=0.01)
            for v, da in out.items()}


def refit_fields(name, grid):
    """SHDR parameters from 00_shdr_climatology.py (sigma units, a2 negative)."""
    shdr = xr.open_dataset(INTERIM_DIR / "shdr_refit" / f"SHDR_{name}_climatology_RG.nc")["shdr"]
    out = {p: shdr.sel(param=p, drop=True) for p in ["a1", "a3", "D1", "b3"]}
    out["a2"] = abs(shdr.sel(param="a2", drop=True))
    return {v: da.reindex(lat=grid.lat, lon=grid.lon, method="nearest", tolerance=0.01)
            for v, da in out.items()}


def replace_shdr(base, name):
    """Overwrite the SHDR parameters (and biases) of one model already in base."""
    for v, da in refit_fields(name, base).items():
        base[v].loc[dict(model=name)] = da.transpose(*base[v].sel(model=name).dims).values
        base[f"bias_{v}"].loc[dict(model=name)] = (
            base[v].sel(model=name) - base[OBS_NAME[v]]).transpose(
            *base[f"bias_{v}"].sel(model=name).dims).values
    return base


def forcing_fields(name, grid):
    forc = xr.open_dataset(RAW_DIR / "models" / name / "forcing.nc").rename(month="time")
    ft, fs = density_fluxes(forc.tos, forc.sos, forc.hfds, forc.evspsbl, forc.pr)
    fields = dict(sst=forc.tos, sos=forc.sos, tauu=forc.tauu, hflux=forc.hfds,
                  ep=forc.evspsbl - forc.pr, curl=curl_as_original(forc.tauu, forc.tauv),
                  wsm=np.hypot(forc.tauu, forc.tauv), Ft=ft, Fs=fs)
    return {v: da.reindex(lat=grid.lat, lon=grid.lon, method="nearest", tolerance=0.01)
            for v, da in fields.items()}


def replace_sst(base, name):
    """Overwrite SST, Ft and Fs (and their biases) of one model already in base."""
    fields = forcing_fields(name, base)
    for v in ["sst", "Ft", "Fs"]:
        da = fields[v]
        dims = base[v].sel(model=name).dims
        base[v].loc[dict(model=name)] = da.transpose(*dims).values
        base[f"bias_{v}"].loc[dict(model=name)] = (
            base[v].sel(model=name) - base[OBS_NAME[v]]).transpose(*dims).values
    return base


def model_slice(name, base):
    fields = shdr_fields(name, base)
    fields.update(forcing_fields(name, base))

    out = {}
    for v, da in fields.items():
        out[v] = da.assign_coords(model=name)
        out[f"bias_{v}"] = (da - base[OBS_NAME[v]]).assign_coords(model=name)
    # variables not derived for the new models (Gmax, zGmax) are left empty
    template = base.isel(model=0, drop=True)
    for v in base.data_vars:
        if v not in out and "model" in base[v].dims:
            out[v] = xr.full_like(template[v], np.nan).assign_coords(model=name)
    return xr.Dataset(out)


if __name__ == "__main__":
    base = xr.open_dataset(BASE_FILE).load().drop_vars("number", errors="ignore")
    for m in REFIT_MODELS:
        if (INTERIM_DIR / "shdr_refit" / f"SHDR_{m}_climatology_RG.nc").exists():
            base = replace_shdr(base, m)
    for m in REFIT_SST:
        if (RAW_DIR / "models" / m / "forcing.nc").exists():
            base = replace_sst(base, m)
    new = xr.concat([model_slice(m, base) for m in NEW_MODELS], dim="model")
    per_model = [v for v in base.data_vars if "model" in base[v].dims]
    shared = [v for v in base.data_vars if "model" not in base[v].dims]
    merged = xr.merge([xr.concat([base[per_model], new], dim="model"), base[shared]])
    merged.to_netcdf(OUT_FILE)
