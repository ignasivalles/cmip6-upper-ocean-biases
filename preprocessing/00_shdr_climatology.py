"""SHDR fit of a model's monthly temperature/salinity climatology.

Input: monthly climatologies of thetao and so on the native model grid, made
with CDO from the CMIP6 historical files, e.g.

    cdo -L ymonmean -selyear,1970/2014 -mergetime thetao_Omon_<model>_historical_*.nc \
        <model>_thetao_climatology.nc

Potential density (sigma0, TEOS-10) is computed down to 900 m and SHDR is
fitted to every profile and calendar month. The parameters are then
regridded bilinearly to the 1x1 degree grid.

Output: <OUT_DIR>/SHDR_<model>_climatology_RG.nc  (time, param, lat, lon)

Usage:
    python preprocessing/00_shdr_climatology.py <MODEL> <CLIM_DIR> <OUT_DIR> [N_PROCESSES]
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
import xesmf as xe

import SHDR

MAX_DEPTH = 900.0
PARAMS = ["D1", "b2", "c2", "b3", "a2", "a1", "a3", "em"]


def load_climatology(clim_dir, model):
    theta = xr.open_dataset(clim_dir / f"{model}_thetao_climatology.nc")["thetao"]
    salt = xr.open_dataset(clim_dir / f"{model}_so_climatology.nc")["so"]
    depth = next(d for d in theta.dims if d not in ("time",) and theta[d].ndim == 1
                 and d not in ("i", "j", "x", "y", "lat", "lon", "nlat", "nlon"))
    theta, salt = (x.rename({depth: "lev"}).sel(lev=slice(0, MAX_DEPTH)) for x in (theta, salt))
    return theta, salt


def sigma0(theta, salt):
    z = theta.lev.values[None, :, None, None] * np.ones(theta.shape)
    sa = gsw.SA_from_SP(salt.values, gsw.p_from_z(-z, 0), 0, 0)
    return gsw.sigma0(sa, gsw.CT_from_pt(sa, theta.values))


def fit_one(args):
    k, rho, z = args
    ok = np.isfinite(rho)
    if ok.sum() < 10:
        return k, None
    try:
        return k, np.asarray(SHDR.fit_profile(rho[ok], z[ok]), dtype=float)
    except Exception:
        return k, None


def fit_all(rho, z, n_proc):
    """rho: (time, lev, ny, nx). Returns (time, ny, nx, param)."""
    nt, _, ny, nx = rho.shape
    out = np.full((nt, ny, nx, len(PARAMS)), np.nan)
    ocean = np.argwhere(np.isfinite(rho[0, 0]))
    tasks = ((t * ny * nx + j * nx + i, rho[t, :, j, i], z) for t in range(nt) for j, i in ocean)
    with Pool(n_proc) as pool:
        for k, p in pool.imap_unordered(fit_one, tasks, chunksize=200):
            if p is not None:
                t, rest = divmod(k, ny * nx)
                out[t, rest // nx, rest % nx, :len(p)] = p[:len(PARAMS)]
    return out


def regrid(shdr, theta):
    lat = next(c for c in ("latitude", "lat", "nav_lat") if c in theta.coords)
    lon = next(c for c in ("longitude", "lon", "nav_lon") if c in theta.coords)
    ydim, xdim = theta.dims[-2:]
    src = xr.Dataset(coords={"lat": theta[lat], "lon": theta[lon]})
    da = xr.DataArray(shdr, dims=("time", ydim, xdim, "param"),
                      coords={"time": np.arange(1, 13), "param": PARAMS}).assign_coords(
        lat=theta[lat], lon=theta[lon])
    target = xr.Dataset({"lat": ("lat", np.arange(-89.5, 90, 1)),
                         "lon": ("lon", np.arange(0.5, 360, 1))})
    rg = xe.Regridder(src, target, "bilinear", periodic=True, ignore_degenerate=True)
    return rg(da.transpose("time", "param", ydim, xdim)).rename("shdr")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    model, clim_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    n_proc = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    theta, salt = load_climatology(clim_dir, model)
    rho = sigma0(theta.load(), salt.load())
    shdr = fit_all(rho, theta.lev.values, n_proc)
    out = regrid(shdr, theta).to_dataset()
    out.attrs = {"model": model, "climatology": "1970-2014", "max_depth_m": MAX_DEPTH}
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_dir / f"SHDR_{model}_climatology_RG.nc")
