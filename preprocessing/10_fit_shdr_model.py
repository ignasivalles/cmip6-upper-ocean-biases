"""Monthly climatology (1970-2014), regridding to 1x1 degree and SHDR fit for one
model downloaded with 09_download_model.py.

Temperature and salinity are averaged into a monthly climatology first and SHDR
is fitted once per month and grid cell (used for the five models added to the
original ensemble).

Outputs, in RAW_DIR/models/<MODEL>/:
    climatology.nc   thetao, so, density (month, lev, lat, lon)
    forcing.nc       tos, sos, hfds, tauu, tauv, evspsbl, pr (month, lat, lon),
                     if the surface fields were downloaded
    shdr.nc          SHDR parameters (month, param, lat, lon)

Usage:
    python preprocessing/10_fit_shdr_model.py <MODEL>
"""

import sys
from multiprocessing import Pool

import gsw
import numpy as np
import xarray as xr
import xesmf as xe

import SHDR
from common.paths import RAW_DIR

MAX_DEPTH = 900.0
N_WORKERS = 10
TARGET = xr.Dataset({"lat": ("lat", np.arange(-89.5, 90, 1)), "lon": ("lon", np.arange(0.5, 360, 1))})
PARAMS = ["D1", "b2", "c2", "b3", "a2", "a1", "a3"]


def detect_grid(da):
    """(is_curvilinear, spatial dims in the order they appear, lat name, lon name)."""
    if "lat" in da.dims and "lon" in da.dims:
        return False, tuple(d for d in da.dims if d in ("lat", "lon")), "lat", "lon"
    for pair in [{"j", "i"}, {"y", "x"}, {"nj", "ni"}, {"nlat", "nlon"}]:
        if pair <= set(da.dims):
            lat = "latitude" if "latitude" in da.coords else "lat"
            lon = "longitude" if "longitude" in da.coords else "lon"
            return True, tuple(d for d in da.dims if d in pair), lat, lon
    raise ValueError(f"Unsupported grid, dims={da.dims}")


def open_var(model_dir, var):
    files = sorted((model_dir / var).glob("*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 12})
    # some models (e.g. CESM2-FV2) give depth in cm
    if "lev" in ds.coords:
        units = str(ds.lev.attrs.get("units", "")).lower()
        if units in ("cm", "centimeter", "centimeters"):
            ds = ds.assign_coords(lev=ds.lev / 100.0)
        elif units not in ("m", "meter", "meters", ""):
            raise ValueError(f"Unknown depth units {units!r}")
    # year-month slicing works with non-standard calendars
    return ds.sel(time=slice("1970-01", "2014-12"))[var]


def monthly_climatology(da):
    """Mean of each calendar month, looping over time to keep memory low."""
    sums, counts = {}, {}
    for t in range(da.sizes["time"]):
        m = int(da.time.dt.month.values[t])
        x = da.isel(time=t).values
        ok = np.isfinite(x)
        sums.setdefault(m, np.zeros(x.shape))[ok] += x[ok]
        counts.setdefault(m, np.zeros(x.shape, int))[ok] += 1
    return np.stack([np.where(counts[m] > 0, sums[m] / np.maximum(counts[m], 1), np.nan)
                     for m in range(1, 13)])


def climatology_on_1x1(model_dir, var, max_depth=None):
    da = open_var(model_dir, var)
    if max_depth is not None:
        da = da.sel(lev=slice(0, max_depth))
    curvilinear, (d0, d1), lat, lon = detect_grid(da)
    dims = ("month",) + (("lev",) if "lev" in da.dims else ()) + (d0, d1)
    coords = {"month": np.arange(1, 13), d0: da[d0].values, d1: da[d1].values}
    if "lev" in da.dims:
        coords["lev"] = da.lev.values
    clim = xr.DataArray(monthly_climatology(da), dims=dims, coords=coords)

    if curvilinear:
        src = xr.Dataset({"lat": ((d0, d1), da[lat].values), "lon": ((d0, d1), da[lon].values)})
    else:
        src = xr.Dataset({"lat": ("lat", da.lat.values), "lon": ("lon", da.lon.values)})
    regrid = xe.Regridder(src, TARGET, "bilinear", periodic=True, ignore_degenerate=True)
    return regrid(clim)


def potential_density(theta, salt):
    """Potential density (TEOS-10, sigma0 + 1000) from potential temperature and
    practical salinity."""
    lat2d, lon2d = np.meshgrid(theta.lat.values, theta.lon.values, indexing="ij")
    dens = np.full(theta.shape, np.nan)
    for m in range(theta.sizes["month"]):
        for k, z in enumerate(theta.lev.values):
            t, s = theta.isel(month=m, lev=k).values, salt.isel(month=m, lev=k).values
            ok = np.isfinite(t) & np.isfinite(s)
            if ok.any():
                sa = gsw.SA_from_SP(s[ok], gsw.p_from_z(-z, lat2d[ok]), lon2d[ok], lat2d[ok])
                dens[m, k][ok] = gsw.sigma0(sa, gsw.CT_from_pt(sa, t[ok])) + 1000.0
    return xr.DataArray(dens, dims=theta.dims, coords=theta.coords)


def fit_one(args):
    i, j, m, rho, z = args
    ok = np.isfinite(rho)
    if ok.sum() < 10:
        return None
    try:
        return i, j, m, SHDR.fit_profile(rho[ok], z[ok])[:7]
    except Exception:
        return None


def fit_shdr(dens):
    z = dens.lev.values
    cells = np.argwhere(np.isfinite(dens.isel(month=0, lev=0).values))
    tasks = [(i, j, m, dens.values[m, :, i, j], z) for i, j in cells for m in range(12)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(fit_one, tasks, chunksize=200)

    out = np.full((12, len(PARAMS), dens.sizes["lat"], dens.sizes["lon"]), np.nan)
    for r in filter(None, results):
        i, j, m, params = r
        out[m, :, i, j] = params
    return xr.DataArray(out, dims=("month", "param", "lat", "lon"), name="shdr",
                        coords={"month": np.arange(1, 13), "param": PARAMS,
                                "lat": dens.lat, "lon": dens.lon})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model_dir = RAW_DIR / "models" / sys.argv[1]

    theta = climatology_on_1x1(model_dir, "thetao", MAX_DEPTH)
    salt = climatology_on_1x1(model_dir, "so", MAX_DEPTH)
    dens = potential_density(theta, salt)
    xr.Dataset({"thetao": theta, "so": salt, "density": dens}).to_netcdf(model_dir / "climatology.nc")

    surface = ["tos", "sos", "hfds", "tauu", "tauv", "evspsbl", "pr"]
    if all((model_dir / v).exists() for v in surface):
        forcing = {v: climatology_on_1x1(model_dir, v) for v in surface}
        xr.Dataset(forcing).to_netcdf(model_dir / "forcing.nc")

    fit_shdr(dens).to_dataset().to_netcdf(model_dir / "shdr.nc")
