"""Main-pycnocline density gradient computed directly from the model temperature
and salinity climatologies (no SHDR fit), for the validation of b3 (Figure S4).

For each model and calendar month, the gradient below the seasonal pycnocline
(common.gradient.background_gradient) is computed at every cell of the two
validation boxes and averaged over months.

Input:  monthly temperature and salinity climatologies on the 1x1 degree grid,
        RAW_DIR/clim/<model>_{thetao,so}_RG.nc (original models) or
        RAW_DIR/models/<model>/climatology.nc (10_fit_shdr_model.py)
Output: INTERIM_DIR/b3_raw_gradient.nc
"""

import numpy as np
import xarray as xr

from common.gradient import BOXES, Z_BASE, background_gradient, box_mask, sigma0_profile
from common.paths import ENSEMBLE_FILE, INTERIM_DIR, RAW_DIR

CLIM_DIR = RAW_DIR / "clim"
EXAMPLE = dict(model="CESM2", lat=-40.0, lon=285.0, month=11)


def find(model, var):
    files = sorted(CLIM_DIR.glob(f"{model}_{var}_RG.nc")) or sorted(CLIM_DIR.glob(f"*_{model}_{var}_RG.nc"))
    return files[0] if files else None


def load_ts(model):
    added = RAW_DIR / "models" / model / "climatology.nc"
    if added.exists():
        ds = xr.open_dataset(added)
        theta, salt = ds["thetao"], ds["so"]
    else:
        t, s = find(model, "thetao"), find(model, "so")
        if t is None or s is None:
            return None
        theta, salt = xr.open_dataset(t)["thetao"], xr.open_dataset(s)["so"]
    depth_name = next(d for d in theta.dims if d not in ("time", "month", "lat", "lon"))
    theta, salt = (x.rename({depth_name: "lev"}) for x in (theta, salt))
    theta, salt = (x.rename({"time": "month"}) if "time" in x.dims else x for x in (theta, salt))
    use = theta.lev.values <= Z_BASE
    return theta.isel(lev=use).sortby("lev"), salt.isel(lev=use).sortby("lev")


def model_gradient(theta, salt, mask):
    theta = theta.reindex(lat=mask.lat, lon=mask.lon, method="nearest").where(mask)
    salt = salt.reindex(lat=mask.lat, lon=mask.lon, method="nearest").where(mask)
    depth = theta.lev.values
    out = np.full((theta.sizes["month"],) + mask.shape, np.nan)
    for m in range(theta.sizes["month"]):
        t = theta.isel(month=m).load().values
        s = salt.isel(month=m).load().values
        for i, j in zip(*np.nonzero(mask.values)):
            if np.isnan(t[:, i, j]).any() or np.isnan(s[:, i, j]).any():
                continue
            sig = sigma0_profile(t[:, i, j], s[:, i, j], depth, mask.lat.values[i], mask.lon.values[j])
            out[m, i, j] = background_gradient(sig, depth)[0]
    return np.nanmean(out, axis=0)


if __name__ == "__main__":
    models = xr.open_dataset(ENSEMBLE_FILE).model.values
    grid = xr.open_dataset(ENSEMBLE_FILE)[["b3"]].sel(lat=slice(-41, -14))
    mask = box_mask(grid, BOXES[0]) | box_mask(grid, BOXES[1])

    fields, used = [], []
    for model in models:
        ts = load_ts(model)
        if ts is None:
            print(f"{model}: no raw climatology, skipped")
            continue
        fields.append(model_gradient(*ts, mask))
        used.append(model)
        print(f"{model}: done")

    ds = xr.Dataset({"raw_gradient": (("model", "lat", "lon"), np.stack(fields))},
                    coords={"model": used, "lat": mask.lat, "lon": mask.lon})
    ds["raw_gradient"].attrs = {"units": "kg m-4",
                                "long_name": "d(sigma0)/dz below the seasonal pycnocline, annual mean"}

    # one example profile for panel (a)
    theta, salt = load_ts(EXAMPLE["model"])
    sel = dict(month=EXAMPLE["month"], lat=EXAMPLE["lat"], lon=EXAMPLE["lon"])
    t = theta.sel(sel, method="nearest").values
    s = salt.sel(sel, method="nearest").values
    ok = np.isfinite(t) & np.isfinite(s)
    ds["example_sigma0"] = ("depth", sigma0_profile(t[ok], s[ok], theta.lev.values[ok],
                                                     EXAMPLE["lat"], EXAMPLE["lon"]))
    ds = ds.assign_coords(depth=theta.lev.values[ok])
    ds.attrs = {f"example_{k}": v for k, v in EXAMPLE.items()}
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(INTERIM_DIR / "b3_raw_gradient.nc")
