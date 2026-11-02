"""
Fix the missing SST and heat-flux data for FGOALS-f3-L in the allbias file.

The original regridding failed for FGOALS-f3-L because its tos/hfds are on a
curvilinear (j×i) grid with 2-D lat/lon.  This script:
  1. Reads the raw native-grid tos and hfds files.
  2. Regrids them to the standard 1°×1° grid with xesmf (bilinear).
  3. Computes the 1970-2014 monthly climatology.
  4. Writes sst, hflux, bias_sst and bias_hflux back into the allbias NetCDF
     for the FGOALS-f3-L model slice.

Usage:
    python preprocessing/08_fix_fgoals_forcing.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

from common.paths import INTERIM_DIR, RAW_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────

TOS_FILE  = RAW_DIR / "CMIP6_forcings/FGOALS-f3-L/tos/tos_Omon_FGOALS-f3-L_historical_r1i1p1f1_gn_185001-201412.nc"
HFDS_FILE = RAW_DIR / "CMIP6_forcings/FGOALS-f3-L/hfds/hfds_Omon_FGOALS-f3-L_historical_r1i1p1f1_gn_185001-201412.nc"
OBS_SST   = RAW_DIR / "ERA5_data/ERA5_sst_rg/ERA5_sst_climatology_1970_2014.nc"
OBS_HFLUX = RAW_DIR / "ORAS5/ORAS5_hflux_rg/ORAS5_hflux_climatology_1970_2014.nc"
ALLBIAS   = INTERIM_DIR / "allbias_29CMIP6_EN4.nc"

MODEL      = "FGOALS-f3-L"
CLIM_START = 1970
CLIM_END   = 2014


# ── Standard target grid ──────────────────────────────────────────────────────
def target_grid():
    return xr.Dataset({
        "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
        "lon": (["lon"], np.arange(0.5, 360.5, 1.0)),
    })


# ── Fix broken time axis (FGOALS-f3-L has corrupted cftime coords) ────────────
def fix_time(ds, var, start_year=1850, start_month=1):
    """Reconstruct a clean monthly time axis from position (data order is correct)."""
    import cftime
    n = ds.sizes["time"]
    new_times = [
        cftime.DatetimeGregorian(start_year + (start_month - 1 + i) // 12,
                                 (start_month - 1 + i) % 12 + 1, 15)
        for i in range(n)
    ]
    ds = ds.assign_coords(time=("time", new_times))
    return ds


# ── Monthly climatology on native grid then regrid 12 maps ───────────────────
def clim_then_regrid(ds_native, var, ds_tgt):
    """
    Compute the 1970-2014 monthly climatology on the native (j,i) grid first
    (much cheaper), then regrid the 12 maps to the regular 1°×1° grid.
    """
    ds_fixed = fix_time(ds_native[[var]], var)
    da = ds_fixed[var]
    time_vals = da.time.values
    years  = np.array([int(t.year)  for t in time_vals])
    months = np.array([int(t.month) for t in time_vals])
    mask   = (years >= CLIM_START) & (years <= CLIM_END)
    print(f"    → {mask.sum()} time steps selected for climatology")

    # Load only the selected slice into memory
    da_sel  = da.isel(time=np.where(mask)[0]).load()
    mon_sel = months[mask]

    # Monthly mean on native grid → (12, j, i)
    clim_native = np.stack(
        [da_sel.values[mon_sel == m].mean(axis=0) for m in range(1, 13)], axis=0
    )
    lat2d = ds_native.latitude.values
    lon2d = ds_native.longitude.values
    ds_clim_native = xr.Dataset({
        var: xr.DataArray(
            clim_native,
            dims=["time", "j", "i"],
            coords={"time": np.arange(1, 13),
                    "lat": (["j", "i"], lat2d),
                    "lon": (["j", "i"], lon2d)},
        )
    })

    # Regrid 12 maps to 1°×1°
    regridder = xe.Regridder(ds_clim_native, ds_tgt, "bilinear", periodic=True,
                             ignore_degenerate=True)
    out = regridder(ds_clim_native[var], keep_attrs=True)   # (time=12, lat, lon)
    return xr.DataArray(
        out.values,
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(1, 13),
                "lat": ds_tgt.lat.values,
                "lon": ds_tgt.lon.values},
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ds_tgt = target_grid()

    # ── tos ───────────────────────────────────────────────────────────────────
    print("Reading tos (native grid climatology then regrid)...")
    tos_raw  = xr.open_dataset(TOS_FILE)
    tos_clim = clim_then_regrid(tos_raw, "tos", ds_tgt)   # (12, 180, 360) in K

    # ── hfds ──────────────────────────────────────────────────────────────────
    print("Reading hfds (native grid climatology then regrid)...")
    hfds_raw  = xr.open_dataset(HFDS_FILE)
    hfds_clim = clim_then_regrid(hfds_raw, "hfds", ds_tgt)

    # ── Quick sanity check ────────────────────────────────────────────────────
    for name, arr in [("tos_clim", tos_clim), ("hfds_clim", hfds_clim)]:
        n = int(np.isfinite(arr.values).sum())
        print(f"  {name}: {n}/{arr.values.size} finite")

    # ── Observations ──────────────────────────────────────────────────────────
    obs_sst_ds   = xr.open_dataset(OBS_SST)
    obs_hflux_ds = xr.open_dataset(OBS_HFLUX)

    obs_sst_c   = obs_sst_ds["sst"] - 273.15    # K → °C, (month/time, lat, lon)
    obs_hflux_v = obs_hflux_ds["sohefldo"]

    # Normalise time axis to 1–12
    for da, name in [(obs_sst_c, "sst"), (obs_hflux_v, "hflux")]:
        tdim = "month" if "month" in da.dims else "time"
        if da.sizes[tdim] == 12:
            da = da.assign_coords({tdim: np.arange(1, 13)}).rename({tdim: "time"})
        if name == "sst":
            obs_sst_c = da
        else:
            obs_hflux_v = da

    # ── Biases ────────────────────────────────────────────────────────────────
    tos_c      = tos_clim - 273.15           # K → °C
    bias_sst   = (tos_c   - obs_sst_c).values
    bias_hflux = (hfds_clim - obs_hflux_v).values

    # ── Patch the allbias file (write to temp, then replace) ─────────────────
    print(f"Patching {ALLBIAS} ...")
    allbias = xr.open_dataset(ALLBIAS).load()
    allbias.close()

    model_idx = list(allbias.model.values).index(MODEL)

    def _patch(var_name, new_arr_2d):
        """Replace FGOALS-f3-L slice (shape time×lat×lon) in var_name."""
        dims = allbias[var_name].dims
        m_ax = list(dims).index("model")
        sl = [slice(None)] * allbias[var_name].values.ndim
        sl[m_ax] = model_idx
        # new_arr_2d is (time, lat, lon); reorder if model is not last
        if m_ax == allbias[var_name].values.ndim - 1:
            allbias[var_name].values[tuple(sl)] = new_arr_2d
        else:
            allbias[var_name].values[tuple(sl)] = new_arr_2d

    _patch("sst",        tos_c.values)
    _patch("hflux",      hfds_clim.values)
    _patch("bias_sst",   bias_sst)
    _patch("bias_hflux", bias_hflux)

    # Verify
    dims = allbias["bias_sst"].dims
    m_ax = list(dims).index("model")
    sl = [slice(None)] * allbias["bias_sst"].values.ndim
    sl[m_ax] = model_idx
    n = int(np.isfinite(allbias["bias_sst"].values[tuple(sl)]).sum())
    print(f"  bias_sst finite after patch: {n} / {allbias['bias_sst'].values[tuple(sl)].size}")

    # Write to temp file then replace original
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False,
                                     dir=ALLBIAS.parent) as tmp:
        tmp_path = Path(tmp.name)
    allbias.to_netcdf(tmp_path)
    shutil.move(str(tmp_path), str(ALLBIAS))
    print(f"Saved: {ALLBIAS}")


if __name__ == "__main__":
    main()
