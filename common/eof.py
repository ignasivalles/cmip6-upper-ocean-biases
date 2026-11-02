"""b3 hotspot, inter-model EOF and pointwise (partial) correlations."""

import numpy as np
import xarray as xr
from scipy import stats

# 75th percentile of the ensemble-mean b3 bias over the Indo-Pacific, 10-40S [kg m-4]
B3_HOTSPOT_THRESHOLD = 0.000720


def ensemble_mean_b3(ds):
    b3 = ds["bias_b3"].mean(("model", "month"))
    return b3.where((b3 > -1e-3) & (b3 < 1e-2))


def extract_hotspot(b3_mean, mask, lat_band=slice(-40, -10),
                    threshold_low=B3_HOTSPOT_THRESHOLD, threshold_high=0.01):
    """Cells of the ensemble-mean b3 bias above threshold_low (NaN elsewhere)."""
    sub = b3_mean.where(mask).sel(lat=lat_band)
    return sub.where((sub > threshold_low) & (sub < threshold_high))


def hotspot_masks(ds, masks, threshold_low=B3_HOTSPOT_THRESHOLD):
    """Boolean Pacific and Indian hotspot masks."""
    b3_mean = ensemble_mean_b3(ds)
    pac = extract_hotspot(b3_mean, masks["pacific"], threshold_low=threshold_low).notnull()
    ind = extract_hotspot(b3_mean, masks["indian"], threshold_low=threshold_low).notnull()
    return pac, ind


def inter_model_eof(field, mask, n=2):
    """EOFs of a per-model field (model, lat, lon) within a mask.

    Each grid cell is centred and standardised across models. PCs are
    normalised to unit variance; PC1 is signed so that positive values
    mean a larger mean field within the mask (the sign of higher modes is
    arbitrary; they are returned as -U s).
    Returns pcs (list of DataArray), eofs (list), variance fractions.
    """
    X = field.where(mask).stack(space=("lat", "lon")).fillna(0)
    X = X - X.mean("model")
    X = (X / X.std("model")).fillna(0)

    U, s, Vt = np.linalg.svd(X.values, full_matrices=False)
    var_frac = s**2 / np.sum(s**2)

    field_mean = field.where(mask).mean(("lat", "lon"))
    pcs, eofs = [], []
    for i in range(n):
        pc = xr.DataArray(U[:, i] * s[i], coords=[X["model"]], name=f"PC{i + 1}")
        if i == 0:
            sign = np.sign(np.corrcoef(field_mean.values, pc.values)[0, 1]) or 1.0
        else:
            sign = -1.0
        pcs.append(sign * (pc - pc.mean()) / pc.std())
        eofs.append(xr.DataArray(sign * Vt[i], coords=[X["space"]]).unstack("space"))
    return pcs, eofs, var_frac[:n]


def corr_with_field(field, index):
    """Pointwise Pearson r and p between field (model, lat, lon) and index (model)."""
    field, index = xr.align(field, index, join="inner")

    def _rp(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 4:
            return np.nan, np.nan
        return stats.pearsonr(a[ok], b[ok])

    return xr.apply_ufunc(_rp, field, index, input_core_dims=[["model"], ["model"]],
                          vectorize=True, output_core_dims=[[], []])


def partial_corr_with_field(field, index, control):
    """Pointwise partial r(field, index | control) and two-sided p (df = n - 3)."""
    field, index = xr.align(field, index, join="inner")
    control = control.sel(model=field.model)

    ok = np.isfinite(index.values) & np.isfinite(control.values)
    r_yz = stats.pearsonr(index.values[ok], control.values[ok])[0]
    r_xz, _ = corr_with_field(field, control)
    r_xy, _ = corr_with_field(field, index)

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    r = (r_xy - r_xz * r_yz) / denom.where(denom > 0)

    dof = field.model.size - 3
    t = r * np.sqrt(dof / (1 - r**2))
    p = xr.apply_ufunc(lambda tv: 2 * stats.t.sf(np.abs(tv), dof), t, vectorize=True)
    return r, p


def regress_on_index(field, index):
    """Pointwise least-squares slope of field (model, lat, lon) on index (model), and p."""
    field, index = xr.align(field, index, join="inner")

    def _fit(y, x):
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 4:
            return np.nan, np.nan
        res = stats.linregress(x[ok], y[ok])
        return res.slope, res.pvalue

    return xr.apply_ufunc(_fit, field, index, input_core_dims=[["model"], ["model"]],
                          vectorize=True, output_core_dims=[[], []])
