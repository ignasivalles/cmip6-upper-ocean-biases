"""Main-pycnocline density gradient diagnosed directly from a density profile,
independently of the SHDR fit (used to validate b3)."""

import gsw
import numpy as np

Z_BASE = 900.0   # deepest level used, as in the SHDR fits [m]
PEAK_FRAC = 0.05

# open-ocean subtropical boxes used for the b3 checks (Figs. S4, S8)
BOXES = [dict(name="Indian", lat=(-30, -15), lon=(60, 100), extent=[20, 140, -45, 0]),
         dict(name="Pacific", lat=(-40, -20), lon=(205, 265), extent=[160, 300, -55, 0])]


def box_mask(da, box):
    return ((da.lat >= box["lat"][0]) & (da.lat <= box["lat"][1])
            & (da.lon >= box["lon"][0]) & (da.lon <= box["lon"][1]))


def sigma0_profile(theta, salt, depth, lat, lon):
    """Potential density anomaly from potential temperature and practical salinity."""
    p = gsw.p_from_z(-depth, lat)
    sa = gsw.SA_from_SP(salt, p, lon, lat)
    return gsw.sigma0(sa, gsw.CT_from_pt(sa, theta))


def background_gradient(sigma0, depth, peak_frac=PEAK_FRAC):
    """Linear d(sigma0)/dz below the seasonal pycnocline.

    The base of the seasonal pycnocline is the first level below the gradient
    maximum where |d sigma0/dz| drops under peak_frac of that maximum; the
    gradient is the least-squares slope from there to the deepest level.
    Returns slope, intercept, depth of the gradient maximum and of the base.
    """
    grad = np.diff(sigma0) / np.diff(depth)
    zmid = (depth[:-1] + depth[1:]) / 2
    peak = int(np.argmax(np.abs(grad)))
    if np.abs(grad[peak]) <= 0:
        return np.nan, np.nan, np.nan, np.nan
    below = np.nonzero(np.abs(grad[peak:]) < peak_frac * np.abs(grad[peak]))[0]
    base = peak + below[0] if below.size else len(grad) - 1

    z, s = depth[base + 1:], sigma0[base + 1:]
    if z.size < 2:
        return np.nan, np.nan, zmid[peak], zmid[base]
    slope, intercept = np.polyfit(z, s, 1)
    return slope, intercept, zmid[peak], zmid[base]
