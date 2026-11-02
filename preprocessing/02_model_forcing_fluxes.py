"""Wind-stress curl, wind-stress magnitude and surface density fluxes (Ft, Fs)
from the regridded monthly CMIP6 surface fields (01_regrid_forcing.py).

Output: RAW_DIR/CMIP6_forcings_RG_360/{curl,wsm,SSDflux}_output/<model>_atm_forcing_*.nc
"""

import os

from common.paths import RAW_DIR
from flux_tools import calculate_model_curl, calculate_model_ssdflux, calculate_model_wsm

FORCING_DIR = RAW_DIR / "CMIP6_forcings_RG_360"

if __name__ == "__main__":
    files = sorted(str(f) for f in FORCING_DIR.glob("*_atm_forcing.nc"))
    for name, func in [("curl", calculate_model_curl), ("wsm", calculate_model_wsm),
                       ("SSDflux", calculate_model_ssdflux)]:
        out = FORCING_DIR / f"{name}_output"
        os.makedirs(out, exist_ok=True)
        func(files, str(out))
