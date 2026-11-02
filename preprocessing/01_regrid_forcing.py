"""Regrid the monthly CMIP6 surface fields of each model (tos, sos, hfds, tauu,
tauv, evspsbl, pr; 1970-2014) to the 1x1 degree grid.

Input:  RAW_DIR/CMIP6_forcings/<model>/<variable>/*.nc
Output: RAW_DIR/CMIP6_forcings_RG_360/<model>_atm_forcing.nc
"""

from cmip6_tools import process_models_forcing
from common.paths import RAW_DIR

if __name__ == "__main__":
    process_models_forcing(str(RAW_DIR / "CMIP6_forcings"))
