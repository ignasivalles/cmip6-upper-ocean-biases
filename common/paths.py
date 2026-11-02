import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Processed data (Zenodo archive). Override with CMIP6_BIAS_DATA=/path/to/data
DATA_DIR = Path(os.environ.get("CMIP6_BIAS_DATA", ROOT / "data"))

# Raw CMIP6 / reanalysis files, only needed for preprocessing
RAW_DIR = Path(os.environ.get("CMIP6_RAW_DATA", ROOT / "raw"))

FIG_DIR = Path(os.environ.get("CMIP6_FIG_DIR", ROOT / "output"))

ENSEMBLE_FILE = DATA_DIR / "cmip6_upper_ocean_n34.nc"
OBS_PRODUCTS_FILE = DATA_DIR / "obs_shdr_iap_cora.nc"
NCEP_FILE = DATA_DIR / "ncep_r1_fluxes.nc"
REGION_FILE = DATA_DIR / "RECCAP2_region_masks_all_v20221025.nc"

# Intermediate products of the preprocessing chain
INTERIM_DIR = RAW_DIR / "interim"
