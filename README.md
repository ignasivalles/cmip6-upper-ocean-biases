# Upper-ocean structure biases in CMIP6

Code for

> Vallès-Casanova, I., Somavilla, R., Naveira Garabato, A., González-Pola, C., & Fernández-Diaz, J. M.
> Upper-ocean structure biases in CMIP6: local surface forcing vs. remote Southern Ocean control.
> *Geophysical Research Letters* (submitted).

The upper-ocean density profiles of 34 CMIP6 historical simulations and of
EN4, IAPv4 and CORA are described with the SHDR fit (González-Pola et al., 2007).
The model biases in surface density, mixed-layer depth and pycnocline structure are
then related to surface forcing biases (dominance analysis) and, for the
subtropical main pycnocline, to Southern Ocean biases.

## Reproducing the figures

```bash
conda env create -f environment.yml
conda activate cmip6-biases
pip install -e .
```

Download the processed data from Zenodo (doi: TBD) into `data/` (see
`data/README.md`), then run any script in `figures/`, e.g.

```bash
python figures/fig2_local_forcing.py
for f in figures/fig*.py; do python "$f"; done
```

Figures and the tables behind them are written to `output/`. Fig. 2 takes a
couple of minutes (1000 bootstrap samples per region); the rest run in seconds
to a minute.

| Script | Figure |
|---|---|
| `fig1_bias_maps.py` | 1 |
| `fig2_local_forcing.py` | 2 |
| `fig3_b3_hotspot.py` | 3 |
| `fig4_southern_ocean.py` | 4 |
| `figS01_S02_iap_cora.py` | S1, S2 |
| `figS04_b3_validation.py` | S4 |
| `figS05_local_collinearity.py` | S5 |
| `figS06_hotspot_threshold.py` | S6 |
| `figS07_remote_collinearity.py` | S7 |
| `figS08_obs_products.py` | S8 |
| `figS09_seasonal_cycle.py` | S9 |
| `figS10_forcing_biases.py` | S10 |
| `figS11_ncep.py` | S11 |
| `figS12_flux_regression.py` | S12 |
| `figS13_pc2.py` | S13 |
| `figS14_model_sensitivity.py` | S14 |
| `figS15_amoc_proxy.py` | S15 |

## Preprocessing

`preprocessing/` contains the steps that produce the processed data from the
CMIP6 output and the observational products. They need the raw data (several
TB, not included); set `CMIP6_RAW_DATA` to its location.

| Step | |
|---|---|
| `00_shdr_climatology.py` | SHDR fit of each model's 1970-2014 monthly temperature/salinity climatology (CDO `ymonmean`), regridded to 1x1 degree |
| `01_regrid_forcing.py` | regrid the monthly model surface fields to 1x1 degree |
| `02_model_forcing_fluxes.py` | model wind-stress curl and surface density fluxes |
| `03_prepare_observations.py` | regrid observations; wind-stress curl and surface density fluxes |
| `04_obs_climatologies.py` | 1970-2014 climatologies of ERA5 and ORAS5 fields |
| `05_ensemble_climatologies.py` | 1970-2014 climatologies of all models and of the SHDR references |
| `06_build_bias_dataset.py` | combine models and observations (original 29 models) |
| `07_add_ep_bias.py` | add E-P |
| `08_fix_fgoals_forcing.py` | regrid FGOALS-f3-L SST and heat flux (curvilinear grid) |
| `09_download_model.py` | download model fields from ESGF |
| `10_fit_shdr_model.py` | climatology, density and SHDR fit for an additional model |
| `11_merge_models.py` | add the five additional models (34 in total) |
| `12_ncep_climatology.py` | NCEP/NCAR Reanalysis-1 climatology |
| `13_raw_density_gradient.py` | density gradient from the raw model profiles (Fig. S4) |
| `14_export_dataset.py` | write the files archived on Zenodo |

`SHDR.py` is the SHDR implementation of https://github.com/manufrn/SHDR (MIT license),
with the fit depth range used here.

## Data sources

CMIP6 output: ESGF (https://esgf-metagrid.cloud.dkrz.de). EN4: Met Office Hadley
Centre. IAPv4: Institute of Atmospheric Physics. CORA: Copernicus Marine Service.
ERA5 and ORAS5: Copernicus Climate Change Service. NCEP/NCAR Reanalysis-1: NOAA
PSL. Basin masks: RECCAP2.

## License

MIT, see `LICENSE`.
