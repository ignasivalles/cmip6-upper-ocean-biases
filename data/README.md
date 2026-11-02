Processed data used by the figure scripts. Download the files from Zenodo
(doi: TBD) into this folder, or point `CMIP6_BIAS_DATA` to where they are.

| File | Content |
|---|---|
| `cmip6_upper_ocean_n34.nc` | SHDR parameters (a1, a2, D1, b3), surface fields (SST, SSS, zonal wind stress, net heat flux, E-P) and surface density fluxes (Ft, Fs, wind-stress curl) for 34 CMIP6 models and the observational references (EN4, ERA5, ORAS5). Monthly climatology, 1970-2014, 1x1 degree. |
| `obs_shdr_iap_cora.nc` | SHDR parameters for the IAPv4 and CORA references. |
| `ncep_r1_fluxes.nc` | NCEP/NCAR Reanalysis-1 net heat flux, E-P and wind-stress curl. |
| `b3_raw_gradient.nc` | Main-pycnocline density gradient computed directly from the model profiles (Figure S4). |
| `RECCAP2_region_masks_all_v20221025.nc` | RECCAP2 ocean basin masks. |

Biases are model minus observation and are computed when the data are loaded
(`common/data.py`). The stored curl fields are too small by a factor 180/pi
(grid spacing in degrees); this is corrected on loading.
