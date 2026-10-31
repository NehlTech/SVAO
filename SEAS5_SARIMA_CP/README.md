# SEAS5_SARIMA_CP

Verification and statistical post-processing of ECMWF SEAS5 seasonal temperature forecasts at six Ghana Meteorological Agency (GMet) stations.

Working title: Verifying and post-processing seasonal temperature forecasts for Ghana based on ECMWF SEAS5.

## Status
Setup only. No results yet. Numbers will appear here after the notebooks are run.

## Layout
- `notebooks/`  numbered Colab notebooks (00 to 07), run in order
- `src/`        shared helper code
- `data/raw/`   not tracked (GMet workbook, SEAS5 and ERA5-Land downloads); kept on Google Drive
- `data/processed/`  cleaned monthly tables
- `results/`    JSON and CSV outputs read by the paper
- `figures/`    figures used in the paper
- `config.json` stations, years, seeds

## Data
- GMet daily Tmax and Tmin, six stations, 1995 to 2025. Sharing permission is being confirmed, so the raw workbook is not published here.
- ECMWF SEAS5 monthly 2 m temperature, Copernicus Climate Data Store (`seasonal-monthly-single-levels`), systems 5 and 51.
- ERA5-Land hourly 2 m temperature at station points (`reanalysis-era5-land-timeseries`), used as an independent check.
- NOAA GSOD, used only as a small external check.

The earlier ARIMA-LSTM and SVAO notebooks (01 to 14) are in the top level of this repository and are unchanged.

## Notebook index and run order

| Notebook | Purpose | Main outputs |
|---|---|---|
| 00 | GitHub and project setup | folder structure, config |
| 01 | Data quality, homogeneity, SEAS5 file checks | gmet_monthly_clean.csv, nb01 results |
| 02 | Reference forecasts (climatology, trend climatology, seasonal ARIMA) | reference_forecasts.csv |
| 03 | Verification of SEAS5 | seas5_verification_table.csv |
| 04 | Post-processing (linear, ridge, boosting, network, season weighting) | postprocessed_forecasts.csv |
| 05 | Hybrid ARIMA-LSTM | hybrid_forecasts.csv |
| 06 | Conformal prediction intervals | conformal_intervals.csv |
| 07 | Data sensitivity (adjusted series, Tmax and Tmin, one station out) | sensitivity_forecasts.csv |
| 08 | SEAS5 settings, trends, hybrid training size, station tests | nb08 results |
| 09 | Final numbers, decision-rule register, repository check | results/all_paper_numbers.json |

Run the notebooks in order. Raw data (GMet workbook, ERA5-Land and SEAS5 extracts) are kept in Google Drive and are not committed. Every number quoted in the manuscript is read from `results/all_paper_numbers.json`.
