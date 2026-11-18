# Analysis scripts behind the paper

These scripts produce the results reported in the paper on SEAS5 temperature forecasts for Ghana. The forecasts and result tables they write are already in `data/processed` and `results`. The scripts are kept here so that every number can be traced to the code that made it.

## hybrid_and_combination

Training and analysis for the hybrid ARIMA-LSTM models, the sea-temperature inputs, the forecast combinations and the learning curve over the number of stations.

| Script | What it does | Result files |
| --- | --- | --- |
| `train16.py` | LSTM alone and the hybrid built on trend climatology, trained with a rolling origin | `lstm_alone_and_trend_hybrid_forecasts.csv` |
| `analyse16.py` | Compares the hybrid with its parts, chooses the hybrid base for each lead on the development start months (2005 to 2012), confirms on 2013 to 2025, and tests forecast combinations | `nb16b_*` |
| `train17.py`, `analyse17.py` | The same models with Tropical Southern Atlantic (TSA) and Nino 3.4 sea-temperature inputs from `ext/tsa.csv` and `ext/nino34.csv`, and the test of whether they help | `nb17_*` |
| `train18.py`, `analyse18.py` | Hybrid built on the ridge forecast, and the combination of the strongest models | `nb18_*` |
| `lc19.py`, `analyse19.py` | Learning curve over 25 groups of two to six stations | `nb19_*` |

## bcsd_and_shuffle

| Script | What it does | Output |
| --- | --- | --- |
| `run21.py` | Comparison of the station-based ridge with ridge on the SEAS5-BCSD anomaly (start months 2005 to 2024) | `out/tests21.csv`, `rmse21.csv`, `scored21.csv`, `by_station21.csv` |
| `run22.py` | Ridge scored with a rolling origin and with a shuffled five-fold split | `out/leak22.csv` |
| `run22b.py` | The same shuffling test for gradient boosting | `out/leak22_gbm.csv` |

`bcsd.csv.gz` holds the SEAS5-BCSD values read at the six stations (Weber et al., 2026).

## Running

The scripts read the shared forecast tables from a folder called `proc`, which is `data/processed` in this repository, and write tables to `res` (`results`) or `out`. Run a script from a working folder in which `proc` and `res` point to those folders, for example with symbolic links. Fixed seeds are set inside each script. The station records come from the Ghana Meteorological Agency and are shared only with the agency's permission.
