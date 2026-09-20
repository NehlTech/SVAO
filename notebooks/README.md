# Notebooks

Run in order. All notebooks are CPU-only; none requires a GPU.

| # | Notebook | What it does |
|---|---|---|
| 00 | `00_Sync_Notebooks_To_Repo.ipynb` |  |
| 01 | `01_Setup_and_Data.ipynb` | Ingestion, quality control, monthly aggregation, chronological split |
| 01 | `01_corpus_construction.ipynb` |  |
| 02 | `02_Exploratory_Data_Analysis.ipynb` | Heteroscedasticity, stationarity, trend and missingness testing |
| 02 | `02_attack_generation.ipynb` |  |
| 02 | `02_attack_generation (1).ipynb` |  |
| 03 | `03_Module1_LinearTrendEstimator.ipynb` | ARIMA order decision, burn-in sensitivity, residual extraction |
| 03 | `03_leakage_safe_split.ipynb` |  |
| 03 | `03_leakage_safe_split (1).ipynb` |  |
| 04 | `04_Module2_SeasonalVarianceProfiler.ipynb` | Five variance parameterisations, selection by validation likelihood |
| 04 | `04_module1_encoding.ipynb` |  |
| 04 | `04_module1_encoding (1).ipynb` |  |
| 04 | `04_module1_encoding (2).ipynb` |  |
| 04 | `04_module1_encoding (3).ipynb` |  |
| 04 | `04_module1_encoding (4).ipynb` |  |
| 04 | `04_module1_encoding (5).ipynb` |  |
| 06 | `06_module3_intent.ipynb` |  |

Not yet written:

- `05_Module3_VarianceWeightedTraining.ipynb` — LSTM training, SVAO weighting, lambda calibration
- `06_Module4_ForecastRecombination.ipynb` — Hybrid recombination
- `07_Full_Pipeline.ipynb` — End-to-end run
- `08_Baselines.ipynb` — Persistence, climatology, standalone ARIMA and LSTM
- `09_Benchmark_Evaluation.ipynb` — Metrics, Diebold-Mariano, restart spread
- `10_Ablation_Study.ipynb` — Lambda sweep, normalisation on/off, order sensitivity
- `11_Final_Results_and_Figures.ipynb` — all_paper_numbers.json and every thesis figure

## Reproducing

Obtain the GMet records and verify the digest in `REPRODUCIBILITY.md`,
then run the notebooks in order. Each restores its inputs from Drive and
asserts that its predecessors' artefacts exist, so running out of order
fails loudly rather than producing wrong numbers.

No number appears in the thesis except from `outputs/`.

