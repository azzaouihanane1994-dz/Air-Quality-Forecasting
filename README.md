# Air Quality Forecasting with WindNO

Source code and experimental outputs for **Wind-Conditioned Neural Operators for Gridded Air-Quality Forecasting**.

WindNO forecasts daily PM₂.₅, PM₁₀, O₃ and NO₂ fields over a 27 × 29 grid covering Algeria and its surroundings. It combines a temporal encoder, wind-conditioned latent advection and Fourier operator mixing, with pollutant regression, PM₂.₅ exceedance and four-category AQI heads.

The targets are CAMS EAC4 reanalysis fields. These experiments evaluate reanalysis emulation and gridded risk-category mapping; they do not constitute independent station validation.

## Repository layout

| Path | Contents |
| --- | --- |
| `datasates_downloaders/` | CAMS, NASA and Sentinel-5P acquisition scripts; original directory spelling retained |
| `datasets_preprocessing_2/` | 27 × 29 preprocessing, emissions alignment, static covariates and fire labels |
| `datasets_preprocessing/` | Earlier preprocessing and inspection scripts |
| `ML_1/` | Global tensor assembly, dataset checks and PM₂.₅ distribution analysis |
| `ML_3/WindNO_2.py` | Model implementation and original training/evaluation pipeline |
| `ML_3/experiments.py` | Final chronological protocol, physical-unit evaluation and learned/deterministic baselines |
| `ML_3/event_diagnostics.py` | High-PM₁₀, active-fire and strong-wind diagnostics |
| `ML_3/WindNO_2+ablations.py` | No-diffusion ablation suite, rollout evaluation and comparative plots |
| `ML_3/benchmark_results/` | Final benchmark tables, normalization metadata and eligible checkpoints |
| `ML_3/results/`, `ML_3/logs/` | Original predictions, plots and training logs |
| `ML_3/windno_2_no_diffusion_suite/` | Ablation results and comparisons |
| `ML_3/windno_2_no_diffusion_suite_old/` | Archived earlier ablation outputs |
| `datasets/` | Eligible data, metadata and dataset cards |

Every included file is at most **50,000,000 bytes (50 MB)**. Oversized tensors, predictions and checkpoints are listed in [docs/excluded-files.csv](docs/excluded-files.csv). This repository does not contain the full training dataset.

## Installation

Use Python 3.11 or 3.12 in a virtual environment, then run these commands from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For acquisition and geospatial preprocessing:

```bash
python -m pip install -r requirements-preprocessing.txt
```

Cartopy is optional for the training scripts but required by several standalone geographic plotting scripts. Install the appropriate PyTorch build separately if CUDA support is needed. Dependencies are listed without claiming a frozen environment from the original runs.

## Data preparation

See [docs/data-preparation.md](docs/data-preparation.md) for source inputs, preprocessing order and required tensors. Once `datasets/ready/` contains the complete daily tensors and metadata:

```bash
python ML_1/1_prepare_global_dataset.py
python ML_1/2_sanity_check_global_dataset.py
```

The model input is `datasets/global_airpollution_2015_2024_4d.npz`. Larger data files must be obtained or rebuilt locally. Small included arrays are not a complete substitute for the missing tensors.

## Benchmark experiments

The final protocol trains on **2015–2022**, validates on **2023**, and tests on **2024**, with train-only normalization and regression metrics in µg/m³. Train all four learned baselines across the reported seeds:

```bash
python scripts/run_benchmarks.py --epochs 8 --seeds 42 43 44
python ML_3/event_diagnostics.py
```

The runner calls the existing experiment script for WindNO, ConvLSTM, U-Net and plain FNO. `--epochs` is configurable; the command is a reproduction entry point, not a guarantee of bitwise agreement across platforms. See per-seed configuration and history files for recorded runs.

To regenerate the deterministic baselines, data inventory and aggregate tables without training:

```bash
python ML_3/experiments.py
```

For the original model pipeline and the ablation suite:

```bash
python ML_3/WindNO_2.py
python "ML_3/WindNO_2+ablations.py" --epochs 20 --batch 16
```

These original pipelines have their own configurations, including a training end year of 2023. Use `experiments.py` for the benchmark split. Ablation regression outputs are internal-scale diagnostics and should not be compared directly with the final physical-unit tables.

## Results

The benchmark results report WindNO mean RMSE **24.89 ± 0.54 µg/m³**, PM₂.₅ exceedance F₁ **0.8769 ± 0.0006**, and AQI macro-F₁ **0.5844 ± 0.0023** across three seeds. The plain-FNO regression result is close; use pollutant-level results when comparing models.

See [docs/results.md](docs/results.md) for table locations and interpretation. Existing outputs are retained as recorded; rerunning scripts writes into their configured output directories.

## Credentials and file limits

Downloader credentials are read from environment variables listed in `.env.example`. Set them in your shell; the scripts do not automatically load `.env` files. CAMS and NASA clients use their normal local authentication mechanisms. Never commit credentials.

```bash
python scripts/check_repository.py
```

This checks Python syntax, file sizes and literal credential assignments without downloading data or training models. GitHub Actions runs the same check.

## Citation and license

See [CITATION.cff](CITATION.cff). Journal, DOI and publication metadata can be added when the article is published.

Source code is available under the [MIT license](LICENSE). Third-party datasets retain the terms of their providers; the source-code license does not replace those terms.
