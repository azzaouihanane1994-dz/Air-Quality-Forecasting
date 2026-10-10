# Experimental results

## Final chronological benchmark

Use the following files in `ML_3/benchmark_results/`:

| File | Purpose |
| --- | --- |
| `regression_comparison_ugm3_agg.csv` | Deterministic and learned model comparison in µg/m³ |
| `categorical_comparison_agg.csv` | Seed-aggregated exceedance and AQI metrics |
| `learned_model_regression_seed_summary_raw.csv` | Per-seed pollutant regression metrics |
| `learned_model_classification_seed_summary_raw.csv` | Per-seed classification metrics |
| `class_frequencies_by_split.csv` | Class balance for train/validation/test |
| `event_diagnostics_agg.csv` | Event-subset diagnostics |
| `feature_inventory.csv` | Input features and roles |
| `experiment_summary.json` | Recorded configuration, dataset schema and split counts |

The final split is 2015–2022 / 2023 / 2024. The reported learned-model results aggregate seeds 42, 43 and 44. Per-seed histories and normalization JSON files accompany the tables.

## Original experiments and ablations

`ML_3/results/windno_2/` contains regression plots, confusion matrices, ROC/PR curves, loss curves, geographic examples and oracle/persisted-forcing rollout outputs.

`ML_3/windno_2_no_diffusion_suite/` contains the baseline, `no_advection`, `no_fno_1x1` and `wind_gating_only` variants, plus `compare/` plots and suite summaries. The `_old` directory is an archive, not the primary result source.

These original and ablation pipelines use different configurations from the final chronological benchmark. Their raw regression metrics are internal storage-scale diagnostics. Use the benchmark physical-unit tables for concentration accuracy, and the ablation outputs for controlled component analysis.

Oracle forcing assumes future exogenous inputs are available; persisted forcing holds them fixed. Neither setting establishes operational skill with real forecast meteorology.

## Reproduction limits

Full dataset tensors and any other files above 50 MB are excluded. The complete acquisition/preprocessing chain also requires an ERA5 exporter that was not present in the supplied source. Syntax and packaging checks do not establish that training has been rerun or that outputs reproduce bitwise on another platform.
