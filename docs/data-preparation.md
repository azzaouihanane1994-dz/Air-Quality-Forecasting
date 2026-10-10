# Data preparation

Run scripts from the repository root. Paths to raw data are configured near the top of each preprocessing script.

## Source inputs

- CAMS EAC4 GRIB: `datasets/CAMS/CAMS global reanalysis EAC4_26features.grib`.
- CAMS emissions: files under `datasets/CAMS/CAMS global emission inventories/`.
- ERA5 daily fields aligned to the EAC4 grid.
- HOTOSM Algeria roads, buildings and populated-place GeoPackages under `datasets/humdata/`.
- NASA fire detections under `datasets/nasa_fires/`.

Acquisition helpers are in `datasates_downloaders/`. Sentinel-5P helpers are provided as supporting utilities; their existence does not imply that satellite observations were used for independent validation.

## Processing order

```bash
python "datasets_preprocessing_2/1- consolidate_cams_algeria_emission_inventories  - monthly.py"
python "datasets_preprocessing_2/2- extract_eac4_algeria_ml.py"
python "datasets_preprocessing_2/3- merge_downscale_emis_to_eac4_27x29.py"
python "datasets_preprocessing_2/4- inspect_hotosm_gpks.py"
python "datasets_preprocessing_2/5 -build_and_append_static_humdata.py"
python "datasets_preprocessing_2/6- prepare_nasa_fire_labels.py"
```

The first three scripts consolidate emissions, extract daily EAC4 fields and interpolate emissions onto the 27 × 29 grid. Static and fire preprocessing produce their own arrays and metadata. The `copy.py` static script and the earlier `datasets_preprocessing/` directory are retained for provenance; choose one static-processing variant rather than running both blindly.

The supplied source does **not** contain a dedicated ERA5-to-ready-tensor exporter. The retained ERA5 data card describes daily averaging of available 00Z/12Z fields, nearest-grid mapping, and selected/engineered features. Obtain the original ERA5 tensor or implement that conversion against the retained metadata before running tensor assembly. This is a required input, not a step automated by this repository.

Place the following matching tensor/metadata pairs in `datasets/ready/`:

| Tensor | Metadata |
| --- | --- |
| `CAMS_27x29_4d.npy` | `CAMS_27x29_meta.npz` |
| `ERA5_27x29_4d.npy` | `ERA5_27x29_meta.npz` |
| `FIRE_ALG_27x29_4d.npy` | `FIRE_ALG_27x29_meta.npz` |
| `static_humdata_27x29_3d.npy` | `static_humdata_27x29_meta.npz` |

The upstream scripts use other output filenames and directories. Match the tensors to the ready-data cards and move/copy their corresponding metadata together before assembly; do not rename incompatible arrays solely to satisfy a filename.

```bash
python ML_1/1_prepare_global_dataset.py
python ML_1/2_sanity_check_global_dataset.py
```

Expected daily output keys include `X_4d`, `y_reg_4d`, `y_class_4d`, `y_aqi_4d`, `dates_4d`, `lat`, `lon`, `feature_names` and `target_names`. The recorded feature tensor shape is `(3653, 48, 27, 29)` and regression target shape is `(3653, 4, 27, 29)`.

Regression storage units are kg/m³; final evaluation multiplies by `1e9` to obtain µg/m³. PM₂.₅ exceedance uses 25 µg/m³. The four-category AQI map uses the thresholds recorded in the dataset metadata; treat it as this study's category scheme rather than an automatically interchangeable regulatory AQI standard.

See `excluded-files.csv` for every file omitted by the 50 MB limit. Full tensors and some prediction/checkpoint files must be regenerated or supplied outside Git. Upstream providers' access and redistribution terms continue to apply.
