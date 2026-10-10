#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity check for global_airpollution_2015_2024_*.npz datasets.

Reads:
    global_airpollution_2015_2024_4d.npz
    global_airpollution_2015_2024_flat.npz

Prints and writes to:
    global_airpollution_2015_2024_summary.txt

Information reported:
    - Keys inside each .npz
    - Shapes of all main arrays
    - Number of days / samples
    - Grid size, number of features, number of targets
    - Date ranges (full period)
    - Example feature and target names
    - Class label distribution:
        * binary PM2.5 exceedance (y_class)
        * multi-class AQI (y_aqi)
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("./datasets")
NPZ_4D = ROOT / "global_airpollution_2015_2024_4d.npz"
NPZ_FLAT = ROOT / "global_airpollution_2015_2024_flat.npz"
OUT_TXT = ROOT / "global_airpollution_2015_2024_summary.txt"


def add(line_list, text=""):
    """Append a line to our report buffer."""
    line_list.append(str(text))


def summary_4d(lines):
    add(lines, "#" * 80)
    add(lines, "# 4D DATASET")
    add(lines, "#" * 80)

    if not NPZ_4D.exists():
        add(lines, f"File not found: {NPZ_4D}")
        return

    data = np.load(NPZ_4D, allow_pickle=True)

    add(lines, f"File: {NPZ_4D.resolve()}")
    add(lines, f"Keys: {sorted(data.files)}")
    add(lines)

    # Basic core arrays (new format)
    X_4d = data["X_4d"]             # [D, F, H, W]
    y_reg_4d = data["y_reg_4d"]     # [D, 4, H, W]
    y_class_4d = data["y_class_4d"] # [D, 1, H, W]  (binary PM2.5)
    y_aqi_4d = data["y_aqi_4d"]     # [D, 1, H, W]  (multi-class AQI)

    dates_4d = data["dates_4d"].astype("datetime64[D]")
    lat = data["lat"]
    lon = data["lon"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]
    pm25_threshold = float(data["pm25_threshold"])

    # AQI metadata (if present)
    aqi_class_names = data.get("aqi_class_names", None)
    aqi_bounds_pm25 = data.get("aqi_bounds_pm25_ugm3", None)
    aqi_bounds_pm10 = data.get("aqi_bounds_pm10_ugm3", None)
    aqi_bounds_go3  = data.get("aqi_bounds_go3_ugm3", None)
    aqi_bounds_no2  = data.get("aqi_bounds_no2_ugm3", None)

    D, F_tot, H, W = X_4d.shape

    add(lines, f"X_4d shape:        {X_4d.shape}  # (days, features, lat, lon)")
    add(lines, f"y_reg_4d shape:    {y_reg_4d.shape}  # (days, 4, lat, lon)")
    add(lines, f"y_class_4d shape:  {y_class_4d.shape}  # (days, 1, lat, lon)  [binary PM2.5]")
    add(lines, f"y_aqi_4d shape:    {y_aqi_4d.shape}  # (days, 1, lat, lon)  [multi-class AQI]")
    add(lines)
    add(lines, f"Grid: lat={H}, lon={W}, total cells={H*W}")
    add(lines, f"Features (non-target): {F_tot}")
    add(lines, f"Regression targets: {len(target_names)} -> {list(target_names)}")
    add(lines, f"PM2.5 threshold for binary classification: {pm25_threshold} µg/m³")
    add(lines)

    # AQI meta info
    if aqi_class_names is not None:
        add(lines, "AQI class names (integer index -> name):")
        for i, name in enumerate(aqi_class_names):
            add(lines, f"  {i}: {name}")
        add(lines)

    if all(v is not None for v in [aqi_bounds_pm25, aqi_bounds_pm10, aqi_bounds_go3, aqi_bounds_no2]):
        add(lines, "AQI concentration bounds (µg/m³) per pollutant:")
        add(lines, f"  PM2.5: {np.array2string(aqi_bounds_pm25, precision=2)}")
        add(lines, f"  PM10 : {np.array2string(aqi_bounds_pm10, precision=2)}")
        add(lines, f"  O3   : {np.array2string(aqi_bounds_go3,  precision=2)}")
        add(lines, f"  NO2  : {np.array2string(aqi_bounds_no2,  precision=2)}")
        add(lines)

    # Dates
    dates_pd = pd.to_datetime(dates_4d)
    add(lines, f"Days (unique): {len(dates_4d)}")
    add(lines, f"Date range: {dates_pd.min().date()}  -->  {dates_pd.max().date()}")
    add(lines)

    # Feature name preview
    add(lines, "First 10 feature names:")
    for fn in feature_names[:10]:
        add(lines, f"  - {fn}")
    if len(feature_names) > 10:
        add(lines, f"  ... (total {len(feature_names)} features)")
    add(lines)

    # ---------------- Binary classification stats (all days) ----------------
    cls_all = (y_class_4d.reshape(D, -1) > 0).astype(np.int32)
    polluted_count = int(cls_all.sum())
    total_cells = int(cls_all.size)
    frac = polluted_count / total_cells if total_cells > 0 else 0.0
    add(lines, "Global binary PM2.5 classification statistics (all days):")
    add(lines, f"  polluted cells (label=1): {polluted_count}")
    add(lines, f"  total cells:              {total_cells}")
    add(lines, f"  polluted fraction:        {frac:.4f}")
    add(lines)

    # ---------------- AQI class distribution (all days) ---------------------
    aqi_all = y_aqi_4d.reshape(D, -1).astype(int)  # [D, H*W]
    aqi_flat = aqi_all.ravel()
    unique_aqi, counts_aqi = np.unique(aqi_flat, return_counts=True)

    add(lines, "Global AQI class distribution (all days):")
    n_total_aqi = int(aqi_flat.size)
    for c, cnt in zip(unique_aqi, counts_aqi):
        frac_c = cnt / n_total_aqi if n_total_aqi > 0 else 0.0
        if aqi_class_names is not None and c < len(aqi_class_names):
            name = aqi_class_names[c]
            add(lines, f"  class {c} ({name}): {cnt} ({frac_c:.4f})")
        else:
            add(lines, f"  class {c}: {cnt} ({frac_c:.4f})")
    add(lines)


def summary_flat(lines):
    add(lines, "#" * 80)
    add(lines, "# FLAT DATASET")
    add(lines, "#" * 80)

    if not NPZ_FLAT.exists():
        add(lines, f"File not found: {NPZ_FLAT}")
        return

    data = np.load(NPZ_FLAT, allow_pickle=True)

    add(lines, f"File: {NPZ_FLAT.resolve()}")
    add(lines, f"Keys: {sorted(data.files)}")
    add(lines)

    # New flat arrays (no train/test split)
    X_flat = data["X_flat"]              # [N, F]
    y_reg_flat = data["y_reg_flat"]      # [N, 4]
    y_class_flat = data["y_class_flat"]  # [N]  binary
    y_aqi_flat = data["y_aqi_flat"]      # [N]  multi-class AQI

    dates_flat = data["dates_flat"].astype("datetime64[D]")
    lat_flat = data["lat_flat"]
    lon_flat = data["lon_flat"]

    feature_names = data["feature_names"]
    target_names = data["target_names"]
    pm25_threshold = float(data["pm25_threshold"])
    aqi_class_names = data.get("aqi_class_names", None)

    add(lines, f"X_flat shape:       {X_flat.shape}  # (samples, features)")
    add(lines, f"y_reg_flat shape:   {y_reg_flat.shape}  # (samples, 4)")
    add(lines, f"y_class_flat shape: {y_class_flat.shape}  # (samples,)  [binary PM2.5]")
    add(lines, f"y_aqi_flat shape:   {y_aqi_flat.shape}  # (samples,)  [multi-class AQI]")
    add(lines)

    n_samples, n_features = X_flat.shape
    add(lines, f"Number of samples:  {n_samples}")
    add(lines, f"Number of features: {n_features}")
    add(lines, f"Regression targets: {list(target_names)}")
    add(lines, f"PM2.5 threshold for binary classification: {pm25_threshold} µg/m³")
    add(lines)

    # Dates
    dt = pd.to_datetime(dates_flat)
    add(lines, "Dates (unique over all samples):")
    add(lines, f"  count days: {len(np.unique(dates_flat))}")
    add(lines, f"  range: {dt.min().date()}  -->  {dt.max().date()}")
    add(lines)

    # Feature name preview
    add(lines, "First 10 feature names:")
    for fn in feature_names[:10]:
        add(lines, f"  - {fn}")
    if len(feature_names) > 10:
        add(lines, f"  ... (total {len(feature_names)} features)")
    add(lines)

    # ---------------- Binary classification distribution --------------------
    yb = np.asarray(y_class_flat).astype(int)
    n = yb.size
    pos = int((yb == 1).sum())
    neg = int((yb == 0).sum())
    add(lines, "Binary PM2.5 classification stats (flat, all samples):")
    add(lines, f"  total samples: {n}")
    add(lines, f"  polluted=1:    {pos} ({pos / n:.4f})")
    add(lines, f"  clean=0:       {neg} ({neg / n:.4f})")
    add(lines)

    # ---------------- AQI class distribution (flat) -------------------------
    ya = np.asarray(y_aqi_flat).astype(int)
    unique_aqi, counts_aqi = np.unique(ya, return_counts=True)
    add(lines, "AQI class distribution (flat, all samples):")
    for c, cnt in zip(unique_aqi, counts_aqi):
        frac_c = cnt / n if n > 0 else 0.0
        if aqi_class_names is not None and c < len(aqi_class_names):
            name = aqi_class_names[c]
            add(lines, f"  class {c} ({name}): {cnt} ({frac_c:.4f})")
        else:
            add(lines, f"  class {c}: {cnt} ({frac_c:.4f})")
    add(lines)

    # Quick sanity on coordinates
    add(lines, "Coordinate sanity (flat):")
    add(lines, f"  lat_flat range: {lat_flat.min():.3f} .. {lat_flat.max():.3f}")
    add(lines, f"  lon_flat range: {lon_flat.min():.3f} .. {lon_flat.max():.3f}")
    add(lines)


def main():
    lines = []
    add(lines, "GLOBAL AIR POLLUTION DATASET SANITY CHECK")
    add(lines, "=" * 80)
    add(lines)

    summary_4d(lines)
    add(lines)
    summary_flat(lines)

    # Join and print
    report = "\n".join(lines)
    print(report)

    # Write to text file
    OUT_TXT.write_text(report, encoding="utf-8")
    print(f"\nSummary written to: {OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()
