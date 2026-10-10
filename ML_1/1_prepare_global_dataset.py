#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare global ML-ready dataset (daily) from ./datasets/ready,
WITHOUT performing any train/test split. The ML training pipeline
will handle splitting later.

Inputs in ./datasets/ready:

    CAMS_27x29_4d.npy           + CAMS_27x29_meta.npz
    ERA5_27x29_4d.npy           + ERA5_27x29_meta.npz
    FIRE_ALG_27x29_4d.npy       + FIRE_ALG_27x29_meta.npz
    static_humdata_27x29_3d.npy + static_humdata_27x29_meta.npz

Targets (from CAMS):
    Regression:     eac4:pm2p5, eac4:pm10, eac4:go3, eac4:no2
    Binary class:   pm25_high = 1 if PM2.5 >= 25 µg/m³
    Multi-class AQI:
        - per-pollutant classes based on thresholds for PM2.5, PM10, O3, NO2
        - final AQI class = max over pollutant-specific classes

Outputs:

    global_airpollution_2015_2024_4d.npz
        X_4d, y_reg_4d
        y_class_4d          (binary PM2.5 label)
        y_aqi_4d            (multi-class AQI)
        dates_4d, lat, lon
        feature_names, target_names
        pm25_threshold
        aqi_class_names
        aqi_bounds_*_ugm3
        grid_shape_4d

    global_airpollution_2015_2024_flat.npz
        X_flat
        y_reg_flat
        y_class_flat
        y_aqi_flat
        dates_flat, lat_flat, lon_flat
        feature_names, target_names
        pm25_threshold
        aqi_class_names
        aqi_bounds_*_ugm3
        grid_shape_4d
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ------------------------------------- #

DATA_DIR = Path("./datasets/ready")

OUT_4D   = Path("./datasets/global_airpollution_2015_2024_4d.npz")
OUT_FLAT = Path("./datasets/global_airpollution_2015_2024_flat.npz")

# Binary PM2.5 pollution threshold (for legacy / binary label)
POLLUTION_THRESHOLD_UGM3 = 25.0                # in µg/m³  (PM2.5)
POLLUTION_THRESHOLD_KGM3 = POLLUTION_THRESHOLD_UGM3 * 1e-9  # µg -> kg

TARGET_VARS = [
    "eac4:pm2p5",   # PM2.5
    "eac4:pm10",    # PM10
    "eac4:go3",     # O3
    "eac4:no2",     # NO2
]

# ------------------ Multi-class AQI configuration ------------------------- #
# We use a simple 4-class AQI:
#   0: good
#   1: moderate
#   2: unhealthy
#   3: very_unhealthy_or_worse
#
# Bounds are expressed in µg/m³ and are EXAMPLE values.
# They can be adjusted to match WHO / EU / EPA breakpoints.

AQI_CLASS_NAMES = np.array(
    ["good", "moderate", "unhealthy", "very_unhealthy_or_worse"],
    dtype=object,
)

# For each pollutant, we define class boundaries (len = n_classes - 1).
# Class is np.digitize(value, bounds): 0 .. len(bounds)
AQI_BOUNDS_PM25_UGM3 = np.array([15.0, 25.0, 50.0], dtype=np.float32)
AQI_BOUNDS_PM10_UGM3 = np.array([30.0, 50.0, 100.0], dtype=np.float32)
AQI_BOUNDS_GO3_UGM3  = np.array([80.0, 120.0, 180.0], dtype=np.float32)
AQI_BOUNDS_NO2_UGM3  = np.array([20.0, 40.0, 80.0], dtype=np.float32)


# ----------------------------- HELPERS ------------------------------------ #

def memmap_4d(base: str):
    """
    Load 4D data via memmap using shape_4d from meta.

    Returns:
        arr_TFHW: ndarray-like, shape (D, F, H, W)
        meta:     np.load handle
    """
    meta_path = DATA_DIR / f"{base}_meta.npz"
    data_path = DATA_DIR / f"{base}_4d.npy"

    meta = np.load(meta_path, allow_pickle=True)
    shape_4d = tuple(meta["shape_4d"])   # stored as (D, F, H, W) or (D, H, W, F)
    lat = meta["lat"]
    lon = meta["lon"]

    D0, D1, D2, D3 = shape_4d
    nlat = len(lat)
    nlon = len(lon)

    # Detect ordering using lat/lon lengths
    if D2 == nlat and D3 == nlon:
        # (D, F, H, W)
        D, F, H, W = shape_4d
        arr = np.memmap(data_path, dtype=np.float32, mode="r", shape=(D, F, H, W))
        arr_TFHW = arr
    elif D1 == nlat and D2 == nlon:
        # (D, H, W, F)
        D, H, W, F = shape_4d
        arr = np.memmap(data_path, dtype=np.float32, mode="r", shape=(D, H, W, F))
        arr_TFHW = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(
            f"Cannot infer (days, features, lat, lon) layout for {base}. "
            f"shape_4d={shape_4d}, len(lat)={nlat}, len(lon)={nlon}"
        )

    return arr_TFHW, meta


def flatten_THWF(arr_THWF: np.ndarray) -> np.ndarray:
    """
    Flatten [time, H, W, F] -> [samples, F] over ALL times.
    """
    T, H, W, F = arr_THWF.shape
    return arr_THWF.reshape(T * H * W, F)


def repeat_metadata_all(dates: np.ndarray,
                        lat: np.ndarray,
                        lon: np.ndarray):
    """
    Create flattened arrays of date, lat, lon per sample
    over ALL times.
    """
    T = len(dates)
    H, W = len(lat), len(lon)

    dates_flat = np.repeat(dates, H * W)

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")  # [H, W]
    lat_flat = np.tile(lat_grid.ravel(), T)
    lon_flat = np.tile(lon_grid.ravel(), T)

    return dates_flat, lat_flat, lon_flat


def build_aqi_multiclass(y_reg_THW4: np.ndarray) -> np.ndarray:
    """
    Build a multi-class AQI index from the 4 regression targets:
        [PM2.5, PM10, O3, NO2] in SI units (kg/m^3), shape [T, H, W, 4].

    Steps:
        - Convert AQI bounds from µg/m³ to kg/m³.
        - For each pollutant, compute a sub-index (0..n_classes-1) via np.digitize.
        - Take max across pollutants to get final AQI class per (t, y, x).

    Returns:
        aqi_THW: int8 array of shape [T, H, W] with values 0..(n_classes-1).
    """
    n_classes = len(AQI_CLASS_NAMES)
    factor = 1e-9  # µg -> kg

    def subindex(data_THW: np.ndarray, bounds_ugm3: np.ndarray) -> np.ndarray:
        bounds_kgm3 = bounds_ugm3 * factor
        # digitize returns 0..len(bounds); this matches our class indices
        idx = np.digitize(data_THW, bounds_kgm3, right=False)
        return idx.astype(np.int8)

    # Indices in TARGET_VARS
    pm25_idx = TARGET_VARS.index("eac4:pm2p5")
    pm10_idx = TARGET_VARS.index("eac4:pm10")
    o3_idx   = TARGET_VARS.index("eac4:go3")
    no2_idx  = TARGET_VARS.index("eac4:no2")

    pm25_sub = subindex(y_reg_THW4[..., pm25_idx], AQI_BOUNDS_PM25_UGM3)
    pm10_sub = subindex(y_reg_THW4[..., pm10_idx], AQI_BOUNDS_PM10_UGM3)
    o3_sub   = subindex(y_reg_THW4[..., o3_idx],   AQI_BOUNDS_GO3_UGM3)
    no2_sub  = subindex(y_reg_THW4[..., no2_idx],  AQI_BOUNDS_NO2_UGM3)

    # Final AQI = max over pollutants
    aqi_THW = np.maximum.reduce([pm25_sub, pm10_sub, o3_sub, no2_sub])

    if aqi_THW.max() >= n_classes:
        raise ValueError(
            f"AQI classes go up to {aqi_THW.max()}, but only "
            f"{n_classes} AQI_CLASS_NAMES provided. Increase AQI_CLASS_NAMES "
            f"or adjust AQI_BOUNDS_*_UGM3."
        )

    return aqi_THW.astype(np.int8)


# ------------------------------ MAIN -------------------------------------- #

def main():
    # --------- CAMS (defines time/space & targets) ------------------------- #
    cams_TFHW, cams_meta = memmap_4d("CAMS_27x29")
    cams_feature_names = list(cams_meta["feature_names"])
    days = cams_meta["days"].astype("datetime64[D]")
    lat = cams_meta["lat"].astype(float)
    lon = cams_meta["lon"].astype(float)

    D, F_cams, H, W = cams_TFHW.shape
    print(f"[CAMS] days={D}, features={F_cams}, grid={H}x{W}")

    # CAMS -> [T, H, W, F_cams] for concatenation/flattening
    cams_THWF = np.transpose(cams_TFHW, (0, 2, 3, 1))

    # indices of target variables
    target_idx = [cams_feature_names.index(v) for v in TARGET_VARS]
    target_idx = np.array(target_idx, dtype=int)

    # regression targets [T, H, W, 4] and then [T, 4, H, W]
    y_reg_THW4 = cams_THWF[..., target_idx]              # [T, H, W, 4]
    y_reg_4d = np.transpose(y_reg_THW4, (0, 3, 1, 2))    # [T, 4, H, W]

    # ----------------- Binary PM2.5 classification ------------------------- #
    pm25_idx_in_targets = TARGET_VARS.index("eac4:pm2p5")
    pm25_THW = y_reg_THW4[..., pm25_idx_in_targets]
    y_class_THW = (pm25_THW >= POLLUTION_THRESHOLD_KGM3).astype(np.int8)   # [T, H, W]
    y_class_4d = y_class_THW[:, None, :, :]                               # [T, 1, H, W]

    # ----------------- Multi-class AQI label ------------------------------- #
    aqi_THW = build_aqi_multiclass(y_reg_THW4)         # [T, H, W], int8
    y_aqi_4d = aqi_THW[:, None, :, :]                  # [T, 1, H, W]

    # CAMS features WITHOUT the 4 targets
    cams_mask = np.ones(F_cams, dtype=bool)
    cams_mask[target_idx] = False
    cams_features_THWF = cams_THWF[..., cams_mask]
    cams_feature_names_clean = [n for i, n in enumerate(cams_feature_names) if cams_mask[i]]

    # --------- ERA5 -------------------------------------------------------- #
    era_TFHW, era_meta = memmap_4d("ERA5_27x29")
    era_feature_names = list(era_meta["feature_names"])
    D_e, F_era, H_e, W_e = era_TFHW.shape
    print(f"[ERA5] days={D_e}, features={F_era}, grid={H_e}x{W_e}")
    assert (D_e, H_e, W_e) == (D, H, W), "ERA5 shape mismatch with CAMS"
    era_THWF = np.transpose(era_TFHW, (0, 2, 3, 1))

    # --------- FIRE_ALG ---------------------------------------------------- #
    fire_TFHW, fire_meta = memmap_4d("FIRE_ALG_27x29")
    fire_feature_names = list(fire_meta["feature_names"])
    D_f, F_fire, H_f, W_f = fire_TFHW.shape
    print(f"[FIRE] days={D_f}, features={F_fire}, grid={H_f}x{W_f}")
    assert (D_f, H_f, W_f) == (D, H, W), "FIRE shape mismatch with CAMS"
    fire_THWF = np.transpose(fire_TFHW, (0, 2, 3, 1))

    # --------- static_humdata (3D) ---------------------------------------- #
    static_meta = np.load(DATA_DIR / "static_humdata_27x29_meta.npz", allow_pickle=True)
    static_3d = np.load(DATA_DIR / "static_humdata_27x29_3d.npy")   # small, ok to load fully
    static_feature_names = list(static_meta["feature_names"])
    s_shape = static_3d.shape

    # infer layout: (H, W, F) or (F, H, W)
    if s_shape[0] == H and s_shape[1] == W:
        # (H, W, F)
        static_HWF = static_3d
    elif s_shape[1] == H and s_shape[2] == W:
        # (F, H, W)
        static_HWF = np.transpose(static_3d, (1, 2, 0))
    else:
        raise ValueError(f"Cannot infer static_humdata layout, shape={s_shape}, grid={H}x{W}")

    H_s, W_s, F_static = static_HWF.shape
    print(f"[STATIC] grid={H_s}x{W_s}, features={F_static}")
    assert (H_s, W_s) == (H, W), "Static grid mismatch with CAMS"

    # expand static over time: [T, H, W, F_static]
    static_THWF = np.broadcast_to(static_HWF, (D, H, W, F_static))

    # --------- Combine ALL features (non-target) --------------------------- #
    X_THWF = np.concatenate(
        [cams_features_THWF, era_THWF, fire_THWF, static_THWF],
        axis=-1
    )  # [T, H, W, F_tot]

    feature_names = (
        cams_feature_names_clean +
        era_feature_names +
        fire_feature_names +
        static_feature_names
    )
    F_tot = len(feature_names)
    assert X_THWF.shape[-1] == F_tot

    # global 4D: (days, features, H, W)
    X_4d = np.transpose(X_THWF, (0, 3, 1, 2))   # [T, F_tot, H, W]

    print("\n[SHAPES] Global 4D arrays (days, features, H, W)")
    print(f"X_4d shape:        {X_4d.shape}")
    print(f"y_reg_4d shape:    {y_reg_4d.shape}")
    print(f"y_class_4d shape:  {y_class_4d.shape} (binary PM2.5)")
    print(f"y_aqi_4d shape:    {y_aqi_4d.shape} (multi-class AQI)")

    # ---------------- Flat (whole period, no split) ------------------------ #
    y_reg_THWF = np.transpose(y_reg_4d, (0, 2, 3, 1))   # [T, H, W, 4]

    X_flat       = flatten_THWF(X_THWF)        # [T*H*W, F_tot]
    y_reg_flat   = flatten_THWF(y_reg_THWF)    # [T*H*W, 4]
    y_class_flat = y_class_THW.reshape(-1).astype(np.int8)  # [T*H*W]
    y_aqi_flat   = aqi_THW.reshape(-1).astype(np.int8)      # [T*H*W]

    dates_flat, lat_flat, lon_flat = repeat_metadata_all(days, lat, lon)

    print("\n[SHAPES] Flat arrays (whole period)")
    print(f"X_flat shape:        {X_flat.shape}")
    print(f"y_reg_flat shape:    {y_reg_flat.shape}")
    print(f"y_class_flat shape:  {y_class_flat.shape}")
    print(f"y_aqi_flat shape:    {y_aqi_flat.shape}")
    print(f"dates_flat shape:    {dates_flat.shape}")
    print(f"lat_flat shape:      {lat_flat.shape}")
    print(f"lon_flat shape:      {lon_flat.shape}")

    # ----------------------------- SAVE 4D --------------------------------- #
    OUT_4D.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_4D,
        X_4d=X_4d,
        y_reg_4d=y_reg_4d,
        y_class_4d=y_class_4d,
        y_aqi_4d=y_aqi_4d,
        dates_4d=days,
        lat=lat,
        lon=lon,
        feature_names=np.array(feature_names),
        target_names=np.array(TARGET_VARS),
        pm25_threshold=POLLUTION_THRESHOLD_UGM3,
        aqi_class_names=AQI_CLASS_NAMES,
        aqi_bounds_pm25_ugm3=AQI_BOUNDS_PM25_UGM3,
        aqi_bounds_pm10_ugm3=AQI_BOUNDS_PM10_UGM3,
        aqi_bounds_go3_ugm3=AQI_BOUNDS_GO3_UGM3,
        aqi_bounds_no2_ugm3=AQI_BOUNDS_NO2_UGM3,
        grid_shape_4d=np.array([D, F_tot, H, W], dtype=np.int32),
    )
    print(f"\nSaved 4D dataset (whole period) to: {OUT_4D.resolve()}")

    # ----------------------------- SAVE FLAT ------------------------------- #
    np.savez_compressed(
        OUT_FLAT,
        X_flat=X_flat,
        y_reg_flat=y_reg_flat,
        y_class_flat=y_class_flat,
        y_aqi_flat=y_aqi_flat,
        dates_flat=dates_flat.astype("datetime64[D]"),
        lat_flat=lat_flat,
        lon_flat=lon_flat,
        feature_names=np.array(feature_names),
        target_names=np.array(TARGET_VARS),
        pm25_threshold=POLLUTION_THRESHOLD_UGM3,
        aqi_class_names=AQI_CLASS_NAMES,
        aqi_bounds_pm25_ugm3=AQI_BOUNDS_PM25_UGM3,
        aqi_bounds_pm10_ugm3=AQI_BOUNDS_PM10_UGM3,
        aqi_bounds_go3_ugm3=AQI_BOUNDS_GO3_UGM3,
        aqi_bounds_no2_ugm3=AQI_BOUNDS_NO2_UGM3,
        grid_shape_4d=np.array([D, F_tot, H, W], dtype=np.int32),
    )
    print(f"Saved flat dataset (whole period) to: {OUT_FLAT.resolve()}")


if __name__ == "__main__":
    main()
