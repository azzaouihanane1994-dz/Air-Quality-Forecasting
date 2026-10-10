#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_eac4_cams_daily_195x210.py

Inputs (hardcoded, relative to repo root):
  ./datasets/CAMS/EAC4_ALG_meta.npz
  ./datasets/CAMS/EAC4_ALG_X.npy                  # shape either (T*27*29, F) or (T,27,29,F)
  ./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_meta.npz
  ./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_X.npy   # shape either (120*195*210, F) or (120,195,210,F)

Outputs (daily, 2015-01-01..2024-12-31 = 3653 days):
  ./datasets/CAMS/eac4_daily_195x210_4d.npy       # (D, F_eac4, 195, 210)
  ./datasets/CAMS/eac4_daily_195x210_flat.npy     # (D*195*210, F_eac4)
  ./datasets/CAMS/cams_emis_daily_4d.npy          # (D, F_emis, 195, 210)
  ./datasets/CAMS/cams_emis_daily_flat.npy        # (D*195*210, F_emis)
  Plus companion metadata NPZs and cards for each.

Notes:
- EAC4 12-hourly slices are aggregated to **daily mean** BEFORE spatial interpolation.
- Spatial interpolation EAC4 → 195x210 uses xarray.interp (bilinear). Edges use nearest fill.
- Emissions monthly → daily uses equal-weight profile per month (conserves monthly mean).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception as e:
    raise SystemExit("Please install xarray and pandas/numpy (conda recommended).") from e

# ------------------------ Hardcoded paths ------------------------
ROOT = Path("./datasets/CAMS")
EAC4_META = ROOT / "EAC4_ALG_meta.npz"
EAC4_X    = ROOT / "EAC4_ALG_X.npy"

EMIS_META = ROOT / "CAMS_ALG_v6_2_2015_2024_meta.npz"
EMIS_X    = ROOT / "CAMS_ALG_v6_2_2015_2024_X.npy"

# Outputs
EAC4_OUT_4D   = ROOT / "eac4_daily_195x210_4d.npy"
EAC4_OUT_FLAT = ROOT / "eac4_daily_195x210_flat.npy"
EAC4_META_OUT = ROOT / "eac4_daily_195x210_meta.npz"
EAC4_CARD     = ROOT / "eac4_daily_195x210.card.md"

EMIS_OUT_4D   = ROOT / "cams_emis_daily_4d.npy"
EMIS_OUT_FLAT = ROOT / "cams_emis_daily_flat.npy"
EMIS_META_OUT = ROOT / "cams_emis_daily_meta.npz"
EMIS_CARD     = ROOT / "cams_emis_daily.card.md"
# ---------------------------------------------------------------

def _detect_matrix_shape(path, T, Y, X):
    """Return (is2d, F) and a memmap view for either 2D (T*Y*X, F) or 4D (T,Y,X,F)."""
    mm = np.memmap(path, dtype=np.float32, mode="r")
    total = mm.size
    cells = Y * X
    if total % (T * cells) != 0:
        raise SystemExit(f"[error] Unexpected size for {path}: {total} not divisible by T*cells={T*cells}")
    F = total // (T * cells)
    # Try 2D first
    try:
        X2D = np.memmap(path, dtype=np.float32, mode="r", shape=(T * cells, F))
        return True, F, X2D, None
    except Exception:
        X4D = np.memmap(path, dtype=np.float32, mode="r", shape=(T, Y, X, F))
        return False, F, None, X4D

def _interp_eac4_day_to_emis_grid(day_cube_fx_yx, lat_src, lon_src, lat_tgt, lon_tgt):
    """
    day_cube_fx_yx: np.ndarray (F, Ys, Xs) for one day
    Returns: np.ndarray (F, Yt, Xt) bilinearly interpolated
    """
    Ys, Xs = lat_src.shape[0], lon_src.shape[0]
    Yt, Xt = lat_tgt.shape[0], lon_tgt.shape[0]
    out = np.empty((day_cube_fx_yx.shape[0], Yt, Xt), dtype=np.float32)
    # Build xarray grid once per day; loop over features
    src_lat = xr.DataArray(lat_src, dims=("lat",))
    src_lon = xr.DataArray(lon_src, dims=("lon",))
    tgt = {"lat": xr.DataArray(lat_tgt, dims=("lat",)),
           "lon": xr.DataArray(lon_tgt, dims=("lon",))}
    for k in range(day_cube_fx_yx.shape[0]):
        da = xr.DataArray(day_cube_fx_yx[k], dims=("lat","lon"), coords={"lat": src_lat, "lon": src_lon})
        # bilinear
        da_i = da.interp(tgt, method="linear")
        # pad edges with nearest if any NaNs appear (outside convex hull)
        if np.isnan(da_i.values).any():
            da_i = da.interp(tgt, method="nearest")
        out[k] = da_i.values.astype(np.float32, copy=False)
    return out

def _build_daily_index(tmin="2015-01-01", tmax="2024-12-31"):
    return pd.date_range(tmin, tmax, freq="D")

def eac4_to_daily_195x210():
    print("[EAC4] Loading meta…")
    m = np.load(EAC4_META, allow_pickle=True)
    time = m["time"]
    lat_src = m["lat"]   # 27
    lon_src = m["lon"]   # 29
    feats = m.get("feature_names", None)
    Ys, Xs = lat_src.shape[0], lon_src.shape[0]
    T = time.shape[0]
    print(f"[EAC4] Source grid: {Ys}×{Xs}, time steps: {T} (~12-hourly)")

    print("[EMIS] Loading target grid from emissions meta…")
    em = np.load(EMIS_META, allow_pickle=True)
    lat_tgt = em["lat"] if "lat" in em else em["target_lat"]
    lon_tgt = em["lon"] if "lon" in em else em["target_lon"]
    Yt, Xt = lat_tgt.shape[0], lon_tgt.shape[0]
    print(f"[EMIS] Target grid: {Yt}×{Xt} (should be 195×210)")

    print("[EAC4] Opening matrix and detecting storage shape…")
    is2d, F, X2D, X4D = _detect_matrix_shape(EAC4_X, T, Ys, Xs)
    print(f"[EAC4] Features: {F} | Layout: {'2D (T*Y*X,F)' if is2d else '4D (T,Y,X,F)'}")

    # Group timesteps by day
    idx_daily = _build_daily_index()
    tp = pd.to_datetime(time)
    keys = pd.to_datetime(tp.date)
    groups = {}
    for ti, d in enumerate(keys):
        groups.setdefault(d, []).append(ti)
    D = idx_daily.size
    cells_tgt = Yt * Xt

    print("[EAC4] Allocating outputs…")
    Y4 = np.memmap(EAC4_OUT_4D, dtype=np.float32, mode="w+", shape=(D, F, Yt, Xt))
    Yflat = np.memmap(EAC4_OUT_FLAT, dtype=np.float32, mode="w+", shape=(D * cells_tgt, F))

    print("[EAC4] Processing day-by-day → daily mean, then bilinear to 195×210…")
    for di, day in enumerate(idx_daily):
        idxs = groups.get(day, [])
        if not idxs:
            # No data → NaN day
            Y4[di] = np.nan
            Yflat[di * cells_tgt:(di + 1) * cells_tgt] = np.nan
            continue

        # Stack intraday slices
        if is2d:
            stack = []
            for ti in idxs:
                r0 = ti * (Ys * Xs); r1 = r0 + (Ys * Xs)
                block = np.asarray(X2D[r0:r1, :], dtype=np.float32)  # (cells, F)
                stack.append(block.reshape(Ys, Xs, F)[None, ...])     # (1, Ys, Xs, F)
            arr = np.concatenate(stack, axis=0)                        # (K, Ys, Xs, F)
            daily_mean = np.nanmean(arr, axis=0)                       # (Ys, Xs, F)
            daily_fx_yx = np.transpose(daily_mean, (2, 0, 1))          # (F, Ys, Xs)
        else:
            stack = np.asarray([X4D[ti] for ti in idxs])               # (K, Ys, Xs, F)
            daily_mean = np.nanmean(stack, axis=0)                     # (Ys, Xs, F)
            daily_fx_yx = np.transpose(daily_mean, (2, 0, 1))          # (F, Ys, Xs)

        # Bilinear to target
        daily_tgt = _interp_eac4_day_to_emis_grid(daily_fx_yx, lat_src, lon_src, lat_tgt, lon_tgt)  # (F, Yt, Xt)
        Y4[di] = daily_tgt
        Yflat[di * cells_tgt:(di + 1) * cells_tgt] = np.transpose(daily_tgt, (1, 2, 0)).reshape(cells_tgt, F)

        if (di + 1) % 50 == 0 or di == D - 1:
            Y4.flush(); Yflat.flush()
            print(f"[EAC4] Day {di + 1}/{D} processed — {str(day.date())}")

    del Y4; del Yflat

    # Meta + card
    np.savez_compressed(
        EAC4_META_OUT,
        days=idx_daily.values.astype("datetime64[ns]"),
        lat=lat_tgt, lon=lon_tgt,
        feature_names=feats if feats is not None else np.arange(F),
        shape_4d=np.array([D, F, Yt, Xt], dtype=np.int64),
        note="EAC4 daily mean from 12-hourly, bilinear to 195x210."
    )
    approx_gb = (D * F * Yt * Xt * 4) / (1024**3)
    with open(EAC4_CARD, "w", encoding="utf-8") as f:
        f.write("# EAC4 → Daily (195×210)\n\n")
        f.write(f"- 4D: `{EAC4_OUT_4D}` (D, F, 195, 210)\n")
        f.write(f"- Flat: `{EAC4_OUT_FLAT}` (D*195*210, F)\n")
        f.write(f"- D={D}, F={F}, approx raw size: ~{approx_gb:.2f} GB\n")
        f.write("- Daily mean from ~12-hourly; bilinear interp to emission grid (nearest at edges).\n")

    print("[EAC4] ✅ Done. Wrote 4D, flat, meta, and card.")

def cams_emissions_monthly_to_daily():
    print("[CAMS-EMIS] Loading meta…")
    m = np.load(EMIS_META, allow_pickle=True)
    time = m["time"] if "time" in m else m["time_index"]
    lat = m["lat"] if "lat" in m else m["target_lat"]
    lon = m["lon"] if "lon" in m else m["target_lon"]
    feats = m.get("feature_names", None)
    Tm = time.shape[0]  # ~120 months
    Yt, Xt = lat.shape[0], lon.shape[0]
    print(f"[CAMS-EMIS] Grid: {Yt}×{Xt}, months: {Tm}")

    print("[CAMS-EMIS] Detecting X.npy storage…")
    is2d, F, X2D, X4D = _detect_matrix_shape(EMIS_X, Tm, Yt, Xt)
    print(f"[CAMS-EMIS] Features: {F} | Layout: {'2D' if is2d else '4D'}")

    # Build daily index & month groups
    idx_daily = _build_daily_index()
    D = idx_daily.size
    month_starts = pd.to_datetime(time).to_period("M").to_timestamp()
    # Map month -> list of day positions
    month_to_daily_idx = {}
    daily_series = pd.Series(np.arange(D), index=idx_daily)
    for ms in month_starts:
        # All days within that month
        month_days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0), freq="D")
        month_to_daily_idx[ms] = daily_series.loc[month_days].values

    # Prepare outputs
    Y4 = np.memmap(EMIS_OUT_4D, dtype=np.float32, mode="w+", shape=(D, F, Yt, Xt))
    Yflat = np.memmap(EMIS_OUT_FLAT, dtype=np.float32, mode="w+", shape=(D * Yt * Xt, F))

    print("[CAMS-EMIS] Allocating daily (equal-weight within month, conserving monthly mean)…")
    for mi, ms in enumerate(month_starts):
        # Fetch monthly field (Yt, Xt, F)
        if is2d:
            r0 = mi * (Yt * Xt); r1 = r0 + (Yt * Xt)
            month_cells_feat = np.asarray(X2D[r0:r1, :], dtype=np.float32)     # (cells, F)
            monthly_yxf = month_cells_feat.reshape(Yt, Xt, F)                   # (Y,X,F)
        else:
            monthly_yxf = np.asarray(X4D[mi], dtype=np.float32)                 # (Y,X,F)

        # Equal-weight allocation across month days
        d_idx = month_to_daily_idx.get(ms, [])
        if len(d_idx) == 0:
            continue
        # The same monthly field for each day
        for di in d_idx:
            Y4[di] = np.transpose(monthly_yxf, (2, 0, 1))                       # (F,Y,X)
            Yflat[di * (Yt * Xt):(di + 1) * (Yt * Xt)] = monthly_yxf.reshape(Yt * Xt, F)
        if (mi + 1) % 6 == 0 or mi == len(month_starts) - 1:
            Y4.flush(); Yflat.flush()
            print(f"[CAMS-EMIS] Month {mi + 1}/{Tm} allocated — {ms.strftime('%Y-%m')}")

    del Y4; del Yflat

    np.savez_compressed(
        EMIS_META_OUT,
        days=idx_daily.values.astype("datetime64[ns]"),
        lat=lat, lon=lon,
        feature_names=feats if feats is not None else np.arange(F),
        shape_4d=np.array([D, F, Yt, Xt], dtype=np.int64),
        note="CAMS emissions monthly replicated to each day within month (equal-weight)."
    )
    approx_gb = (D * F * Yt * Xt * 4) / (1024**3)
    with open(EMIS_CARD, "w", encoding="utf-8") as f:
        f.write("# CAMS Emissions → Daily (195×210)\n\n")
        f.write(f"- 4D: `{EMIS_OUT_4D}` (D, F, 195, 210)\n")
        f.write(f"- Flat: `{EMIS_OUT_FLAT}` (D*195*210, F)\n")
        f.write(f"- D={D}, F={F}, approx raw size: ~{approx_gb:.2f} GB\n")
        f.write("- Daily allocation within each month is equal-weight (conserves monthly mean).\n")

    print("[CAMS-EMIS] ✅ Done. Wrote 4D, flat, meta, and card.")

def main():
    print("=== Step 1/2: EAC4 → daily (mean) and upsample to 195×210 ===")
    eac4_to_daily_195x210()

    print("\n=== Step 2/2: CAMS emissions monthly → daily (equal-weight) ===")
    cams_emissions_monthly_to_daily()

    print("\nAll done. Outputs are in ./datasets/CAMS.")

if __name__ == "__main__":
    main()
