#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_downscale_emis_to_eac4_27x29.py

What it does
------------
1) Loads EAC4 Algeria daily dataset on the **27×29** grid.
2) Loads CAMS emissions (monthly or daily) on **whatever grid** (e.g., 195×210).
3) If emissions are monthly (≈120), expands them to **one instance per day** (3653) by equal-weight per month.
4) Spatially regrids emissions to EAC4’s **27×29** grid via **bilinear interpolation** (note: not strictly conservative).
5) Drops **redundant features** from emissions if their **base name** exists in EAC4
   (EAC4 has priority; base-name normalization strips prefixes like 'emis:'/'eac4:' and level suffixes).
6) Time-aligns to the **intersection of days**.
7) Outputs merged arrays in two formats:
   - 4D: (day, feature, 27, 29)
   - flat: (day*27*29, feature)
   Also writes a meta NPZ and a markdown data card.

Inputs (hardcoded)
------------------
./datasets/CAMS/EAC4_ALG_meta.npz
./datasets/CAMS/EAC4_ALG_X.npy        # shape either (T*27*29, F) or (T,27,29,F)

./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_meta.npz
./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_X.npy     # shape either (T*Y*X, F) or (T,Y,X,F)
                                                  # T can be 120 (monthly) or 3653 (daily)

Outputs
-------
./datasets/CAMS/merged_27x29_4d.npy
./datasets/CAMS/merged_27x29_flat.npy
./datasets/CAMS/merged_27x29_meta.npz
./datasets/CAMS/merged_27x29.card.md

Dependencies
------------
- numpy, pandas, xarray (for interpolation), netCDF4 (optional backend for xarray)
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception as e:
    raise SystemExit("Please install xarray (and pandas/numpy).") from e

ROOT = Path("./datasets/CAMS")

# Inputs
EAC4_META = ROOT / "EAC4_ALG_meta.npz"
EAC4_X    = ROOT / "EAC4_ALG_X.npy"

EMIS_META = ROOT / "CAMS_ALG_v6_2_2015_2024_meta.npz"
EMIS_X    = ROOT / "CAMS_ALG_v6_2_2015_2024_X.npy"

# Outputs
OUT_4D    = ROOT / "merged_27x29_4d.npy"
OUT_FLAT  = ROOT / "merged_27x29_flat.npy"
OUT_META  = ROOT / "merged_27x29_meta.npz"
OUT_CARD  = ROOT / "merged_27x29.card.md"

# Verbosity & chunking
PRINT_EVERY_DAYS = 50
FEAT_CHUNK = 32

# --------------------------- helpers ---------------------------
def _detect_matrix_shape(path, T, Y, X):
    """Return (is2d, F) + memmap either as 2D (T*Y*X,F) or 4D (T,Y,X,F)."""
    mm = np.memmap(path, dtype=np.float32, mode="r")
    total = mm.size
    cells = Y * X
    if total % (T * cells) != 0:
        # Could be the other layout with a different T guess; raise with context
        raise SystemExit(f"[error] Unexpected size for {path}: {total} not divisible by T*cells={T*cells}")
    F = total // (T * cells)
    # Try 2D first
    try:
        X2D = np.memmap(path, dtype=np.float32, mode="r", shape=(T*cells, F))
        return True, F, X2D, None
    except Exception:
        X4D = np.memmap(path, dtype=np.float32, mode="r", shape=(T, Y, X, F))
        return False, F, None, X4D

def _normalize_name(n: str) -> str:
    n = str(n).lower().strip()
    if ":" in n: n = n.split(":", 1)[1]
    for key in ("@isobaricinhpa=", "@level=", "@step="):
        if key in n:
            n = n.split(key, 1)[0]
    return n

def _emissions_expand_monthly_to_daily(meta_time):
    """Return daily index and mapping month_index -> daily positions (equal-weight)."""
    idx_daily = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    month_starts = pd.to_datetime(meta_time).to_period("M").to_timestamp()
    day_to_i = pd.Series(np.arange(idx_daily.size), index=idx_daily)
    month_to_daily_idx = {}
    for ms in month_starts:
        month_days = pd.date_range(ms, ms + pd.offsets.MonthEnd(0), freq="D")
        # intersect with our fixed idx to be safe
        month_to_daily_idx[ms] = day_to_i.index.intersection(month_days)
    return idx_daily, month_to_daily_idx

def _interp_to_eac4_grid(day_yxf, lat_src, lon_src, lat_tgt, lon_tgt):
    """Bilinear interpolate a (Ysrc, Xsrc, F) field to (Ytgt, Xtgt, F)."""
    Ys, Xs, F = day_yxf.shape
    Yt, Xt = lat_tgt.shape[0], lon_tgt.shape[0]
    out = np.empty((Yt, Xt, F), dtype=np.float32)
    src_lat = xr.DataArray(lat_src, dims=("lat",))
    src_lon = xr.DataArray(lon_src, dims=("lon",))
    tgt = {"lat": xr.DataArray(lat_tgt, dims=("lat",)),
           "lon": xr.DataArray(lon_tgt, dims=("lon",))}
    # Loop features to keep memory small
    for k in range(F):
        da = xr.DataArray(day_yxf[:,:,k], dims=("lat","lon"), coords={"lat": src_lat, "lon": src_lon})
        da_i = da.interp(tgt, method="linear")
        # If any NaNs (outside hull), nearest as fallback
        v = da_i.values
        if np.isnan(v).any():
            v2 = xr.DataArray(day_yxf[:,:,k], dims=("lat","lon"),
                              coords={"lat": src_lat, "lon": src_lon}).interp(tgt, method="nearest").values
            nanmask = np.isnan(v)
            v[nanmask] = v2[nanmask]
        out[:,:,k] = v.astype(np.float32, copy=False)
    return out

def main():
    # ------------------ Load metas ------------------
    if not EAC4_META.exists() or not EMIS_META.exists():
        raise SystemExit("Missing input metas.")

    eac4 = np.load(EAC4_META, allow_pickle=True)
    ea_time = eac4["time"]
    ea_lat  = eac4["lat"]
    ea_lon  = eac4["lon"]
    ea_names = eac4.get("feature_names", np.array([], dtype=object))
    Ys, Xs = ea_lat.shape[0], ea_lon.shape[0]
    Te = ea_time.shape[0]
    print(f"[EAC4] grid={Ys}x{Xs}, time={Te}, features={ea_names.size}")

    emis = np.load(EMIS_META, allow_pickle=True)
    em_time = emis["time"] if "time" in emis else (emis["time_index"] if "time_index" in emis else None)
    em_lat  = emis["lat"] if "lat" in emis else emis["target_lat"]
    em_lon  = emis["lon"] if "lon" in emis else emis["target_lon"]
    em_names= emis.get("feature_names", np.array([], dtype=object))
    Yf, Xf = em_lat.shape[0], em_lon.shape[0]
    Tm = None if em_time is None else em_time.shape[0]
    print(f"[EMIS] grid={Yf}x{Xf}, time={Tm if Tm is not None else 'unknown'}, features={em_names.size}")

    # ------------------ Open matrices ------------------
    # EAC4 is assumed daily already; detect layout
    print("[EAC4] opening data matrix…")
    is2d_ea, Fe, EA2D, EA4D = _detect_matrix_shape(EAC4_X, Te, Ys, Xs)
    print(f"[EAC4] features={Fe}, layout={'2D' if is2d_ea else '4D'}")

    # Emissions time determine: monthly (≈120) or daily (3653)
    if em_time is None:
        raise SystemExit("[error] Emissions meta missing 'time' or 'time_index'.")
    # detect layout for emissions using em_time & its grid
    print("[EMIS] opening data matrix…")
    is2d_em, Fm, EM2D, EM4D = _detect_matrix_shape(EMIS_X, em_time.shape[0], Yf, Xf)
    print(f"[EMIS] features={Fm}, layout={'2D' if is2d_em else '4D'}")

    # ------------------ Build target daily index ------------------
    ea_days = pd.to_datetime(ea_time).normalize()
    if em_time.dtype.name.startswith("datetime64"):
        em_steps = pd.to_datetime(em_time).normalize()
    else:
        # String or period-like → parse
        em_steps = pd.to_datetime(em_time).normalize()

    # If emissions are monthly (len≈120), expand to daily equal-weight
    monthly_mode = (em_steps.nunique() <= 130) and (em_steps.to_period("M").nunique() >= 100)
    if monthly_mode:
        print("[EMIS] Detected monthly emissions; expanding to daily equal-weight…")
        idx_daily, month_to_daily_idx = _emissions_expand_monthly_to_daily(em_time)
        D = idx_daily.size
    else:
        print("[EMIS] Detected daily emissions.")
        # assume same canonical day range as EAC4
        idx_daily = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        D = idx_daily.size

    # ------------------ Deduplicate names ------------------
    ea_base = set(_normalize_name(n) for n in ea_names)
    emis_keep_mask = np.ones(Fm, dtype=bool)
    for j, n in enumerate(em_names):
        if _normalize_name(n) in ea_base:
            emis_keep_mask[j] = False
    kept_idx = np.where(emis_keep_mask)[0]
    Fm_kept = int(kept_idx.size)
    print(f"[dedupe] Dropping {Fm - Fm_kept} emissions features that duplicate EAC4 names; keeping {Fm_kept}.")

    # ------------------ Time alignment ------------------
    # Build mapping day -> EAC4 timestep indices
    ea_day_to_tidx = {}
    for ti, d in enumerate(ea_days):
        ea_day_to_tidx.setdefault(d, []).append(ti)
    # Build emissions mapping depending on monthly/daily
    if monthly_mode:
        month_starts = pd.to_datetime(em_time).to_period("M").to_timestamp()
        # month index to monthly time position
        month_to_mi = {pd.to_datetime(ms): i for i, ms in enumerate(month_starts)}
    else:
        em_day_to_tidx = {}
        for ti, d in enumerate(em_steps):
            em_day_to_tidx.setdefault(d, []).append(ti)

    # ------------------ Prepare outputs ------------------
    cells = Ys * Xs
    Ftot = Fe + Fm_kept
    print(f"[shape] days={D}, grid={Ys}x{Xs}, Fe={Fe}, Fm_kept={Fm_kept}, Ftot={Ftot}")
    OUT4 = np.memmap(OUT_4D, dtype=np.float32, mode="w+", shape=(D, Ftot, Ys, Xs))
    OUT2 = np.memmap(OUT_FLAT, dtype=np.float32, mode="w+", shape=(D*cells, Ftot))

    # ------------------ Process day by day ------------------
    for di, day in enumerate(idx_daily):
        # --- EAC4 daily mean from its (possibly sub-daily) timesteps ---
        ea_idxs = ea_day_to_tidx.get(day, [])
        if not ea_idxs:
            # No data for this day
            ea_day_yxf = np.full((Ys, Xs, Fe), np.nan, dtype=np.float32)
        else:
            if is2d_ea:
                stack = []
                for ti in ea_idxs:
                    r0 = ti * cells; r1 = r0 + cells
                    block = np.asarray(EA2D[r0:r1, :], dtype=np.float32).reshape(Ys, Xs, Fe)  # (Y,X,F)
                    stack.append(block[None, ...])
                ea_day_yxf = np.nanmean(np.concatenate(stack, axis=0), axis=0)  # (Y,X,F)
            else:
                stack = np.asarray([EA4D[ti] for ti in ea_idxs])  # (K, Y, X, F)
                ea_day_yxf = np.nanmean(stack, axis=0)            # (Y, X, F)

        # --- Emissions for this day on native grid (Yf, Xf, Fm_kept) ---
        if monthly_mode:
            # Which month?
            ms = pd.to_datetime(f"{day.year}-{day.month:02d}-01")
            mi = month_to_mi.get(ms, None)
            if mi is None:
                em_day_yxf = np.full((Ys, Xs, Fm_kept), np.nan, dtype=np.float32)
            else:
                if is2d_em:
                    r0 = mi * (Yf*Xf); r1 = r0 + (Yf*Xf)
                    monthly_yxf = np.asarray(EM2D[r0:r1, :], dtype=np.float32).reshape(Yf, Xf, Fm)[:, :, kept_idx]
                else:
                    monthly_yxf = np.asarray(EM4D[mi], dtype=np.float32)[:, :, kept_idx]  # (Yf,Xf,Fm_kept)
                # interp to EAC4 grid
                em_day_yxf = _interp_to_eac4_grid(monthly_yxf, em_lat, em_lon, ea_lat, ea_lon)  # (Ys, Xs, Fm_kept)
        else:
            idxs = em_day_to_tidx.get(day, [])
            if not idxs:
                em_day_yxf = np.full((Ys, Xs, Fm_kept), np.nan, dtype=np.float32)
            else:
                if is2d_em:
                    # mean over intraday if any
                    stack = []
                    for ti in idxs:
                        r0 = ti * (Yf*Xf); r1 = r0 + (Yf*Xf)
                        block = np.asarray(EM2D[r0:r1, :], dtype=np.float32).reshape(Yf, Xf, Fm)[:, :, kept_idx]
                        stack.append(block[None, ...])
                    native_mean = np.nanmean(np.concatenate(stack, axis=0), axis=0)  # (Yf,Xf,Fm_kept)
                else:
                    stack = np.asarray([EM4D[ti] for ti in idxs])  # (K, Yf, Xf, Fm)
                    native_mean = np.nanmean(stack, axis=0)[:, :, kept_idx]
                # interp to EAC4 grid
                em_day_yxf = _interp_to_eac4_grid(native_mean, em_lat, em_lon, ea_lat, ea_lon)

        # --- Combine (EAC4 first, then emissions-kept) ---
        day_fx_yx = np.concatenate([np.transpose(ea_day_yxf, (2,0,1)),  # (Fe,Y,X)
                                    np.transpose(em_day_yxf, (2,0,1))], axis=0)  # (Fm_kept,Y,X)
        OUT4[di] = day_fx_yx
        OUT2[di*cells:(di+1)*cells] = np.transpose(day_fx_yx, (1,2,0)).reshape(cells, Ftot)

        if (di+1) % PRINT_EVERY_DAYS == 0 or di == D-1:
            OUT4.flush(); OUT2.flush()
            print(f"[write] day {di+1}/{D} — {str(day.date())}")

    del OUT4; del OUT2

    # ------------------ Write meta + card ------------------
    # Assemble feature names (prefixes eac4: / emis: kept)
    ea_names_pref = np.array([f"eac4:{str(n)}" for n in ea_names], dtype=object)
    em_names_pref = np.array([f"emis:{str(n)}" for n in em_names[kept_idx]], dtype=object)
    merged_names = np.concatenate([ea_names_pref, em_names_pref])

    np.savez_compressed(
        OUT_META,
        days=pd.date_range("2015-01-01","2024-12-31",freq="D").values.astype("datetime64[ns]"),
        lat=ea_lat, lon=ea_lon,
        feature_names=merged_names,
        shape_4d=np.array([D, Ftot, Ys, Xs], dtype=np.int64),
        notes=np.array({
            "grid": "EAC4 27x29 target grid; emissions bilinear-interpolated (non-conservative).",
            "time": "One instance per day. If emissions were monthly, expanded equal-weight per day.",
            "dedupe": "Dropped emissions features whose normalized names exist in EAC4; EAC4 kept with priority.",
            "order": "Feature axis = [EAC4 features, then kept emissions features]."
        }, dtype=object),
        sources=np.array({
            "eac4_meta": str(EAC4_META),
            "eac4_X": str(EAC4_X),
            "emissions_meta": str(EMIS_META),
            "emissions_X": str(EMIS_X),
        }, dtype=object)
    )
    approx_gb_4d = (D * Ftot * Ys * Xs * 4) / (1024**3)
    with open(OUT_CARD, "w", encoding="utf-8") as f:
        f.write("# Merged Daily Dataset on EAC4 Grid (27×29)\n\n")
        f.write("## Files\n")
        f.write(f"- **4D**: `{OUT_4D}`  — shape `(days={D}, features={Ftot}, y={Ys}, x={Xs})`\n")
        f.write(f"- **Flat**: `{OUT_FLAT}`  — shape `(days*{Ys}*{Xs}, features={Ftot})`\n")
        f.write(f"- **Meta**: `{OUT_META}`\n\n")
        f.write("## Time & Grid\n")
        f.write(f"- Days: `{D}` (2015-01-01 → 2024-12-31)\n")
        f.write(f"- Grid: `{Ys} × {Xs}` (EAC4 Algeria subgrid)\n\n")
        f.write("## Features\n")
        f.write(f"- EAC4 features: `{Fe}`\n")
        f.write(f"- Emissions kept (after dedupe): `{Fm_kept}`\n")
        f.write(f"- **Merged total**: `{Ftot}`\n\n")
        f.write("## Notes\n")
        f.write("- Emissions → EAC4 grid via bilinear interpolation (not strictly conservative).\n")
        f.write("- If emissions were monthly, daily expansion uses equal weights within each month.\n")
        f.write("- Duplicated variables by base-name were removed from emissions (EAC4 has priority).\n")
        f.write(f"- Approx raw 4D size (float32): ~{approx_gb_4d:.2f} GB.\n")

    print("✅ Done. Wrote:")
    print(f"  - {OUT_4D}")
    print(f"  - {OUT_FLAT}")
    print(f"  - {OUT_META}")
    print(f"  - {OUT_CARD}")

if __name__ == "__main__":
    main()
