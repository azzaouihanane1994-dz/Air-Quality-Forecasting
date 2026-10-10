#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
from datetime import datetime

ROOT = Path("./datasets/CAMS")
META_PATH = ROOT / "ALG_merged_meta.npz"
X_PATH    = ROOT / "ALG_merged_X.npy"
FEATS_CSV = ROOT / "ALG_merged_features.csv"  # full list export

SHOW_FIRST_N = 40  # how many feature names to print to console

def sizeof_gb(path: Path):
    try:
        return os.path.getsize(path) / (1024**3)
    except Exception:
        return None

def main():
    if not META_PATH.exists():
        raise SystemExit(f"Meta not found: {META_PATH}")

    m = np.load(META_PATH, allow_pickle=True)

    # --- core fields ---
    shape_info = json.loads(str(m["shape_info"]))
    n_rows     = shape_info["n_instances"]
    n_feat     = shape_info["n_features"]
    n_time     = shape_info["n_time"]
    n_lat      = shape_info["n_lat"]
    n_lon      = shape_info["n_lon"]
    bbox       = shape_info.get("bbox", {})
    ordering   = shape_info.get("ordering", "")

    time = m["time"].astype("datetime64[ns]")
    lat  = m["lat"].astype(float)
    lon  = m["lon"].astype(float)

    names = m["feature_names"].astype(str)
    units = m["feature_units"].astype(str)
    longs = m["feature_long_names"].astype(str)

    # Optional “quality” JSON (may be empty)
    quality = {}
    if "quality" in m:
        try:
            quality = json.loads(str(m["quality"]))
        except Exception:
            quality = {}

    # --- basic prints ---
    print("="*72)
    print("ALG_merged — quick summary")
    print("="*72)
    print(f"Matrix path:  {X_PATH}")
    sz_gb = sizeof_gb(X_PATH)
    if sz_gb is not None:
        print(f"Matrix size:  ~{sz_gb:.2f} GB (file on disk)")

    print(f"Rows (instances): {n_rows:,}")
    print(f"Columns (features): {n_feat:,}")
    print(f"Time steps: {n_time}  |  Grid: {n_lat} lat × {n_lon} lon  -> rows/step: {n_lat*n_lon:,}")
    if time.size:
        t0 = str(time.min())[:10]
        t1 = str(time.max())[:10]
        print(f"Time coverage: {t0} … {t1}")
    if bbox:
        print("BBox:", f"lat {bbox.get('lat_min')}..{bbox.get('lat_max')}, "
                      f"lon {bbox.get('lon_min')}..{bbox.get('lon_max')}")
    # approx step size
    if lat.size > 1 and lon.size > 1:
        lat_step = float(np.mean(np.diff(np.sort(lat))))
        lon_step = float(np.mean(np.diff(np.sort(lon))))
        print(f"Approx grid step: Δlat≈{lat_step:.4f}°, Δlon≈{lon_step:.4f}°")

    print("\nRow/col ordering:")
    print(f"  {ordering}")

    # --- feature breakdown ---
    emis_mask = np.array([n.startswith("emis:") for n in names])
    eac4_mask = np.array([n.startswith("eac4:") for n in names])
    print("\nFeature groups:")
    print(f"  emissions (prefix 'emis:'): {int(emis_mask.sum())}")
    print(f"  EAC4      (prefix 'eac4:'): {int(eac4_mask.sum())}")

    # optional: “all-NaN” columns (if provided)
    if quality and "keep_columns_mask" in quality:
        keep = np.array(quality["keep_columns_mask"], dtype=bool)
        print(f"  columns with ≥1 finite value: {int(keep.sum())}/{n_feat}")
        if (~keep).any():
            bad_idx = np.where(~keep)[0]
            print(f"  all-NaN columns (first 20): {bad_idx[:20].tolist()} …" if bad_idx.size > 20 else
                  f"  all-NaN columns: {bad_idx.tolist()}")

    # --- print first N feature names nicely ---
    print("\nFirst feature names:")
    for i in range(min(SHOW_FIRST_N, n_feat)):
        unit = units[i] if i < len(units) else ""
        print(f"  [{i:>4}] {names[i]}  ({unit})")

    # --- export full feature list to CSV ---
    try:
        import csv
        with open(FEATS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index","name","unit","long_name"])
            for i, (n, u, l) in enumerate(zip(names, units, longs)):
                w.writerow([i, n, u, l])
        print(f"\nFull feature list written → {FEATS_CSV}")
    except Exception as e:
        print(f"\n[warn] Could not write {FEATS_CSV}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
