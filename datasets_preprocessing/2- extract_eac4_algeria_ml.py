#!/usr/bin/env python3
"""
CAMS Global Reanalysis EAC4 (GRIB) -> Algeria-only, ML-ready (CLEAN, NO SPLITS, VERBOSE)
- Progressive GRIB opener that splits on edition/keys to avoid 'multiple values' stalls.
- Includes ALL variables; level/step dims expand to separate columns.
- Verbose logging; writes a CSV catalog of all final columns.

Inputs:
  ./datasets/CAMS/CAMS global reanalysis EAC4_26features.grib

Outputs:
  ./datasets/CAMS/EAC4_ALG_X.npy         (NumPy memmap: rows x features, float32)
  ./datasets/CAMS/EAC4_ALG_meta.npz      (coords, feature names/units/long names, shapes, quality)
  ./datasets/CAMS/EAC4_ALG_features.csv  (feature catalog)
  ./datasets/CAMS/EAC4_ALG.card.md       (dataset card)
"""

from pathlib import Path
import json, re, csv, time
import numpy as np

try:
    import xarray as xr
except ImportError:
    raise SystemExit("Please install: pip install xarray cfgrib eccodes numpy")

import cfgrib
from cfgrib.dataset import DatasetBuildError

# --------------- Config ---------------
GRIB_PATH = Path("./datasets/CAMS/CAMS global reanalysis EAC4_26features.grib")
OUT_DIR   = Path("./datasets/CAMS")
OUT_X     = OUT_DIR / "EAC4_ALG_X.npy"
OUT_META  = OUT_DIR / "EAC4_ALG_meta.npz"
OUT_CARD  = OUT_DIR / "EAC4_ALG.card.md"
OUT_FEATS = OUT_DIR / "EAC4_ALG_features.csv"

# Algeria bbox
LAT_MIN, LAT_MAX = 18.0, 37.5
LON_MIN, LON_MAX = -9.0, 12.0

TIME_BLOCK = 1
VERBOSE = True
# -------------------------------------

def _unique_vals(index: cfgrib.Index, key: str):
    try:
        vals = index.unique(key)
        return [v.item() if hasattr(v, "item") else v for v in vals if v is not None]
    except Exception:
        return []

def _coord_names(ds: xr.Dataset):
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if lat_name is None or lon_name is None:
        for c in ds.coords:
            cl = c.lower()
            if lat_name is None and "lat" in cl: lat_name = c
            if lon_name is None and ("lon" in cl or "long" in cl): lon_name = c
    if lat_name is None or lon_name is None:
        raise SystemExit("Could not find latitude/longitude coordinate names.")
    return lat_name, lon_name

def _norm_lon_to_180(lon_vals: np.ndarray) -> np.ndarray:
    return ((lon_vals + 180) % 360) - 180 if (lon_vals > 180).mean() > 0.5 else lon_vals

def _bbox_indexers(lat: np.ndarray, lon: np.ndarray):
    lon = _norm_lon_to_180(lon)
    lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise SystemExit("BBox selection produced empty indices; check lat/lon ranges.")
    return lat_idx, lon_idx, lon

def _safe_units(da: xr.DataArray):
    for k in ("units","unit","Units","UNITS","long_units"):
        if k in da.attrs: return str(da.attrs[k])
    return ""

def _long_name(da: xr.DataArray, fallback: str):
    for k in ("long_name","standard_name","name","title","description"):
        if k in da.attrs: return str(da.attrs[k])
    return fallback

def _level_dim_name(da: xr.DataArray):
    for name in da.dims:
        n = name.lower()
        if n in ("isobaricinhpa","isobaricinpa","heightaboveground","depthbelowlandlayer","hybrid","level","lev"):
            if name not in ("time","latitude","longitude","lat","lon"):
                return name
    if "step" in da.dims: return "step"
    return None

def _open_recursive(path: Path, ed: int, base_filters: dict, index_keys: list, depth: int = 0):
    indent = "  " * depth
    filters = dict(base_filters); filters["edition"] = ed
    try:
        ds = cfgrib.open_dataset(str(path), filter_by_keys=filters, backend_kwargs={"indexpath": ""})
        print(f"{indent}[open] edition={ed} filters={filters} -> OK (vars={len(ds.data_vars)})")
        return [ds]
    except DatasetBuildError as e:
        msg = str(e)
        print(f"{indent}[split] {msg} | filters={filters}")
        m = re.search(r"multiple values for key '([^']+)'", msg) or re.search(r"multiple values for key ([A-Za-z0-9_]+)", msg)
        if not m: raise
        key_to_split = m.group(1)

        idx = cfgrib.Index(str(path), index_keys=index_keys, filter_by_keys=filters, errors="ignore")
        values = _unique_vals(idx, key_to_split)
        if not values:
            candidates = ["dataType","typeOfLevel","stepType","gridType","productDefinitionTemplateNumber","table2Version","localDefinitionNumber"]
            if key_to_split not in candidates: candidates.insert(0, key_to_split)
            for k in candidates:
                vals = _unique_vals(idx, k)
                if vals:
                    key_to_split, values = k, vals
                    break
        if not values:
            print(f"{indent}[warn] Could not list uniques for '{key_to_split}'. Re-raising.")
            raise

        out = []
        for v in values:
            sub_filters = dict(filters); sub_filters[key_to_split] = v
            print(f"{indent}[split] -> trying {key_to_split}={v}")
            out.extend(_open_recursive(path, ed, sub_filters, index_keys, depth+1))
        return out

def _open_all_groups(path: Path):
    INDEX_KEYS = ["shortName","typeOfLevel","stepType","gridType","dataType","productDefinitionTemplateNumber"]
    all_dsets = []
    t0 = time.time()
    for edition in (1, 2):
        print(f"[open] Starting edition={edition} …")
        try:
            all_dsets.extend(_open_recursive(path, edition, {}, INDEX_KEYS, depth=1))
            print(f"[open] edition={edition} -> {len(all_dsets)} dataset(s) so far")
        except Exception as e:
            print(f"[open] edition={edition} skipped: {e}")
    if not all_dsets:
        print("[open] Fallback: xr.open_dataset(engine='cfgrib')")
        ds = xr.open_dataset(str(path), engine="cfgrib", backend_kwargs={"indexpath": ""})
        all_dsets = [ds]
    print(f"[open] TOTAL datasets opened: {len(all_dsets)} in {time.time()-t0:,.1f}s")
    return all_dsets

def _build_feature_catalog(dsets):
    feature_specs = []
    lat_ref = lon_ref = None
    lat_sel = lon_sel = None
    time_union = None

    # establish bbox/time from first ds
    for ds in dsets:
        lat_name, lon_name = _coord_names(ds)
        lat_vals = np.asarray(ds[lat_name].values)
        lon_vals = np.asarray(ds[lon_name].values)
        lon_norm = _norm_lon_to_180(lon_vals)
        lat_idx, lon_idx, lon_norm = _bbox_indexers(lat_vals, lon_norm)
        lat_ref, lon_ref = lat_vals, lon_norm
        lat_sel, lon_sel = lat_idx, lon_idx
        if "time" in ds:
            tvals = np.asarray(ds["time"].values)
            time_union = tvals if time_union is None else np.unique(np.concatenate([time_union, tvals]))
        break
    if lat_sel is None: raise SystemExit("No dataset with lat/lon found.")

    # collect features from all groups
    for i, ds in enumerate(dsets):
        lat_name, lon_name = _coord_names(ds)
        if ds[lon_name].values.max() > 180:
            fixed_lon = ((ds[lon_name].values + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: fixed_lon})
        if "time" in ds:
            tvals = np.asarray(ds["time"].values)
            time_union = tvals if time_union is None else np.unique(np.concatenate([time_union, tvals]))

        for var in ds.data_vars:
            da = ds[var]
            ren = {}
            if lat_name != "lat": ren[lat_name] = "lat"
            if lon_name != "lon": ren[lon_name] = "lon"
            da = da.rename(ren)
            if "time" not in da.dims:
                da = da.expand_dims(time=[np.datetime64("1970-01-01")])
            lev_dim = _level_dim_name(da)
            units = _safe_units(da)
            lname = _long_name(da, var)

            if lev_dim and (lev_dim in da.coords or lev_dim in da.dims):
                lev_vals = da[lev_dim].values
                for lv in lev_vals:
                    lv_str = f"{int(lv)}" if np.issubdtype(lev_vals.dtype, np.integer) else (f"{float(lv):g}" if np.issubdtype(lev_vals.dtype, np.floating) else str(lv))
                    feature_specs.append({
                        "ds_i": i, "var": var, "lev_dim": lev_dim, "lev_val": lv,
                        "col_name": f"{var}@{lev_dim}={lv_str}", "units": units,
                        "long_name": f"{lname} ({lev_dim}={lv_str})",
                    })
            else:
                feature_specs.append({
                    "ds_i": i, "var": var, "lev_dim": None, "lev_val": None,
                    "col_name": var, "units": units, "long_name": lname,
                })

    lat_alg = lat_ref[lat_sel]
    lon_alg = lon_ref[lon_sel]

    # verbose catalog
    print(f"[catalog] Algeria bbox lat {LAT_MIN}..{LAT_MAX}  lon {LON_MIN}..{LON_MAX}")
    print(f"[catalog] Algeria grid -> n_lat={lat_alg.size}, n_lon={lon_alg.size}")
    print(f"[catalog] Features discovered: {len(feature_specs)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FEATS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["column_name","dataset_index","var","level_dim","level_val","units","long_name"])
        for spec in feature_specs:
            w.writerow([spec["col_name"], spec["ds_i"], spec["var"],
                        spec["lev_dim"] or "", spec["lev_val"] if spec["lev_dim"] else "",
                        spec["units"], spec["long_name"]])
    print(f"[catalog] Feature catalog written -> {OUT_FEATS}")
    prv = "\n".join(f"  - {spec['col_name']} [{spec['units']}]" for spec in feature_specs[:20])
    print("[catalog] First 20 columns:\n" + (prv if prv else "  (no columns?)"))

    return feature_specs, lat_alg, lon_alg, lat_sel, lon_sel, time_union

def _write_memmap_rows(dsets, feature_specs, lat_sel, lon_sel, time_union, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_feat = len(feature_specs)
    n_time = time_union.size
    n_lat  = lat_sel.size
    n_lon  = lon_sel.size
    n_spatial = n_lat * n_lon
    n_rows = n_time * n_spatial

    X_mm = np.memmap(out_path, mode="w+", dtype=np.float32, shape=(n_rows, n_feat))

    finite_counts = np.zeros(n_feat, dtype=np.int64)
    nan_counts    = np.zeros(n_feat, dtype=np.int64)
    mins          = np.full(n_feat, np.inf, dtype=np.float64)
    maxs          = np.full(n_feat, -np.inf, dtype=np.float64)

    ds_map = {i: ds for i, ds in enumerate(dsets)}
    t0 = time.time()
    print(f"[write] Start streaming: n_time={n_time}, n_spatial={n_spatial}, n_rows={n_rows:,}, n_feat={n_feat}")

    for b, t0m in enumerate(range(0, n_time, TIME_BLOCK), start=1):
        t1m = min(t0m + TIME_BLOCK, n_time)
        row_start = t0m * n_spatial
        row_end   = t1m * n_spatial

        for j, spec in enumerate(feature_specs):
            ds = ds_map[spec["ds_i"]]
            da = ds[spec["var"]]
            lat_name, lon_name = _coord_names(ds)
            ren = {}
            if lat_name != "lat": ren[lat_name] = "lat"
            if lon_name != "lon": ren[lon_name] = "lon"
            if ren: da = da.rename(ren)
            if da["lon"].values.max() > 180:
                da = da.assign_coords(lon=((da["lon"].values + 180) % 360) - 180)
            if "time" not in da.dims:
                da = da.expand_dims(time=[np.datetime64("1970-01-01")])
            da = da.reindex(time=time_union)
            blk = da.isel(time=slice(t0m, t1m), lat=lat_sel, lon=lon_sel)
            if spec["lev_dim"]:
                blk = blk.sel({spec["lev_dim"]: spec["lev_val"]})
            arr = np.array(blk.data, dtype=np.float32).reshape(-1)
            bad_inf = ~np.isfinite(arr) & np.isinf(arr)
            if bad_inf.any(): arr[bad_inf] = np.nan
            X_mm[row_start:row_end, j] = arr

            finite = np.isfinite(arr)
            finite_counts[j] += int(finite.sum())
            nan_counts[j]    += int((~finite).sum())
            if finite.any():
                v = arr[finite].astype(np.float64)
                mins[j] = min(mins[j], float(np.nanmin(v)))
                maxs[j] = max(maxs[j], float(np.nanmax(v)))

        X_mm.flush()
        pct = 100.0 * (row_end / n_rows)
        print(f"[write] Block {b}: rows[{row_start}:{row_end}) time[{t0m}:{t1m})  {pct:5.1f}%  elapsed {time.time()-t0:,.1f}s")

    keep_mask = finite_counts > 0
    print(f"[write] Finished. Columns with any finite data: {int(keep_mask.sum())}/{n_feat}")
    return (n_time, n_lat, n_lon, n_rows, finite_counts, nan_counts, mins, maxs, keep_mask)

def main():
    if not GRIB_PATH.exists():
        raise SystemExit(f"GRIB not found: {GRIB_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] Opening GRIB groups progressively …")
    dsets = _open_all_groups(GRIB_PATH)

    print("[info] Building feature catalog + bbox/time …")
    (feature_specs, lat_alg, lon_alg, lat_sel, lon_sel, time_union) = _build_feature_catalog(dsets)
    n_feat = len(feature_specs)
    if n_feat == 0: raise SystemExit("No features found.")

    print(f"[info] Time steps detected: n_time={time_union.size}")
    years = sorted({int(str(t)[:4]) for t in time_union.astype("datetime64[Y]").astype(int) + 1970})
    if years: print(f"[info] Coverage years: {years[0]}–{years[-1]}")

    print("[info] Streaming write …")
    (n_time, n_lat, n_lon, n_rows,
     finite_counts, nan_counts, mins, maxs, keep_mask) = _write_memmap_rows(
        dsets, feature_specs, lat_sel, lon_sel, time_union, OUT_X
    )

    # metadata arrays
    feature_names = np.array([spec["col_name"] for spec in feature_specs], dtype=object)
    feature_units = np.array([spec["units"] or "" for spec in feature_specs], dtype=object)
    feature_long  = np.array([spec["long_name"] or cn for spec, cn in zip(feature_specs, feature_names)], dtype=object)

    shape_info = {
        "n_time": int(n_time), "n_lat": int(n_lat), "n_lon": int(n_lon),
        "n_features": int(n_feat), "n_instances": int(n_rows),
        "bbox": {"lat_min": float(lat_alg.min()), "lat_max": float(lat_alg.max()),
                 "lon_min": float(lon_alg.min()), "lon_max": float(lon_alg.max())},
        "ordering": "row = t*(n_lat*n_lon) + y*n_lon + x ; columns = feature_names"
    }
    quality = {
        "finite_count": finite_counts.tolist(),
        "nan_count": nan_counts.tolist(),
        "min": np.where(np.isfinite(mins), mins, np.nan).tolist(),
        "max": np.where(np.isfinite(maxs), maxs, np.nan).tolist(),
        "keep_columns_mask": keep_mask.tolist(),
        "dropped_columns_suggestion": np.where(~keep_mask)[0].tolist(),
    }
    np.savez_compressed(
        OUT_META,
        X_path=np.array(str(OUT_X), dtype=object),
        time=time_union.astype("datetime64[ns]"),
        lat=lat_alg, lon=lon_alg,
        feature_names=feature_names,
        feature_units=feature_units,
        feature_long_names=feature_long,
        shape_info=np.array(json.dumps(shape_info), dtype=object),
        quality=json.dumps(quality),
    )

    approx_gb_raw = (n_rows * n_feat * 4) / (1024**3)
    kept = int(np.array(keep_mask).sum())
    md = []
    md.append("# CAMS EAC4 (GRIB) — Algeria-only, ML-ready (Cleaned, No Splits)")
    md.append("")
    md.append(f"- **Source**: `{GRIB_PATH}`")
    md.append(f"- **Matrix (memmap)**: `{OUT_X}`")
    md.append(f"- **Metadata**: `{OUT_META}`")
    md.append(f"- **Rows (instances)**: {n_rows:,}  |  **Features (columns)**: {n_feat:,}  (kept≥1 finite: {kept})")
    if years: md.append(f"- **Time steps**: {n_time} ({years[0]}–{years[-1]})")
    md.append(f"- **Grid (Algeria bbox)**: {n_lat} lat × {n_lon} lon  (lat {LAT_MIN}..{LAT_MAX}, lon {LON_MIN}..{LON_MAX})")
    md.append(f"- **Raw matrix size (float32, uncompressed)**: ~{approx_gb_raw:.2f} GB (stored as memmap)")
    md.append("")
    md.append("## Features")
    md.append("- All GRIB variables included. Level/step dims expanded one column per value (e.g., `t@isobaricInhPa=850`).")
    md.append("- `quality.keep_columns_mask` flags columns with at least one finite value (safe to use).")
    md.append("")
    md.append("## Cleaning")
    md.append("- Replaced ±inf with NaN.")
    md.append("- Per-feature finite counts and min/max recorded (ignoring NaNs).")
    md.append("- Suggested dropped columns (all-NaN) listed in metadata.")
    OUT_CARD.write_text("\n".join(md), encoding="utf-8")

    print("✅ Done:")
    print(f"  - {OUT_X}")
    print(f"  - {OUT_META}")
    print(f"  - {OUT_FEATS}")
    print(f"  - {OUT_CARD}")

if __name__ == "__main__":
    main()
