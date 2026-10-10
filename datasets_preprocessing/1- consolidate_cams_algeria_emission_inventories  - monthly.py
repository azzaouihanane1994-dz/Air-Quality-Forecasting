#!/usr/bin/env python3
"""
CAMS v6.2 (2015–2024) -> Algeria-only, ML-ready, out-of-core consolidator.

Writes:
  - ./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_X.npy        (NumPy memmap, rows x features)
  - ./datasets/CAMS/CAMS_ALG_v6_2_2015_2024_meta.npz      (coords, names, scalers, splits, shapes)
  - ./datasets/CAMS/CAMS_ALG_v6_2_2015_2024.card.md       (summary)

Notes:
  - Uses per-species primary variables from your 2024 cards (agl/awb).
  - Manual CF time decoding ("months since ...").
  - Crops to Algeria bbox: lat 18.0..37.5, lon -9.0..12.0 (degrees).
  - Keeps everything float32. Splits are by rows after full shuffle (seeded).
"""

import re, json
from pathlib import Path
import numpy as np

try:
    import xarray as xr
except ImportError as e:
    raise SystemExit("Please install: pip install xarray netCDF4 dask") from e

# ---------------------------- HARD-CODED CONFIG ------------------------------------
ROOT = Path("./datasets/CAMS/CAMS global emission inventories")
OUT_DIR = Path("./datasets/CAMS")
VER = "v6.2"
YEARS = list(range(2015, 2025))  # inclusive

# Primary variable per species (from your 2024 cards)
SPECIES_VAR = {
    "ammonia":           "agl",
    "black-carbon":      "awb",
    "carbon-monoxide":   "awb",
    "isoprene":          "awb",
    "methane":           "agl",
    "monoterpenes":      "awb",
    "nitrogen-oxides":   "agl",
    "organic-carbon":    "awb",
    "sulphur-dioxide":   "awb",
}

# Expected global grid (from cards): time=12, lat=1800, lon=3600, lon in -180..180
EXPECTED_LAT_N = 1800
EXPECTED_LON_N = 3600

# Algeria bounding box (deg)
ALG_LAT_MIN, ALG_LAT_MAX = 18.0, 37.5
ALG_LON_MIN, ALG_LON_MAX = -9.0, 12.0

# Output files
OUT_X = OUT_DIR / "CAMS_ALG_v6_2_2015_2024_X.npy"
OUT_META = OUT_DIR / "CAMS_ALG_v6_2_2015_2024_meta.npz"
OUT_CARD = OUT_DIR / "CAMS_ALG_v6_2_2015_2024.card.md"

# Dataloader / split knobs
TIME_BLOCK = 1           # write 1 month at a time (RAM-safe)
RNG_SEED   = 42
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train/val/test
# -----------------------------------------------------------------------------------

def _parse_base_date(s: str) -> np.datetime64:
    import re
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return np.datetime64(s, "D")
    if re.fullmatch(r"\d{4}-\d{2}", s):       return np.datetime64(s, "M")
    if re.fullmatch(r"\d{4}", s):             return np.datetime64(s, "Y")
    raise ValueError(f"Unrecognized base date: {s}")

def _decode_time(vals, units: str):
    u = (units or "").strip().lower()
    if u.startswith("months since"):
        base = u.replace("months since", "").strip()
        base_dt = _parse_base_date(base)
        if base_dt.dtype == "datetime64[D]":
            base_m = np.datetime64(str(base_dt)[:7], "M")
        elif base_dt.dtype == "datetime64[M]":
            base_m = base_dt
        else:
            base_m = base_dt.astype("datetime64[M]")
        months = np.rint(np.asarray(vals, dtype="float64")).astype("int64")
        return (base_m + months.astype("timedelta64[M]")).astype("datetime64[D]").astype("datetime64[ns]")
    if u.startswith("days since"):
        base = u.replace("days since", "").strip()
        base_dt = _parse_base_date(base)
        if base_dt.dtype == "datetime64[M]":
            base_dt = np.datetime64(str(base_dt) + "-01", "D")
        elif base_dt.dtype == "datetime64[Y]":
            base_dt = np.datetime64(str(base_dt) + "-01-01", "D")
        days = np.rint(np.asarray(vals, dtype="float64")).astype("int64")
        return (base_dt + days.astype("timedelta64[D]")).astype("datetime64[ns]")
    return np.asarray(vals)

def _open_nc(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="netcdf4", chunks="auto", decode_times=False)
    if "time" in ds:
        try:
            t = ds["time"]
            ds = ds.assign_coords(time=_decode_time(t.values, str(t.attrs.get("units", ""))))
        except Exception:
            pass
    return ds

def _std_da(da: xr.DataArray, file_year: int) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:  rename["latitude"] = "lat"
    if "longitude" in da.dims: rename["longitude"] = "lon"
    if "y" in da.dims and "lat" not in da.dims: rename["y"] = "lat"
    if "x" in da.dims and "lon" not in da.dims: rename["x"] = "lon"
    da = da.rename(rename)
    if "time" not in da.dims:
        da = da.expand_dims(time=[np.datetime64(f"{file_year}-01-01")])
    da = da.drop_vars([c for c in da.coords if c not in ("time","lat","lon")], errors="ignore")
    # Validate grid
    if da.sizes.get("lat") != EXPECTED_LAT_N or da.sizes.get("lon") != EXPECTED_LON_N:
        raise ValueError(f"Unexpected grid {da.sizes.get('lat')}x{da.sizes.get('lon')} (expected {EXPECTED_LAT_N}x{EXPECTED_LON_N})")
    return da.astype(np.float32).chunk({"time": 1, "lat": EXPECTED_LAT_N, "lon": EXPECTED_LON_N})

def _subset_bbox(lat: np.ndarray, lon: np.ndarray):
    # lon is in [-180, 180] per cards → no 0..360 wrap needed.
    lat_idx = np.where((lat >= ALG_LAT_MIN) & (lat <= ALG_LAT_MAX))[0]
    lon_idx = np.where((lon >= ALG_LON_MIN) & (lon <= ALG_LON_MAX))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise SystemExit("BBox produced empty selection. Check lat/lon arrays.")
    return lat_idx, lon_idx

def _collect_species_bbox():
    species_da = {}
    feature_units, feature_long = {}, {}
    lat_ref = lon_ref = None

    for sp, varname in SPECIES_VAR.items():
        per_year = []
        for yr in YEARS:
            fn = ROOT / f"CAMS-GLOB-ANT_{VER}_{sp}_{yr}.nc"
            if not fn.exists():
                continue
            ds = _open_nc(fn)
            if varname not in ds:
                ds.close()
                raise KeyError(f"{fn.name}: variable '{varname}' not found.")
            da = _std_da(ds[varname], yr)

            # ref coords + bbox indexers
            if lat_ref is None:
                lat_ref = np.asarray(da["lat"].values)
                lon_ref = np.asarray(da["lon"].values)
                lat_sel, lon_sel = _subset_bbox(lat_ref, lon_ref)

            # subset to bbox (Algeria)
            da = da.isel(lat=lat_sel, lon=lon_sel)

            # capture attrs
            if sp not in feature_units:
                u = ""
                for k in ("units","unit","Units","UNITS","long_units"):
                    if k in da.attrs: u = str(da.attrs[k]); break
                feature_units[sp] = u or "kg m-2 s-1"
            if sp not in feature_long:
                ln = ""
                for k in ("long_name","standard_name","description","title","name"):
                    if k in da.attrs: ln = str(da.attrs[k]); break
                feature_long[sp] = ln or sp

            per_year.append(da)
            ds.close()

        species_da[sp] = xr.concat(per_year, dim="time") if per_year else None

    # Final lat/lon for Algeria region:
    lat_alg = lat_ref[lat_sel]
    lon_alg = lon_ref[lon_sel]

    # Build union time across species
    time_union = None
    for sp, da in species_da.items():
        if da is None: continue
        tvals = np.asarray(da["time"].values)
        time_union = tvals if time_union is None else np.unique(np.concatenate([time_union, tvals]))

    # Reindex to union time
    for sp, da in species_da.items():
        if da is None: continue
        species_da[sp] = da.reindex(time=time_union)

    return species_da, feature_units, feature_long, lat_alg, lon_alg, time_union

def _compute_scaler_train(X_memmap, train_idx, n_feat):
    # mean/std over training rows, feature-wise (streamed in chunks)
    means = np.zeros(n_feat, dtype=np.float64)
    sqs   = np.zeros(n_feat, dtype=np.float64)
    counts= np.zeros(n_feat, dtype=np.int64)

    # chunk over training rows to limit RAM
    CHUNK = 200_000
    for i0 in range(0, train_idx.size, CHUNK):
        i1 = min(i0 + CHUNK, train_idx.size)
        block = X_memmap[train_idx[i0:i1], :].astype(np.float64)
        mask = np.isfinite(block)
        counts += mask.sum(axis=0)
        block[~mask] = 0.0
        means += block.sum(axis=0)
        sqs   += (block**2).sum(axis=0)

    means = means / np.maximum(counts, 1)
    vars_ = np.maximum(sqs / np.maximum(counts,1) - means**2, 0.0)
    stds  = np.sqrt(vars_)
    stds[stds == 0.0] = 1.0
    return means.astype(np.float32), stds.astype(np.float32), counts

def main():
    if not ROOT.exists():
        raise SystemExit(f"Missing input dir: {ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    species_da, feat_units, feat_long, lat_alg, lon_alg, time_all = _collect_species_bbox()
    species_order = list(SPECIES_VAR.keys())

    n_time = time_all.size
    n_lat  = lat_alg.size
    n_lon  = lon_alg.size
    n_feat = len(species_order)
    n_spatial = n_lat * n_lon
    n_rows = n_time * n_spatial

    # Create Algeria-only memmap matrix
    X_mm = np.memmap(OUT_X, mode="w+", dtype=np.float32, shape=(n_rows, n_feat))

    # Stream-fill by time
    for t0 in range(0, n_time, TIME_BLOCK):
        t1 = min(t0 + TIME_BLOCK, n_time)
        row_start = t0 * n_spatial
        row_end   = t1 * n_spatial
        for j, sp in enumerate(species_order):
            da = species_da[sp]
            if da is None:
                X_mm[row_start:row_end, j] = np.nan
                continue
            block = da.isel(time=slice(t0, t1)).data
            arr = np.array(block)  # (T, lat_alg, lon_alg)
            X_mm[row_start:row_end, j] = arr.reshape(-1)
        X_mm.flush()
        print(f"Wrote rows [{row_start}:{row_end})  time[{t0}:{t1})")

    # ---- ML helpers: splits + scalers (feature-wise mean/std on train) -------------
    rng = np.random.default_rng(RNG_SEED)
    all_idx = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(all_idx)
    n_train = int(SPLIT_RATIOS[0] * n_rows)
    n_val   = int(SPLIT_RATIOS[1] * n_rows)
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:n_train+n_val]
    test_idx  = all_idx[n_train+n_val:]

    means, stds, counts = _compute_scaler_train(X_mm, train_idx, n_feat)

    # Save compact metadata
    feature_names = np.array(species_order, dtype=object)
    feature_units = np.array([feat_units.get(sp, "kg m-2 s-1") for sp in species_order], dtype=object)
    feature_longs = np.array([feat_long.get(sp, sp) for sp in species_order], dtype=object)

    shape = {
        "n_time": int(n_time), "n_lat": int(n_lat), "n_lon": int(n_lon),
        "n_features": int(n_feat), "n_instances": int(n_rows),
        "bbox": {"lat_min": float(lat_alg.min()), "lat_max": float(lat_alg.max()),
                 "lon_min": float(lon_alg.min()), "lon_max": float(lon_alg.max())},
        "ordering": "row = t * (n_lat*n_lon) + y * n_lon + x ; columns = feature_names"
    }

    np.savez_compressed(
        OUT_META,
        X_path=np.array(str(OUT_X), dtype=object),
        version=np.array(VER, dtype=object),
        time=time_all.astype("datetime64[ns]"),
        lat=lat_alg, lon=lon_alg,
        feature_names=feature_names,
        feature_units=feature_units,
        feature_long_names=feature_longs,
        shape_info=np.array(json.dumps(shape), dtype=object),
        # ML extras:
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        feature_mean=means, feature_std=stds, feature_count=counts,
        rng_seed=np.array(RNG_SEED, dtype=np.int64),
    )

    # Card
    years = sorted({int(str(t)[:4]) for t in time_all.astype("datetime64[Y]").astype(int) + 1970})
    approx_gb_uncompressed = (n_rows * n_feat * 4) / (1024**3)
    md = []
    md.append(f"# CAMS {VER} — Algeria-only Consolidated (ML-ready)")
    md.append("")
    md.append(f"- **Matrix**: `{OUT_X}` (NumPy memmap, float32)")
    md.append(f"- **Metadata**: `{OUT_META}`")
    md.append(f"- **Rows (instances)**: {n_rows:,}  |  **Features**: {n_feat} ({', '.join(species_order)})")
    md.append(f"- **Time steps**: {n_time} ({years[0]}–{years[-1]})")
    md.append(f"- **Grid (Algeria bbox)**: {n_lat} lat × {n_lon} lon  (lat {ALG_LAT_MIN}..{ALG_LAT_MAX}, lon {ALG_LON_MIN}..{ALG_LON_MAX})")
    md.append(f"- **Raw matrix size (uncompressed)**: ~{approx_gb_uncompressed:.2f} GB (for reference; stored as memmap)")
    md.append("")
    md.append("## ML Readiness")
    md.append("- `feature_mean`/`feature_std` computed on training rows only (NaNs ignored).")
    md.append("- `train_idx`, `val_idx`, `test_idx` provided (shuffled with fixed RNG seed).")
    md.append("- Row index mapping: `row = t*(n_lat*n_lon) + y*n_lon + x`.")
    OUT_CARD.write_text("\n".join(md), encoding="utf-8")

    print("✅ Done:")
    print(f"  - {OUT_X}")
    print(f"  - {OUT_META}")
    print(f"  - {OUT_CARD}")

if __name__ == "__main__":
    main()
