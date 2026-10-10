#!/usr/bin/env python3
"""
Out-of-core consolidator for CAMS v6.2 (2015–2024), streaming to a memmap.

- Hard-coded paths, species list, and primary var per species (from 2024 cards).
- Manual CF time decoding ("months since YYYY-01-01" / "days since ...").
- Writes:
    ./datasets/CAMS/CAMS_consolidated_v6_2_2015_2024_X.npy      (memmap matrix)
    ./datasets/CAMS/CAMS_consolidated_v6_2_2015_2024_meta.npz   (coords & metadata)
    ./datasets/CAMS/CAMS_consolidated_v6_2_2015_2024.card.md    (summary card)

Memory profile: loads & writes in small time-slices; no 26GB array is materialized.
"""

import os
import re
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

try:
    import xarray as xr
except ImportError as e:
    raise SystemExit("Please install: pip install xarray netCDF4 dask") from e

# ----------------------------- HARD-CODED CONFIG -----------------------------------
ROOT_INVENTORY = Path("./datasets/CAMS/CAMS global emission inventories")
OUT_DIR        = Path("./datasets/CAMS")
VERSION        = "v6.2"
YEARS          = list(range(2015, 2025))  # inclusive

# From 2024 cards (primary sector variable names)
SPECIES_VAR = {
    "ammonia":           "agl",  # NH3 (card)
    "black-carbon":      "awb",  # BC  (card)
    "carbon-monoxide":   "awb",  # CO  (card)
    "isoprene":          "awb",  # Isoprene (card)
    "methane":           "agl",  # CH4 (card)
    "monoterpenes":      "awb",  # Monoterpenes (card)
    "nitrogen-oxides":   "agl",  # NOx (card)
    "organic-carbon":    "awb",  # OC  (card)
    "sulphur-dioxide":   "awb",  # SO2 (card)
}

# Expected grid (from cards)
EXPECTED_LAT_N = 1800
EXPECTED_LON_N = 3600

# Output files
OUT_X_NPY   = OUT_DIR / "CAMS_consolidated_v6_2_2015_2024_X.npy"     # memmap 2D (instances, features)
OUT_META_NPZ= OUT_DIR / "CAMS_consolidated_v6_2_2015_2024_meta.npz"
OUT_CARD    = OUT_DIR / "CAMS_consolidated_v6_2_2015_2024.card.md"

# Write in blocks of T time steps per pass (1..6 is fine; 1 = minimal RAM)
TIME_BLOCK = 1
# -----------------------------------------------------------------------------------

def _parse_base_date(s: str) -> np.datetime64:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return np.datetime64(s, "D")
    if re.fullmatch(r"\d{4}-\d{2}", s):       return np.datetime64(s, "M")
    if re.fullmatch(r"\d{4}", s):             return np.datetime64(s, "Y")
    raise ValueError(f"Unrecognized base date: {s}")

def _decode_time_array(vals, units: str):
    u = (units or "").strip().lower()
    if u.startswith("months since"):
        base = u.replace("months since", "").strip()
        base_dt = _parse_base_date(base)
        if base_dt.dtype == "datetime64[D]":
            base_m = np.datetime64(str(base_dt)[:7], "M")
        elif base_dt.dtype == "datetime64[M]":
            base_m = base_dt
        elif base_dt.dtype == "datetime64[Y]":
            base_m = base_dt.astype("datetime64[M]")
        else:
            base_m = np.datetime64(str(base_dt)[:7], "M")
        months = np.rint(np.asarray(vals, dtype="float64")).astype("int64")
        decoded = (base_m + months.astype("timedelta64[M]")).astype("datetime64[D]").astype("datetime64[ns]")
        return decoded
    if u.startswith("days since"):
        base = u.replace("days since", "").strip()
        base_dt = _parse_base_date(base)
        if base_dt.dtype == "datetime64[M]":
            base_dt = np.datetime64(str(base_dt) + "-01", "D")
        elif base_dt.dtype == "datetime64[Y]":
            base_dt = np.datetime64(str(base_dt) + "-01-01", "D")
        days = np.rint(np.asarray(vals, dtype="float64")).astype("int64")
        decoded = (base_dt + days.astype("timedelta64[D]")).astype("datetime64[ns]")
        return decoded
    return np.asarray(vals)

def open_nc(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="netcdf4", chunks="auto", decode_times=False)
    if "time" in ds:
        try:
            t = ds["time"]
            decoded = _decode_time_array(t.values, str(t.attrs.get("units", "")))
            ds = ds.assign_coords(time=decoded)
        except Exception:
            pass
    return ds

def _standardize_da(da: xr.DataArray, file_year: int) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:  rename["latitude"] = "lat"
    if "longitude" in da.dims: rename["longitude"] = "lon"
    if "y" in da.dims and "lat" not in da.dims: rename["y"] = "lat"
    if "x" in da.dims and "lon" not in da.dims: rename["x"] = "lon"
    da = da.rename(rename)
    if "time" not in da.dims:
        da = da.expand_dims(time=[np.datetime64(f"{file_year}-01-01")])
    da = da.drop_vars([c for c in da.coords if c not in ("time","lat","lon")], errors="ignore")
    if da.sizes.get("lat") != EXPECTED_LAT_N or da.sizes.get("lon") != EXPECTED_LON_N:
        raise ValueError(f"Unexpected grid {da.sizes.get('lat')}×{da.sizes.get('lon')} (expected "
                         f"{EXPECTED_LAT_N}×{EXPECTED_LON_N})")
    return da.astype(np.float32).chunk({"time": 1, "lat": EXPECTED_LAT_N, "lon": EXPECTED_LON_N})

def _collect_species_arrays():
    species_da = {}
    feature_units = {}
    feature_long  = {}
    lat_ref = lon_ref = None
    time_union = None

    for sp, varname in SPECIES_VAR.items():
        arrays = []
        for yr in YEARS:
            fn = ROOT_INVENTORY / f"CAMS-GLOB-ANT_{VERSION}_{sp}_{yr}.nc"
            if not fn.exists():
                continue
            ds = open_nc(fn)
            if varname not in ds:
                ds.close()
                raise KeyError(f"{fn.name}: variable '{varname}' not found.")
            da = _standardize_da(ds[varname], yr)
            # units/long_name
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
            arrays.append(da)
            if lat_ref is None:
                lat_ref = np.asarray(da["lat"].values)
                lon_ref = np.asarray(da["lon"].values)
                time_union = np.asarray(da["time"].values)
            else:
                time_union = np.unique(np.concatenate([time_union, np.asarray(da["time"].values)]))
            ds.close()

        if arrays:
            species_da[sp] = xr.concat(arrays, dim="time")
        else:
            species_da[sp] = None  # will fill with NaNs later

    # Reindex all species to the union time axis
    for sp in SPECIES_VAR.keys():
        if species_da[sp] is None:
            continue
        species_da[sp] = species_da[sp].reindex(time=time_union)

    return species_da, feature_units, feature_long, lat_ref, lon_ref, time_union

def main():
    if not ROOT_INVENTORY.exists():
        raise SystemExit(f"Inventory folder not found: {ROOT_INVENTORY}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    species_da, feature_units, feature_long, lat_ref, lon_ref, time_all = _collect_species_arrays()
    species_order = list(SPECIES_VAR.keys())

    n_time = time_all.size
    n_lat  = lat_ref.size
    n_lon  = lon_ref.size
    n_feat = len(species_order)
    n_spatial = n_lat * n_lon
    n_instances = n_time * n_spatial

    # Create the memmap file for X
    X_mm = np.memmap(OUT_X_NPY, mode="w+", dtype=np.float32, shape=(n_instances, n_feat))

    # Stream write by time blocks
    for t0 in range(0, n_time, TIME_BLOCK):
        t1 = min(t0 + TIME_BLOCK, n_time)
        row_start = t0 * n_spatial
        row_end   = t1 * n_spatial

        # For each species, fetch the block lazily and write one column
        for j, sp in enumerate(species_order):
            if species_da[sp] is None:
                # Fill with NaN if species had no data for any year
                X_mm[row_start:row_end, j] = np.nan
                continue
            block = species_da[sp].isel(time=slice(t0, t1)).data  # dask-backed
            block_np = np.array(block)  # compute this small block only
            # reshape (T, lat, lon) -> (T*lat*lon,)
            X_mm[row_start:row_end, j] = block_np.reshape(-1)

        # Optional: flush to ensure data hits disk
        X_mm.flush()
        print(f"Wrote time rows [{t0}:{t1}) -> [{row_start}:{row_end})")

    # Save metadata (small)
    feature_names = np.array(species_order, dtype=object)
    feature_units_arr = np.array([feature_units.get(sp, "kg m-2 s-1") for sp in species_order], dtype=object)
    feature_long_arr  = np.array([feature_long.get(sp, sp) for sp in species_order], dtype=object)

    # index maps (kept lightweight by generating on the fly in consumers; here we store shapes only)
    shape_info = {
        "n_time": int(n_time), "n_lat": int(n_lat), "n_lon": int(n_lon),
        "n_features": int(n_feat), "n_instances": int(n_instances),
        "grid": {
            "lat_min": float(np.min(lat_ref)), "lat_max": float(np.max(lat_ref)),
            "lon_min": float(np.min(lon_ref)), "lon_max": float(np.max(lon_ref)),
        },
    }

    np.savez_compressed(
        OUT_META_NPZ,
        X_path=np.array(str(OUT_X_NPY), dtype=object),
        lat=lat_ref, lon=lon_ref, time=time_all.astype("datetime64[ns]"),
        feature_names=feature_names,
        feature_units=feature_units_arr,
        feature_long_names=feature_long_arr,
        shape_info=np.array(json.dumps(shape_info), dtype=object),
        version=np.array(VERSION, dtype=object),
    )

    # Card
    approx_gb = (n_instances * n_feat * 4) / (1024**3)  # raw float32 size (not memmap)
    years = sorted({int(str(t)[:4]) for t in time_all.astype("datetime64[Y]").astype(int) + 1970})
    md = []
    md.append(f"# CAMS {VERSION} — Consolidated (memmap)")
    md.append("")
    md.append(f"- **Matrix file**: `{OUT_X_NPY}` (NumPy memmap, float32)")
    md.append(f"- **Metadata**: `{OUT_META_NPZ}`")
    md.append(f"- **Instances (rows)**: {n_instances:,}")
    md.append(f"- **Features (columns)**: {n_feat} ({', '.join(species_order)})")
    md.append(f"- **Time steps**: {n_time} ({years[0]}–{years[-1]})")
    md.append(f"- **Grid**: {n_lat} lat × {n_lon} lon (~0.1°)")
    md.append(f"- **Estimated raw size**: ~{approx_gb:.2f} GB (uncompressed); stored as on-disk memmap.")
    md.append("")
    md.append("## Notes")
    md.append("- Data written in 1-step time blocks to keep RAM low; adjust TIME_BLOCK if you have headroom.")
    md.append("- Use np.memmap to read without loading entire matrix into memory.")
    md.append("- Coordinates and names/units are in the meta NPZ.")
    OUT_CARD.write_text("\n".join(md), encoding="utf-8")

    print("✅ Wrote:")
    print(f"  - {OUT_X_NPY}")
    print(f"  - {OUT_META_NPZ}")
    print(f"  - {OUT_CARD}")

if __name__ == "__main__":
    main()
