#!/usr/bin/env python3
"""
Generate per-file dataset cards (Markdown) for CAMS 2024 NetCDF files,
handling CF time like "months since 2024-01-01" without requiring cftime.

Requires:
  pip install xarray netCDF4 dask  (cftime optional)

Usage:
  python make_cards_cams_2024.py
"""

import os
import re
import sys
import json
import math
import datetime as dt
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np

try:
    import xarray as xr
except ImportError as e:
    raise SystemExit("This script requires xarray. Install with: pip install xarray netCDF4 dask") from e

# ---- CONFIG -----------------------------------------------------------------------
ROOT = Path("./datasets/CAMS/CAMS global emission inventories")
FILENAMES = [
    "CAMS-GLOB-ANT_v6.2_ammonia_2024.nc",
    "CAMS-GLOB-ANT_v6.2_black-carbon_2024.nc",
    "CAMS-GLOB-ANT_v6.2_carbon-monoxide_2024.nc",
    "CAMS-GLOB-ANT_v6.2_isoprene_2024.nc",
    "CAMS-GLOB-ANT_v6.2_methane_2024.nc",
    "CAMS-GLOB-ANT_v6.2_monoterpenes_2024.nc",
    "CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2024.nc",
    "CAMS-GLOB-ANT_v6.2_organic-carbon_2024.nc",
    "CAMS-GLOB-ANT_v6.2_sulphur-dioxide_2024.nc",
]
# -----------------------------------------------------------------------------------

LIKELY_COORDS = {"lat", "latitude", "lon", "longitude", "time", "month", "year", "x", "y"}

# --- helpers -----------------------------------------------------------------------

def pick_primary_var(ds: xr.Dataset) -> str:
    if not ds.data_vars:
        raise ValueError("No data variables in dataset.")
    candidates = [v for v in ds.data_vars if v.lower() not in LIKELY_COORDS] or list(ds.data_vars)
    def score(v):
        dims = set(ds[v].dims)
        s = 0
        if {"lat","latitude"} & dims: s += 2
        if {"lon","longitude"} & dims: s += 2
        if "time" in dims or "year" in dims or "month" in dims: s += 1
        s += int(ds[v].size / 1e6)
        return s
    return max(candidates, key=score)


def coord_name_alias(name: str) -> str:
    m = name.lower()
    if m in ("latitude", "y"): return "lat"
    if m in ("longitude", "long", "x"): return "lon"
    return name


def _parse_base_date(s: str) -> np.datetime64:
    # Accept YYYY-MM-DD or YYYY-MM or YYYY
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return np.datetime64(s, "D")
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return np.datetime64(s, "M")
    if re.fullmatch(r"\d{4}", s):
        return np.datetime64(s, "Y")
    raise ValueError(f"Unrecognized base date: {s}")


def _to_int_months(x: np.ndarray) -> np.ndarray:
    # CF time sometimes stores months as float; round safely
    return np.rint(np.asarray(x, dtype="float64")).astype("int64")


def _to_int_days(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype="float64")).astype("int64")


def decode_time_coord_maybe(ds: xr.Dataset) -> xr.Dataset:
    """
    If 'time' is present and undecoded, try to decode manually.
    - Handles 'months since YYYY[-MM[-DD]]'
    - Handles 'days since YYYY[-MM[-DD]]'
    Leaves dataset unchanged if decoding isn't needed or fails.
    """
    if "time" not in ds:
        return ds

    tvar = ds["time"]
    # Already datetime-like?
    if np.issubdtype(tvar.dtype, np.datetime64):
        return ds

    units = str(tvar.attrs.get("units", "")).strip().lower()
    cal = str(tvar.attrs.get("calendar", "")).strip().lower() or "standard"

    # Try cftime/xarray decode_cf first if installed
    try:
        import cftime  # noqa: F401
        try:
            ds_decoded = xr.decode_cf(ds, use_cftime=False)  # returns numpy datetime64 when possible
            if "time" in ds_decoded and np.issubdtype(ds_decoded["time"].dtype, np.datetime64):
                return ds_decoded
        except Exception:
            pass
    except Exception:
        pass

    vals = np.asarray(tvar.values)

    try:
        if units.startswith("months since"):
            base = units.replace("months since", "").strip()
            base_dt = _parse_base_date(base)
            # Convert to month-precision then to days (first day of month)
            if base_dt.dtype == "datetime64[D]":
                # drop to month precision (YYYY-MM) to add month deltas correctly
                base_month = np.datetime64(str(base_dt)[:7], "M")
            elif base_dt.dtype == "datetime64[M]":
                base_month = base_dt
            elif base_dt.dtype == "datetime64[Y]":
                base_month = base_dt.astype("datetime64[M]")
            else:
                base_month = np.datetime64(str(base_dt)[:7], "M")

            months = _to_int_months(vals)
            decoded_m = base_month + months.astype("timedelta64[M]")
            decoded_d = decoded_m.astype("datetime64[D]")  # first day of each month
            ds = ds.assign_coords(time=decoded_d.astype("datetime64[ns]"))
            return ds

        if units.startswith("days since"):
            base = units.replace("days since", "").strip()
            base_dt = _parse_base_date(base)
            # Ensure day precision
            if base_dt.dtype == "datetime64[M]":
                # assume day 01
                base_dt = np.datetime64(str(base_dt) + "-01", "D")
            elif base_dt.dtype == "datetime64[Y]":
                base_dt = np.datetime64(str(base_dt) + "-01-01", "D")
            days = _to_int_days(vals)
            decoded_d = base_dt + days.astype("timedelta64[D]")
            ds = ds.assign_coords(time=decoded_d.astype("datetime64[ns]"))
            return ds
    except Exception:
        # fall through to leave ds as-is
        pass

    # If not decodable, leave as numeric
    return ds


def summarize_coords(da: xr.DataArray) -> Dict[str, Any]:
    info = {}
    for c in da.coords:
        arr = da[c]
        cname = coord_name_alias(c)
        try:
            vals = np.asarray(arr.values)
        except Exception:
            continue
        out = {"dim": c, "size": int(arr.size)}
        if np.issubdtype(vals.dtype, np.number):
            out["min"] = float(np.nanmin(vals))
            out["max"] = float(np.nanmax(vals))
            if arr.size > 1:
                diffs = np.diff(vals)
                out["approx_step"] = float(np.nanmedian(diffs)) if diffs.size else None
        elif np.issubdtype(vals.dtype, np.datetime64):
            out["min"] = str(vals.min())
            out["max"] = str(vals.max())
            out["count"] = int(vals.size)
        else:
            out["example_values"] = [str(vals[i]) for i in np.linspace(0, arr.size-1, num=min(5, arr.size), dtype=int)]
        info[cname] = out
    return info


def qsafe(x: np.ndarray, q: float) -> float:
    try:
        return float(np.nanpercentile(x, q))
    except Exception:
        return float("nan")


def summarize_variable(da: xr.DataArray) -> Dict[str, Any]:
    a = da.astype("float64").values  # dask-aware
    stats = {"count_total": int(np.prod(da.shape))}
    finite_mask = np.isfinite(a)
    stats["count_finite"] = int(finite_mask.sum())
    if stats["count_finite"] == 0:
        for k in ("mean","std","min","max","p01","p05","p50","p95","p99"):
            stats[k] = float("nan")
    else:
        v = a[finite_mask]
        stats.update(
            mean=float(v.mean()),
            std=float(v.std()),
            min=float(v.min()),
            max=float(v.max()),
            p01=qsafe(v, 1),
            p05=qsafe(v, 5),
            p50=qsafe(v, 50),
            p95=qsafe(v, 95),
            p99=qsafe(v, 99),
        )
    return stats


def units_of(da: xr.DataArray) -> str:
    for k in ("units","unit","Units","UNITS","long_units"):
        if k in da.attrs:
            return str(da.attrs[k])
    return ""


def long_name_of(da: xr.DataArray) -> str:
    for k in ("long_name","standard_name","description","title","name"):
        if k in da.attrs:
            return str(da.attrs[k])
    return ""


def time_coverage(da: xr.DataArray) -> Tuple[str, str, int]:
    if "time" in da.coords:
        t = np.asarray(da["time"].values)
        if t.size:
            return str(t.min()), str(t.max()), int(t.size)
    if "year" in da.coords:
        y = np.asarray(da["year"].values).astype(int)
        if y.size:
            return f"{y.min()}-01-01", f"{y.max()}-12-31", int(y.size)
    return "", "", 0


def card_markdown(path: Path, ds: xr.Dataset, varname: str, da: xr.DataArray,
                  coord_info: Dict[str, Any], var_stats: Dict[str, Any]) -> str:
    try:
        st = path.stat()
        size_mb = st.st_size / (1024**2)
        mtime = dt.datetime.fromtimestamp(st.st_mtime).isoformat()
    except Exception:
        size_mb = float("nan")
        mtime = "N/A"

    gattrs = {k: str(v) for k, v in ds.attrs.items()} if isinstance(ds.attrs, dict) else {}
    gattrs_json = json.dumps(gattrs, indent=2) if gattrs else "{}"

    v_units = units_of(da) or "(unit not provided)"
    v_long  = long_name_of(da) or varname
    tmin, tmax, tcount = time_coverage(da)

    dims_str = " × ".join([f"{d}={da.sizes[d]}" for d in da.dims])
    grid_str = ""
    if "lat" in coord_info and "lon" in coord_info:
        lat = coord_info["lat"]; lon = coord_info["lon"]
        grid_str = f"{lat.get('size','?')} lat × {lon.get('size','?')} lon"
        if "approx_step" in lat and "approx_step" in lon:
            grid_str += f" (Δlat≈{lat['approx_step']}, Δlon≈{lon['approx_step']})"

    md = []
    md.append(f"# Data Card — {path.name}")
    md.append("")
    md.append(f"**Location**: `{path}`  ")
    md.append(f"**File size**: {size_mb:.2f} MB  ")
    md.append(f"**Last modified**: {mtime}  ")
    md.append("")
    md.append("## Primary Variable")
    md.append(f"- **Name**: `{varname}`")
    md.append(f"- **Long name**: {v_long}")
    md.append(f"- **Units**: {v_units}")
    md.append(f"- **Dimensions**: {dims_str}")
    if grid_str:
        md.append(f"- **Grid**: {grid_str}")
    if tcount:
        md.append(f"- **Time coverage**: {tmin} → {tmax}  (steps: {tcount})")
    md.append("")
    md.append("## Coordinates")
    for cname in sorted(coord_info.keys()):
        ci = coord_info[cname]
        line = f"- **{cname}**: size={ci.get('size','?')}"
        if "min" in ci and "max" in ci:
            line += f", range=[{ci['min']}, {ci['max']}]"
        if "approx_step" in ci and ci["approx_step"] is not None:
            line += f", step≈{ci['approx_step']}"
        if "count" in ci:
            line += f", count={ci['count']}"
        if "example_values" in ci:
            ex = ", ".join(ci["example_values"])
            line += f", examples=({ex})"
        md.append(line)
    md.append("")
    md.append("## Summary Statistics (primary variable)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| total elements | {var_stats['count_total']:,} |")
    md.append(f"| finite elements | {var_stats['count_finite']:,} |")
    md.append(f"| mean | {var_stats['mean']:.6g} |")
    md.append(f"| std | {var_stats['std']:.6g} |")
    md.append(f"| min | {var_stats['min']:.6g} |")
    md.append(f"| p01 | {var_stats['p01']:.6g} |")
    md.append(f"| p05 | {var_stats['p05']:.6g} |")
    md.append(f"| p50 | {var_stats['p50']:.6g} |")
    md.append(f"| p95 | {var_stats['p95']:.6g} |")
    md.append(f"| p99 | {var_stats['p99']:.6g} |")
    md.append(f"| max | {var_stats['max']:.6g} |")
    md.append("")
    md.append("## Global Attributes")
    md.append("")
    md.append("```json")
    md.append(gattrs_json)
    md.append("```")
    md.append("")
    md.append("## Notes")
    md.append("- Opened with decode_times=False, then decoded CF time manually.")
    md.append("- Primary variable chosen heuristically as the largest lat/lon (+time) field.")
    md.append("- Stats ignore NaNs and infinities.")
    return "\n".join(md)


def make_card_for_file(nc_path: Path, engine="netcdf4", chunks="auto") -> Path:
    if not nc_path.exists():
        raise FileNotFoundError(str(nc_path))

    # Open with decode_times=False to avoid CF decoding errors
    ds = xr.open_dataset(nc_path, engine=engine, chunks=chunks, decode_times=False)

    # Try to decode 'time' coordinate manually
    ds = decode_time_coord_maybe(ds)

    varname = pick_primary_var(ds)
    da = ds[varname]

    coord_info = summarize_coords(da)
    stats = summarize_variable(da)

    md = card_markdown(nc_path, ds, varname, da, coord_info, stats)

    out_path = nc_path.with_suffix(nc_path.suffix + ".card.md")  # e.g., .nc.card.md
    out_path.write_text(md, encoding="utf-8")

    ds.close()
    return out_path


def main():
    root = ROOT
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    written = []
    for fn in FILENAMES:
        nc_path = root / fn
        try:
            out = make_card_for_file(nc_path)
            print(f"✅ Wrote: {out}")
            written.append(out)
        except Exception as e:
            print(f"❌ Failed for {nc_path}: {e}", file=sys.stderr)

    print(f"\nDone. Generated {len(written)} card(s).")


if __name__ == "__main__":
    main()
