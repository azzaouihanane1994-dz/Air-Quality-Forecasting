#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-prepare_nasa_fire_labels_27x29_aligned.py
(see chat for full header)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CAMS_DIR     = Path("./datasets/CAMS")
NASA_DIR     = Path("./datasets/nasa_fires")
OUT_PREFIX   = "FIRE_ALG_27x29"

MERGED_META  = CAMS_DIR / "merged_27x29_meta.npz"
MERGED_4D    = CAMS_DIR / "merged_27x29_4d.npy"

OUT_4D       = CAMS_DIR / f"{OUT_PREFIX}_4d.npy"
OUT_FLAT     = CAMS_DIR / f"{OUT_PREFIX}_flat.npy"
OUT_META     = CAMS_DIR / f"{OUT_PREFIX}_meta.npz"
OUT_CARD     = CAMS_DIR / f"{OUT_PREFIX}.card.md"

COL_LAT  = "latitude"; COL_LON  = "longitude"
COL_DATE = "acq_date"; COL_TIME = "acq_time"
COL_FRP  = "frp";      COL_CONF = "confidence"; COL_TYPE = "type"

ALLOWED_TYPES      = {0, 2}
MIN_CONF_NUMERIC   = 80
MIN_CONF_SCORE_CAT = 50
MIN_FRP            = 1.0

PERSIST_FREQ      = 0.30
PERSIST_MIN_YEARS = 3

FEATURES = np.array(["fire_label", "fire_frp"], dtype=object)

def _parse_confidence(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.lower().str.strip()
    mapping = {"l": 0, "low": 0, "n": 50, "nominal": 50, "normal": 50, "h": 100, "high": 100}
    return s.map(mapping).fillna(0).astype(float)

def _to_timestamp(date, time_str):
    dt = pd.to_datetime(date)
    t = pd.Series(time_str).astype(str).str.zfill(4)
    hh = t.str[:2].astype(int); mm = t.str[2:].astype(int)
    return dt + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")

def load_firms_filtered(nasa_dir: Path) -> pd.DataFrame:
    csvs = sorted(nasa_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {nasa_dir}")
    dfs = []; total_raw = 0; total_kept = 0
    print(f"[FIRMS] scanning {len(csvs)} CSVs in {nasa_dir} ...")
    for p in csvs:
        df = pd.read_csv(p); n0 = len(df)
        missing = [c for c in [COL_LAT, COL_LON, COL_DATE, COL_TIME, COL_FRP] if c not in df.columns]
        if missing:
            print(f"  ⚠️ {p.name}: missing {missing}, skip"); continue
        if COL_TYPE in df.columns:
            before = len(df); df = df[df[COL_TYPE].isin({0,2})]
            print(f"  {p.name}: type filter kept {len(df)}/{before}")
        if COL_CONF in df.columns:
            conf = _parse_confidence(df[COL_CONF])
            if pd.api.types.is_numeric_dtype(df[COL_CONF]):
                df = df[conf >= MIN_CONF_NUMERIC]
            else:
                df = df[conf >= MIN_CONF_SCORE_CAT]
        df = df[df[COL_FRP] >= MIN_FRP]
        df["time"] = _to_timestamp(df[COL_DATE], df[COL_TIME])
        df = df[["time", COL_LAT, COL_LON, COL_FRP]].rename(columns={COL_LAT:"lat", COL_LON:"lon", COL_FRP:"frp"})
        total_raw += n0; total_kept += len(df)
        print(f"  {p.name}: raw={n0}, kept={len(df)} ({len(df)/max(n0,1):.3f})")
        if not df.empty: dfs.append(df)
    if not dfs: raise RuntimeError("No valid fire detections after filtering.")
    fires = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"[FIRMS] total kept {total_kept} (from {total_raw} raw rows)")
    return fires

def main():
    meta = np.load(MERGED_META, allow_pickle=True)
    lat  = meta["lat"].astype(float); lon = meta["lon"].astype(float)
    days = pd.to_datetime(meta["days"].astype("datetime64[ns]")); D = len(days)
    NY, NX = len(lat), len(lon)
    print(f"[grid] D={D}, grid={NY}x{NX}")
    LON2, LAT2 = np.meshgrid(lon, lat)
    tree = cKDTree(np.column_stack([LAT2.ravel(), LON2.ravel()]))
    fires = load_firms_filtered(NASA_DIR)
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    tmin, tmax = days.min(), days.max()
    m = ((fires["lat"]>=lat_min)&(fires["lat"]<=lat_max)&(fires["lon"]>=lon_min)&(fires["lon"]<=lon_max)&
         (fires["time"]>=tmin)&(fires["time"]<=tmax))
    fires = fires[m].copy()
    fires["day"] = fires["time"].dt.floor("D")
    day_to_idx = pd.Series(np.arange(D), index=days)
    fires["di"] = fires["day"].map(day_to_idx)
    fires = fires.dropna(subset=["di"]).copy()
    fires["di"] = fires["di"].astype(int)
    out = np.zeros((D, 2, NY, NX), dtype=np.float32)
    if not fires.empty:
        for di, df_day in fires.groupby("di"):
            pts = np.column_stack([df_day["lat"].values, df_day["lon"].values])
            _, idx = tree.query(pts)
            iy = idx // NX; ix = idx % NX
            v = df_day["frp"].values.astype(np.float32)
            np.add.at(out[di,1], (iy, ix), v)  # frp
            out[di,0,iy,ix] = 1.0              # label
    # persistent hot pixels
    days_idx = pd.DatetimeIndex(days)
    yrs = days_idx.year.values
    uniq = np.unique(yrs)
    freq = (out[:,0]>0).mean(axis=0)
    years_with_fire = np.zeros((NY,NX), dtype=np.int32)
    for y in uniq:
        mask = (yrs==y)
        if not mask.any(): continue
        hits = (out[mask,0]>0).any(axis=0)
        years_with_fire += hits.astype(np.int32)
    persistent = (freq>=PERSIST_FREQ) & (years_with_fire>=PERSIST_MIN_YEARS)
    if persistent.any():
        out[:,0,persistent] = 0.0; out[:,1,persistent] = 0.0
    # save
    np.save(OUT_4D, out.astype(np.float32))
    flat = out.transpose(0,2,3,1).reshape(D*NY*NX, 2).astype(np.float32)
    np.save(OUT_FLAT, flat)
    np.savez_compressed(
        OUT_META,
        days=days.values.astype("datetime64[ns]"),
        lat=lat, lon=lon,
        feature_names=np.array(["fire_label","fire_frp"], dtype=object),
        shape_4d=np.array([D, 2, NY, NX], dtype=np.int64),
        notes=np.array({"alignment":"time axis identical to merged_27x29_4d.npy"}, dtype=object)
    )
    card = []
    card += [ "# NASA Fire Labels — EAC4 27×29 (aligned)", "" ]
    card += [ f"- 4D: `{OUT_4D.name}`  (days={D}, features=2, {NY}×{NX})" ]
    card += [ f"- Flat: `{OUT_FLAT.name}`  (days*{NY}*{NX}, features=2)" ]
    card += [ f"- Meta: `{OUT_META.name}`", "" ]
    card += [ "## Features", "- fire_label (0/1)", "- fire_frp (MW, summed per day×cell)", "" ]
    Path(OUT_CARD).write_text("\n".join(card), encoding="utf-8")
    print(f"[done] Wrote:\n  {OUT_4D}\n  {OUT_FLAT}\n  {OUT_META}\n  {OUT_CARD}")

if __name__ == "__main__":
    main()
