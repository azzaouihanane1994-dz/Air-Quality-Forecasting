#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_eac4_cams_concatenate.py

Merges the DAILY, 195×210 outputs of EAC4 (drivers) and CAMS emissions
by concatenating along the **feature axis**.

Expected inputs (created by merge_eac4_cams_daily_195x210.py):
  ./datasets/CAMS/eac4_daily_195x210_4d.npy        (D, F_eac4, 195, 210)
  ./datasets/CAMS/eac4_daily_195x210_flat.npy      (D*195*210, F_eac4)
  ./datasets/CAMS/eac4_daily_195x210_meta.npz

  ./datasets/CAMS/cams_emis_daily_4d.npy           (D, F_emis, 195, 210)
  ./datasets/CAMS/cams_emis_daily_flat.npy         (D*195*210, F_emis)
  ./datasets/CAMS/cams_emis_daily_meta.npz

Generated outputs:
  ./datasets/CAMS/merged_daily_4d.npy              (D, F_total, 195, 210)
  ./datasets/CAMS/merged_daily_flat.npy            (D*195*210, F_total)
  ./datasets/CAMS/merged_daily_meta.npz
  ./datasets/CAMS/merged_daily.card.md

Notes:
- The script validates that days, lat, lon, and grid sizes match.
- Features are prefixed in meta as: 'eac4:<name>' and 'emis:<name>' to avoid collisions.
- Writing is chunked along the feature axis for memory safety.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------- Hardcoded paths ---------------------------
ROOT = Path("./datasets/CAMS")

EAC4_4D   = ROOT / "eac4_daily_195x210_4d.npy"
EAC4_FLAT = ROOT / "eac4_daily_195x210_flat.npy"
EAC4_META = ROOT / "eac4_daily_195x210_meta.npz"

EMIS_4D   = ROOT / "cams_emis_daily_4d.npy"
EMIS_FLAT = ROOT / "cams_emis_daily_flat.npy"
EMIS_META = ROOT / "cams_emis_daily_meta.npz"

MERGED_4D   = ROOT / "merged_daily_4d.npy"
MERGED_FLAT = ROOT / "merged_daily_flat.npy"
MERGED_META = ROOT / "merged_daily_meta.npz"
MERGED_CARD = ROOT / "merged_daily.card.md"
# ----------------------------------------------------------------------

# Chunking knobs
FEAT_CHUNK = 32     # how many features to copy at once (adjust for your RAM/IO)

def _load_meta(path):
    z = np.load(path, allow_pickle=True)
    days = z["days"]
    lat  = z["lat"]
    lon  = z["lon"]
    fns  = z.get("feature_names", None)
    shape4 = z.get("shape_4d", None)
    note = z.get("note", "")
    return days, lat, lon, fns, shape4, (note.item() if isinstance(note, np.ndarray) else note)

def _validate_alignment(e_days, e_lat, e_lon, m_days, m_lat, m_lon):
    if e_days.shape != m_days.shape or not np.all(e_days == m_days):
        raise SystemExit("[error] 'days' arrays differ between EAC4 and CAMS emissions.")
    if e_lat.shape != m_lat.shape or not np.allclose(e_lat, m_lat):
        raise SystemExit("[error] 'lat' arrays differ between EAC4 and CAMS emissions.")
    if e_lon.shape != m_lon.shape or not np.allclose(e_lon, m_lon):
        raise SystemExit("[error] 'lon' arrays differ between EAC4 and CAMS emissions.")

def merge():
    print("[meta] Loading EAC4 meta…")
    e_days, e_lat, e_lon, e_names, e_shape4, e_note = _load_meta(EAC4_META)
    print("[meta] Loading CAMS emissions meta…")
    m_days, m_lat, m_lon, m_names, m_shape4, m_note = _load_meta(EMIS_META)

    print("[meta] Validating timeline and grid alignment…")
    _validate_alignment(e_days, e_lat, e_lon, m_days, m_lat, m_lon)
    D = e_days.shape[0]; Y = e_lat.shape[0]; X = e_lon.shape[0]

    # Determine feature counts
    if e_shape4 is None or m_shape4 is None:
        raise SystemExit("[error] Missing shape_4d in meta files.")
    Fe = int(e_shape4[1]); Fm = int(m_shape4[1]); Ftot = Fe + Fm
    print(f"[shape] D={D}, Y={Y}, X={X}, Fe={Fe}, Fm={Fm}, Ftot={Ftot}")

    # Build merged feature names with prefixes
    if e_names is None: e_names = np.array([f"eac4:feat_{i}" for i in range(Fe)], dtype=object)
    else: e_names = np.array([f"eac4:{str(x)}" for x in e_names], dtype=object)
    if m_names is None: m_names = np.array([f"emis:feat_{i}" for i in range(Fm)], dtype=object)
    else: m_names = np.array([f"emis:{str(x)}" for x in m_names], dtype=object)
    merged_names = np.concatenate([e_names, m_names])

    # Open inputs as memmaps
    print("[io] Opening source arrays…")
    E4 = np.memmap(EAC4_4D, dtype=np.float32, mode="r", shape=(D, Fe, Y, X))
    E2 = np.memmap(EAC4_FLAT, dtype=np.float32, mode="r", shape=(D*Y*X, Fe))
    M4 = np.memmap(EMIS_4D, dtype=np.float32, mode="r", shape=(D, Fm, Y, X))
    M2 = np.memmap(EMIS_FLAT, dtype=np.float32, mode="r", shape=(D*Y*X, Fm))

    # Prepare outputs
    print("[io] Allocating outputs…")
    OUT4 = np.memmap(MERGED_4D, dtype=np.float32, mode="w+", shape=(D, Ftot, Y, X))
    OUT2 = np.memmap(MERGED_FLAT, dtype=np.float32, mode="w+", shape=(D*Y*X, Ftot))

    # Write in feature chunks: first EAC4 then EMIS to keep order clear
    print("[merge][4D] Concatenating along feature axis…")
    # EAC4 block
    for f0 in range(0, Fe, FEAT_CHUNK):
        f1 = min(f0 + FEAT_CHUNK, Fe)
        OUT4[:, f0:f1, :, :] = E4[:, f0:f1, :, :]
        OUT4.flush()
        print(f"  EAC4 4D features copied: {f1}/{Fe}")
    # EMIS block
    base = Fe
    for f0 in range(0, Fm, FEAT_CHUNK):
        f1 = min(f0 + FEAT_CHUNK, Fm)
        OUT4[:, base+f0:base+f1, :, :] = M4[:, f0:f1, :, :]
        OUT4.flush()
        print(f"  EMIS 4D features copied: {f1}/{Fm}")
    del OUT4  # flush

    print("[merge][FLAT] Concatenating along feature axis…")
    # EAC4 block
    for f0 in range(0, Fe, FEAT_CHUNK):
        f1 = min(f0 + FEAT_CHUNK, Fe)
        OUT2[:, f0:f1] = E2[:, f0:f1]
        OUT2.flush()
        print(f"  EAC4 FLAT features copied: {f1}/{Fe}")
    # EMIS block
    for f0 in range(0, Fm, FEAT_CHUNK):
        f1 = min(f0 + FEAT_CHUNK, Fm)
        OUT2[:, base+f0:base+f1] = M2[:, f0:f1]
        OUT2.flush()
        print(f"  EMIS FLAT features copied: {f1}/{Fm}")
    del OUT2  # flush

    # Save meta
    print("[meta] Writing merged meta…")
    np.savez_compressed(
        MERGED_META,
        days=e_days.astype("datetime64[ns]"),
        lat=e_lat, lon=e_lon,
        feature_names=merged_names,
        shape_4d=np.array([D, Ftot, Y, X], dtype=np.int64),
        sources=np.array({
            "eac4_4d": str(EAC4_4D),
            "eac4_flat": str(EAC4_FLAT),
            "emissions_4d": str(EMIS_4D),
            "emissions_flat": str(EMIS_FLAT),
            "eac4_meta": str(EAC4_META),
            "emissions_meta": str(EMIS_META),
        }, dtype=object),
        notes=np.array({
            "eac4_note": e_note,
            "emissions_note": m_note,
            "merge": "Concatenated along feature axis; feature names prefixed with 'eac4:' and 'emis:'."
        }, dtype=object)
    )

    # Data card
    print("[card] Writing merged data card…")
    approx_gb_4d = (D * Ftot * Y * X * 4) / (1024**3)
    with open(MERGED_CARD, "w", encoding="utf-8") as f:
        f.write("# Merged Daily Dataset (EAC4 + CAMS Emissions) — 195×210\n\n")
        f.write("## Files\n")
        f.write(f"- **4D**: `{MERGED_4D}`  — shape `(days={D}, features={Ftot}, y={Y}, x={X})`\n")
        f.write(f"- **Flat**: `{MERGED_FLAT}`  — shape `(days*{Y}*{X}, features={Ftot})`\n")
        f.write(f"- **Meta**: `{MERGED_META}`\n\n")
        f.write("## Time & Grid\n")
        f.write(f"- Days: `{D}` (2015-01-01 → 2024-12-31)\n")
        f.write(f"- Grid: `{Y} × {X}` (lat `{float(e_lat.min()):.3f}`..`{float(e_lat.max()):.3f}`, lon `{float(e_lon.min()):.3f}`..`{float(e_lon.max()):.3f}`)\n\n")
        f.write("## Features\n")
        f.write(f"- EAC4 feature count: `{Fe}`\n")
        f.write(f"- Emissions feature count: `{Fm}`\n")
        f.write(f"- **Merged total**: `{Ftot}`\n")
        # sample names
        try:
            head = ", ".join([str(x) for x in merged_names[:12]])
            tail = ", ".join([str(x) for x in merged_names[-3:]]) if merged_names.size > 15 else ""
            f.write(f"- Sample: {head}{' … ' + tail if tail else ''}\n\n")
        except Exception:
            f.write("\n")
        f.write("## Notes\n")
        f.write("- Features were concatenated along the feature axis.\n")
        f.write("- EAC4 daily series are daily means upsampled to 195×210 by bilinear interpolation.\n")
        f.write("- Emissions daily series are equal-weight allocation of monthly means to each day.\n")
        f.write(f"- Approx raw 4D size (float32): ~{approx_gb_4d:.2f} GB.\n")

    print("✅ Merged outputs written:")
    print(f"  - {MERGED_4D}")
    print(f"  - {MERGED_FLAT}")
    print(f"  - {MERGED_META}")
    print(f"  - {MERGED_CARD}")

if __name__ == "__main__":
    merge()
