#!/usr/bin/env python3
"""
Merge CAMS Emission Inventories (ALG) + CAMS EAC4 (ALG) into one ML-ready matrix
- No train/test split
- Aligns on intersection of times
- Uses emissions grid as target
- Nearest-neighbour remap of EAC4 if its grid differs from emissions grid
- Streams per time slice (RAM-safe)

Inputs (hard-coded):
  ./datasets/CAMS/CAMS_emission_inventories_ALG_v6_2_2015_2024_X.npy
  ./datasets/CAMS/CAMS_emission_inventories_ALG_v6_2_2015_2024_meta.npz
  ./datasets/CAMS/EAC4_ALG_X.npy
  ./datasets/CAMS/EAC4_ALG_meta.npz

Outputs:
  ./datasets/CAMS/ALG_merged_X.npy
  ./datasets/CAMS/ALG_merged_meta.npz
  ./datasets/CAMS/ALG_merged.card.md
"""

from pathlib import Path
import json
import numpy as np

# ------------------- Hard-coded paths -------------------
ROOT = Path("./datasets/CAMS")

EMIS_X   = ROOT / "CAMS_emission_inventories_ALG_v6_2_2015_2024_X.npy"
EMIS_META= ROOT / "CAMS_emission_inventories_ALG_v6_2_2015_2024_meta.npz"

EAC4_X   = ROOT / "EAC4_ALG_X.npy"
EAC4_META= ROOT / "EAC4_ALG_meta.npz"

OUT_X    = ROOT / "ALG_merged_X.npy"
OUT_META = ROOT / "ALG_merged_meta.npz"
OUT_CARD = ROOT / "ALG_merged.card.md"

# Stream one time-slice at a time
TIME_BLOCK = 1
# --------------------------------------------------------


def _load_meta(path_npz: Path):
    m = np.load(path_npz, allow_pickle=True)
    # shape info is json in the emission meta; handle both styles
    shape_json = m["shape_info"].item() if hasattr(m["shape_info"], "item") else str(m["shape_info"])
    try:
        shape = json.loads(shape_json)
    except Exception:
        # Some metas saved as np.array(json.dumps(...), dtype=object)
        shape = json.loads(str(m["shape_info"]))
    time = m["time"].astype("datetime64[ns]")
    lat  = m["lat"].astype(np.float64)
    lon  = m["lon"].astype(np.float64)
    names= m["feature_names"].astype(str)
    units= m["feature_units"].astype(str) if "feature_units" in m else np.array([""]*names.size, dtype=object)
    longs= m["feature_long_names"].astype(str) if "feature_long_names" in m else names
    # quality might be a JSON string
    quality = None
    if "quality" in m:
        try:
            quality = json.loads(str(m["quality"]))
        except Exception:
            quality = None
    return {
        "time": time,
        "lat": lat,
        "lon": lon,
        "names": names,
        "units": units,
        "longs": longs,
        "shape": shape,
        "quality": quality,
        "X_path": str(m["X_path"]) if "X_path" in m else None,
    }


def _row_range_for_time_idx(t_idx: int, n_lat: int, n_lon: int):
    n_spatial = n_lat * n_lon
    start = t_idx * n_spatial
    end   = start + n_spatial
    return start, end


def _nearest_index_map(target_vals: np.ndarray, source_vals: np.ndarray):
    """
    For each target coord, return the index of the nearest source coord.
    Assumes both arrays are 1D, monotonically increasing or decreasing.
    """
    # Ensure increasing
    t = target_vals
    s = source_vals
    if (np.diff(s) < 0).all():
        s = s[::-1]
        exp = True
    else:
        exp = False
    # searchsorted needs increasing
    pos = np.searchsorted(s, t, side="left")
    pos = np.clip(pos, 0, s.size-1)
    pos_left = np.clip(pos-1, 0, s.size-1)
    # pick closer of left/pos
    choose_left = np.abs(t - s[pos_left]) <= np.abs(t - s[pos])
    idx = np.where(choose_left, pos_left, pos)
    if exp:
        idx = (s.size-1) - idx  # map back if reversed
    return idx.astype(np.int64)


def main():
    # Load metas
    em = _load_meta(EMIS_META)
    ea = _load_meta(EAC4_META)

    # Read shapes
    n_time_em = em["shape"]["n_time"]
    n_lat_em  = em["shape"]["n_lat"]
    n_lon_em  = em["shape"]["n_lon"]
    n_feat_em = em["shape"]["n_features"]

    n_time_ea = ea["shape"]["n_time"]
    n_lat_ea  = ea["shape"]["n_lat"]
    n_lon_ea  = ea["shape"]["n_lon"]
    n_feat_ea = ea["shape"]["n_features"]

    # Alignment: time intersection (preserve chronological order)
    time_common = np.intersect1d(em["time"], ea["time"])
    if time_common.size == 0:
        raise SystemExit("No overlapping timestamps between emissions and EAC4.")

    # Build index arrays from each dataset to common time
    # (x64 indexes allow big datasets)
    em_time_to_idx = {int(t): i for i, t in enumerate(em["time"].astype("datetime64[ns]"))}
    ea_time_to_idx = {int(t): i for i, t in enumerate(ea["time"].astype("datetime64[ns]"))}
    em_tidx = np.array([em_time_to_idx[int(t)] for t in time_common], dtype=np.int64)
    ea_tidx = np.array([ea_time_to_idx[int(t)] for t in time_common], dtype=np.int64)

    # Spatial alignment: emissions grid is target
    lat_tgt = em["lat"]
    lon_tgt = em["lon"]
    lat_src = ea["lat"]
    lon_src = ea["lon"]

    same_lat = (lat_tgt.size == lat_src.size) and np.allclose(lat_tgt, lat_src, rtol=0, atol=1e-9)
    same_lon = (lon_tgt.size == lon_src.size) and np.allclose(lon_tgt, lon_src, rtol=0, atol=1e-9)

    if same_lat and same_lon:
        # 1:1 mapping
        iy_map = None
        ix_map = None
        print("[align] EAC4 grid matches emissions grid — no spatial remap needed.")
    else:
        # Nearest-neighbour index map from target (emissions) -> source (EAC4)
        iy_map = _nearest_index_map(lat_tgt, lat_src)
        ix_map = _nearest_index_map(lon_tgt, lon_src)
        print(f"[align] EAC4 grid differs; using nearest-neighbour remap.")
        print(f"        lat: target {lat_tgt.size} vs source {lat_src.size} ; lon: target {lon_tgt.size} vs source {lon_src.size}")

    # Prepare output memmap
    n_time = time_common.size
    n_spatial_em = n_lat_em * n_lon_em
    n_rows = n_time * n_spatial_em
    n_feat = n_feat_em + n_feat_ea

    X_out = np.memmap(OUT_X, mode="w+", dtype=np.float32, shape=(n_rows, n_feat))

    # Open source memmaps
    X_em = np.memmap(EMIS_X, mode="r", dtype=np.float32, shape=(n_time_em * n_spatial_em, n_feat_em))
    X_ea = np.memmap(EAC4_X, mode="r", dtype=np.float32, shape=(n_time_ea * (n_lat_ea * n_lon_ea), n_feat_ea))

    print(f"[write] Merging rows: n_time={n_time}, grid={n_lat_em}x{n_lon_em}, rows={n_rows:,}, cols={n_feat}")

    # Stream per time block
    for k0 in range(0, n_time, TIME_BLOCK):
        k1 = min(k0 + TIME_BLOCK, n_time)

        # target row slab in merged output
        row_out_start = k0 * n_spatial_em
        row_out_end   = k1 * n_spatial_em

        # --- Emissions slice (already on target grid) ---
        em_rows = []
        for tk in range(k0, k1):
            t_idx = em_tidx[tk]
            rs, re = _row_range_for_time_idx(t_idx, n_lat_em, n_lon_em)
            em_rows.append(X_em[rs:re, :])   # (n_spatial_em, n_feat_em)
        em_block = np.vstack(em_rows) if len(em_rows) > 1 else em_rows[0]
        X_out[row_out_start:row_out_end, 0:n_feat_em] = em_block

        # --- EAC4 slice (maybe remap) ---
        ea_rows = []
        for tk in range(k0, k1):
            t_idx = ea_tidx[tk]
            rs, re = _row_range_for_time_idx(t_idx, n_lat_ea, n_lon_ea)
            block = X_ea[rs:re, :]   # shape: (n_lat_ea*n_lon_ea, n_feat_ea)
            # reshape to (lat_ea, lon_ea, feat)
            block3 = block.reshape(n_lat_ea, n_lon_ea, n_feat_ea)

            if iy_map is None:
                # Same grid — keep as-is but reshape to target rows
                mapped = block3  # (n_lat_em, n_lon_em, n_feat_ea)
            else:
                # Nearest-neighbour sampling to emissions grid
                mapped = block3[iy_map[:, None], ix_map[None, :], :]  # (n_lat_em, n_lon_em, n_feat_ea)

            ea_rows.append(mapped.reshape(-1, n_feat_ea))

        ea_block = np.vstack(ea_rows) if len(ea_rows) > 1 else ea_rows[0]
        X_out[row_out_start:row_out_end, n_feat_em:n_feat_em+n_feat_ea] = ea_block

        X_out.flush()
        pct = 100.0 * (row_out_end / n_rows)
        print(f"[write] time[{k0}:{k1}) rows[{row_out_start}:{row_out_end})  {pct:5.1f}%")

    # Build merged metadata
    feature_names = np.concatenate([("emis:" + em["names"]), ("eac4:" + ea["names"])])
    feature_units = np.concatenate([em["units"], ea["units"]])
    feature_longs = np.concatenate([em["longs"], ea["longs"]])

    # Quality summary (optional; if present in metas)
    quality = {}
    if isinstance(em.get("quality"), dict) and isinstance(ea.get("quality"), dict):
        # Keep columns mask if both present; otherwise leave empty
        em_keep = np.array(em["quality"].get("keep_columns_mask", [True]*n_feat_em), dtype=bool)
        ea_keep = np.array(ea["quality"].get("keep_columns_mask", [True]*n_feat_ea), dtype=bool)
        merged_keep = np.concatenate([em_keep, ea_keep]).tolist()
        quality["keep_columns_mask"] = merged_keep

    shape_info = {
        "n_time": int(n_time),
        "n_lat": int(n_lat_em),
        "n_lon": int(n_lon_em),
        "n_features": int(n_feat),
        "n_instances": int(n_rows),
        "bbox": {
            "lat_min": float(em["lat"].min()),
            "lat_max": float(em["lat"].max()),
            "lon_min": float(em["lon"].min()),
            "lon_max": float(em["lon"].max()),
        },
        "ordering": "row = t*(n_lat*n_lon) + y*n_lon + x ; columns = feature_names",
        "time_coverage_note": "intersection of emissions and EAC4 timesteps",
        "spatial_note": ("EAC4 remapped to emissions grid via nearest neighbour"
                         if iy_map is not None else "identical grid"),
    }

    # Save meta
    np.savez_compressed(
        OUT_META,
        X_path=np.array(str(OUT_X), dtype=object),
        time=time_common.astype("datetime64[ns]"),
        lat=em["lat"], lon=em["lon"],             # target grid (emissions)
        feature_names=feature_names.astype(object),
        feature_units=feature_units.astype(object),
        feature_long_names=feature_longs.astype(object),
        shape_info=np.array(json.dumps(shape_info), dtype=object),
        quality=json.dumps(quality) if quality else json.dumps({}),
        sources=np.array(["emissions_v6.2_ALG_2015_2024", "EAC4_ALG"], dtype=object),
        emis_meta_path=np.array(str(EMIS_META), dtype=object),
        eac4_meta_path=np.array(str(EAC4_META), dtype=object),
    )

    # Card
    years = sorted({int(str(t)[:4]) for t in time_common.astype("datetime64[Y]").astype(int) + 1970})
    approx_gb = (n_rows * n_feat * 4) / (1024**3)
    md = []
    md.append("# Algeria Merged Dataset — CAMS Emissions v6.2 + CAMS EAC4")
    md.append("")
    md.append(f"- **Matrix**: `{OUT_X}` (NumPy memmap, float32)")
    md.append(f"- **Metadata**: `{OUT_META}`")
    md.append(f"- **Rows (instances)**: {n_rows:,}")
    md.append(f"- **Columns (features)**: {n_feat:,}  — emissions: {n_feat_em}, EAC4: {n_feat_ea}")
    md.append(f"- **Time steps (intersection)**: {time_common.size} ({years[0]}–{years[-1]})")
    md.append(f"- **Grid (target)**: emissions Algeria bbox — {n_lat_em} lat × {n_lon_em} lon")
    if iy_map is None:
        md.append("- **Spatial alignment**: identical grids")
    else:
        md.append("- **Spatial alignment**: EAC4 → emissions grid via nearest-neighbour (per-cell)")
    md.append("")
    md.append("## Feature naming")
    md.append("- Emissions columns prefixed with `emis:`")
    md.append("- EAC4 columns prefixed with `eac4:`")
    md.append("")
    md.append("## Ordering")
    md.append("- Rows: `row = t*(n_lat*n_lon) + y*n_lon + x`")
    md.append("- Columns: concatenated `[emis features, eac4 features]`")
    OUT_CARD.write_text("\n".join(md), encoding="utf-8")

    print("✅ Done")
    print(f"  - {OUT_X}")
    print(f"  - {OUT_META}")
    print(f"  - {OUT_CARD}")


if __name__ == "__main__":
    main()
