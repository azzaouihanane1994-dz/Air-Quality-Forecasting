#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_and_append_static_humdata.py  — UPDATED

Converts HOTOSM GeoPackages in ./datasets/humdata into **static per-cell features**
on the Algeria 195×210 grid from merged_daily_meta.npz, then (optionally) appends
those static features to your daily merged datasets.

Static features (baseline):
- Roads (lines): total length per cell [meters]; plus length by highway class buckets.
- Buildings (polygons): count per cell; total footprint area [m^2].
- Populated places (points): counts by place class buckets.

Outputs (static):
  ./datasets/humdata/static_humdata_195x210_4d.npy      (F_static, 195, 210)
  ./datasets/humdata/static_humdata_195x210_flat.npy    (195*210, F_static)
  ./datasets/humdata/static_humdata_195x210_meta.npz
  ./datasets/humdata/static_humdata_195x210.card.md

If APPEND_TO_DAILY=True (writes into ./datasets/CAMS/):
  merged_plus_static_4d.npy, merged_plus_static_flat.npy, merged_plus_static_meta.npz, merged_plus_static.card.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

# ------------------- CONFIG -------------------
ROOT = Path("./datasets")
HUM = ROOT / "humdata"
CAMS = ROOT / "CAMS"

MERGED_META = CAMS / "merged_daily_meta.npz"
MERGED_4D   = CAMS / "merged_daily_4d.npy"
MERGED_FLAT = CAMS / "merged_daily_flat.npy"

# Outputs (static)
STATIC_4D   = HUM / "static_humdata_195x210_4d.npy"    # (F_static, 195, 210)
STATIC_FLAT = HUM / "static_humdata_195x210_flat.npy"  # (195*210, F_static)
STATIC_META = HUM / "static_humdata_195x210_meta.npz"
STATIC_CARD = HUM / "static_humdata_195x210.card.md"

# Append switch
APPEND_TO_DAILY = False
PLUS_4D   = CAMS / "merged_plus_static_4d.npy"
PLUS_FLAT = CAMS / "merged_plus_static_flat.npy"
PLUS_META = CAMS / "merged_plus_static_meta.npz"
PLUS_CARD = CAMS / "merged_plus_static.card.md"

# Fast approximate mode vs exact intersections
# - If False (recommended for speed): assign length/area/count to the **cell containing the centroid**.
# - If True: compute exact intersections (accurate, but much slower on big GPKGs).
EXACT_INTERSECT = False

# Buckets we’ll summarize
ROAD_CLASS_BUCKETS = ["motorway","trunk","primary","secondary","tertiary",
                      "residential","service","track"]
PLACE_BUCKETS = ["city","town","village","hamlet","locality","suburb","neighbourhood"]
# ----------------------------------------------


def _mesh_edges_from_centers(lat1d, lon1d):
    """Return (lat_edges, lon_edges) arrays from 1D centers (assumes regular spacing)."""
    dlat = float(np.mean(np.diff(np.sort(lat1d))))
    dlon = float(np.mean(np.diff(np.sort(lon1d))))
    lat_edges = np.concatenate([lat1d - dlat/2, [lat1d[-1] + dlat/2]])
    lon_edges = np.concatenate([lon1d - dlon/2, [lon1d[-1] + dlon/2]])
    return lat_edges, lon_edges


def _cell_index_from_lonlat(lon, lat, lon_edges, lat_edges):
    """Compute (iy, ix) bin of a lon/lat point given cell edges; returns None if outside."""
    ix = np.searchsorted(lon_edges, lon, side="right") - 1
    iy = np.searchsorted(lat_edges, lat, side="right") - 1
    if ix < 0 or ix >= lon_edges.size-1: return None
    if iy < 0 or iy >= lat_edges.size-1: return None
    return int(iy), int(ix)


def _init_feature_stack(ny, nx):
    """
    Returns:
      acc: dict[str, np.ndarray]  # keys WITHOUT 'static:' prefix
      keys: list[str]             # same keys order for stacking
    """
    acc = {}
    keys = []

    # Roads: total length; per-class length
    acc["roads_len_m"] = np.zeros((ny, nx), dtype=np.float64); keys.append("roads_len_m")
    for cls in ROAD_CLASS_BUCKETS:
        k = f"roads_len_m:{cls}"
        acc[k] = np.zeros((ny, nx), dtype=np.float64); keys.append(k)

    # Buildings: count, total area  (NOTE: consistent keys: building_count / building_area_m2)
    acc["building_count"] = np.zeros((ny, nx), dtype=np.float64); keys.append("building_count")
    acc["building_area_m2"] = np.zeros((ny, nx), dtype=np.float64); keys.append("building_area_m2")

    # Places: counts by class
    for p in PLACE_BUCKETS:
        k = f"places:{p}"
        acc[k] = np.zeros((ny, nx), dtype=np.float64); keys.append(k)

    return acc, keys


def _append_arrays(acc, keys):
    """Stack accumulators (using internal keys) into (F, ny, nx)."""
    # Guard — catch typos early
    missing = [k for k in keys if k not in acc]
    if missing:
        raise KeyError(f"Accumulator keys missing: {missing}")
    arrs = [acc[k][None, ...] for k in keys]  # (1, ny, nx) per key
    return np.concatenate(arrs, axis=0), np.array(keys, dtype=object)


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn("Layer without CRS—assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def _metric_length_m(geom: "shapely.geometry.base.BaseGeometry") -> float:
    # Project to WebMercator for length in meters (robust & quick)
    gseries = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    return float(gseries.length.iloc[0])


def _metric_area_m2(geom: "shapely.geometry.base.BaseGeometry") -> float:
    gseries = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    return float(gseries.area.iloc[0])


def build_static():
    # 1) Load grid
    meta = np.load(MERGED_META, allow_pickle=True)
    days = meta["days"]; lat = meta["lat"].astype(float); lon = meta["lon"].astype(float)
    D, F_total, NY, NX = map(int, meta["shape_4d"])
    print(f"[grid] days={D}, grid={NY}x{NX}, merged features={F_total}")

    lat_edges, lon_edges = _mesh_edges_from_centers(lat, lon)

    # 2) Prepare accumulators
    acc, names_order = _init_feature_stack(NY, NX)

    # 3) Roads
    roads_files = sorted(list(HUM.glob("*roads*.gpkg*")))
    if roads_files:
        rf = roads_files[0]
        print(f"[roads] {rf.name}")
        g = gpd.read_file(rf)
        g = _ensure_wgs84(g)
        # infer class column
        hcol = None
        for c in g.columns:
            if c.lower() == "highway":
                hcol = c; break

        # iterate (centroid-based binning; measuring full geometry in meters)
        for idx, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            try:
                lonc, latc = geom.representative_point().x, geom.representative_point().y
            except Exception:
                continue
            idx_cell = _cell_index_from_lonlat(lonc, latc, lon_edges, lat_edges)
            if idx_cell is None: continue
            iy, ix = idx_cell

            # length per feature (we're not splitting by exact cell intersection in fast mode)
            length_m = _metric_length_m(geom)
            acc["roads_len_m"][iy, ix] += length_m
            if hcol is not None:
                cls = str(row[hcol]).lower() if pd.notna(row[hcol]) else ""
                if cls in ROAD_CLASS_BUCKETS:
                    acc[f"roads_len_m:{cls}"][iy, ix] += length_m

        print("  roads accumulated.")

    # 4) Buildings
    bldg_files = sorted(list(HUM.glob("*buildings*.gpkg*")))
    if bldg_files:
        bf = bldg_files[0]
        print(f"[buildings] {bf.name}")
        g = gpd.read_file(bf)
        g = _ensure_wgs84(g)
        for idx, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            try:
                lonc, latc = geom.representative_point().x, geom.representative_point().y
            except Exception:
                continue
            idx_cell = _cell_index_from_lonlat(lonc, latc, lon_edges, lat_edges)
            if idx_cell is None: continue
            iy, ix = idx_cell

            # area per feature
            try:
                area_m2 = _metric_area_m2(geom)
            except Exception:
                area_m2 = 0.0

            # ✅ FIXED names: building_count / building_area_m2
            acc["building_count"][iy, ix] += 1.0
            acc["building_area_m2"][iy, ix] += area_m2
        print("  buildings accumulated.")

    # 5) Places
    place_files = sorted(list(HUM.glob("*populated_places*.gpkg*")))
    if place_files:
        pf = place_files[0]
        print(f"[places] {pf.name}")
        g = gpd.read_file(pf)
        g = _ensure_wgs84(g)
        pcol = None
        for c in g.columns:
            if c.lower() == "place":
                pcol = c; break
        for idx, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type not in ("Point","MultiPoint"):
                try:
                    geom = geom.representative_point()
                except Exception:
                    continue
            lonc, latc = geom.x, geom.y
            idx_cell = _cell_index_from_lonlat(lonc, latc, lon_edges, lat_edges)
            if idx_cell is None: continue
            iy, ix = idx_cell

            acc_key = None
            if pcol is not None and pd.notna(row[pcol]):
                cls = str(row[pcol]).lower()
                if cls in PLACE_BUCKETS:
                    acc_key = f"places:{cls}"
            if acc_key is None:
                acc_key = "places:locality"  # bucket the rest here
            acc[acc_key][iy, ix] += 1.0
        print("  places accumulated.")

    # 6) Assemble arrays and write
    static_fx_yx, static_keys = _append_arrays(acc, names_order)   # internal keys (no 'static:')
    F_static = static_fx_yx.shape[0]
    # Use prefixed names only for meta/cards
    static_names = np.array([f"static:{k}" for k in static_keys], dtype=object)

    print(f"[static] shape = (F_static={F_static}, {NY}, {NX})")

    # Save static cubes
    np.save(STATIC_4D, static_fx_yx.astype(np.float32))
    np.save(STATIC_FLAT, static_fx_yx.transpose(1,2,0).reshape(NY*NX, F_static).astype(np.float32))
    np.savez_compressed(
        STATIC_META,
        lat=lat, lon=lon, feature_names=static_names,
        shape_3d=np.array([F_static, NY, NX], dtype=np.int64),
        notes=np.array({
            "mode": "centroid-bin" if not EXACT_INTERSECT else "exact-intersect",
            "roads": "meters in EPSG:3857; summed per cell",
            "buildings": "area m2 and count per cell",
            "places": "point counts per class per cell"
        }, dtype=object)
    )

    approx_gb = (F_static * NY * NX * 4) / (1024**3)
    card = []
    card += [ "# HOTOSM → Static grid (195×210)", "" ]
    card += [ f"- Static features: `{F_static}`", f"- Grid: `{NY} × {NX}`", f"- Approx raw size: ~{approx_gb:.2f} GB", "" ]
    card += [ "## Feature list" ]
    for nm in static_names:
        card.append(f"- {nm}")
    STATIC_CARD.write_text("\n".join(card), encoding="utf-8")

    return static_fx_yx, static_names, (NY, NX), D, F_total


def append_to_daily(static_fx_yx, static_names, grid_shape, D, F_total):
    NY, NX = grid_shape
    F_static = static_fx_yx.shape[0]
    cells = NY * NX

    # 4D daily (days, features, y, x)
    X4 = np.memmap(MERGED_4D, dtype=np.float32, mode="r", shape=(D, F_total, NY, NX))
    OUT4 = np.memmap(PLUS_4D, dtype=np.float32, mode="w+", shape=(D, F_total + F_static, NY, NX))

    # copy existing
    OUT4[:, :F_total, :, :] = X4
    # broadcast static along the day axis
    OUT4[:, F_total:, :, :] = static_fx_yx[None, :, :, :]
    del OUT4

    # FLAT daily (days*cells, features)
    X2 = np.memmap(MERGED_FLAT, dtype=np.float32, mode="r", shape=(D * cells, F_total))
    OUT2 = np.memmap(PLUS_FLAT, dtype=np.float32, mode="w+", shape=(D * cells, F_total + F_static))
    OUT2[:, :F_total] = X2
    static_flat = static_fx_yx.transpose(1,2,0).reshape(cells, F_static).astype(np.float32)
    # tile per day
    OUT2[:, F_total:] = np.tile(static_flat, (D, 1))
    del OUT2

    # meta + card
    merged_meta = np.load(MERGED_META, allow_pickle=True)
    lat, lon = merged_meta["lat"], merged_meta["lon"]
    days = merged_meta["days"]
    names_merged = merged_meta["feature_names"].astype(object)
    names_plus = np.concatenate([names_merged, static_names])

    np.savez_compressed(
        PLUS_META,
        days=days, lat=lat, lon=lon,
        feature_names=names_plus,
        shape_4d=np.array([D, F_total + F_static, NY, NX], dtype=np.int64),
        sources=np.array({
            "base_4d": str(MERGED_4D),
            "base_flat": str(MERGED_FLAT),
            "base_meta": str(MERGED_META),
            "static_4d": str(STATIC_4D),
            "static_flat": str(STATIC_FLAT),
            "static_meta": str(STATIC_META),
        }, dtype=object),
        notes=np.array({
            "merge": "Concatenated along feature axis; static features broadcast across days."
        }, dtype=object)
    )

    approx_gb = (D * (F_total + F_static) * NY * NX * 4) / (1024**3)
    card = []
    card += [ "# Daily merged + static (EAC4 + CAMS + HOTOSM)", "" ]
    card += [ f"- 4D: `{PLUS_4D.name}`  (days={D}, features={F_total + F_static}, {NY}×{NX})" ]
    card += [ f"- Flat: `{PLUS_FLAT.name}`  (days*{NY}*{NX}, features={F_total + F_static})" ]
    card += [ f"- Meta: `{PLUS_META.name}`", "" ]
    card += [ f"- Added static features: {F_static}", "## Static feature names (prefix `static:`)", ]
    card += [ *(f"- {nm}" for nm in static_names) ]
    card += [ "", f"- Approx raw 4D size: ~{approx_gb:.2f} GB" ]
    PLUS_CARD.write_text("\n".join(card), encoding="utf-8")

    print("✅ Appended static features to daily datasets.")


def main():
    static_fx_yx, static_names, grid_shape, D, F_total = build_static()
    if APPEND_TO_DAILY:
        append_to_daily(static_fx_yx, static_names, grid_shape, D, F_total)
    print("Done.")


if __name__ == "__main__":
    main()
