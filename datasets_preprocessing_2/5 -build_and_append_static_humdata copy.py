#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 -build_and_append_static_humdata_27x29_fixed_v2.py

Bugfix: previous centroid binning assumed lat/lon edges are ascending.
If your meta has lat in **descending** order (common in some grids), np.searchsorted()
fails to find bins and everything lands as "outside", yielding all zeros.

This version:
- sorts lat/lon centers to build ascending edges for searchsorted
- computes (iy, ix) in sorted space, then maps back to the original indices
- adds diagnostics: how many centroids fell inside bbox per layer

Everything else is the same as _fixed.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from shapely.geometry import box

# ------------------- CONFIG -------------------
ROOT = Path("./datasets")
HUM = ROOT / "humdata"
CAMS = ROOT / "CAMS"

MERGED_META = CAMS / "merged_27x29_meta.npz"
MERGED_4D   = CAMS / "merged_27x29_4d.npy"
MERGED_FLAT = CAMS / "merged_27x29_flat.npy"

ROADS_PATH     = HUM / "hotosm_dza_roads_lines_gpkg.gpkg"
ROADS_LAYER    = "hotosm_dza_roads_lines_gpkg"
BUILDINGS_PATH = HUM / "hotosm_dza_buildings_polygons_gpkg.gpkg"
BUILDINGS_LAYER= "hotosm_dza_buildings_polygons_gpkg"
PLACES_PATH    = HUM / "hotosm_dza_populated_places_points_gpkg.gpkg"
PLACES_LAYER   = "hotosm_dza_populated_places_points_gpkg"

STATIC_3D   = HUM / "static_humdata_27x29_3d.npy"
STATIC_FLAT = HUM / "static_humdata_27x29_flat.npy"
STATIC_META = HUM / "static_humdata_27x29_meta.npz"
STATIC_CARD = HUM / "static_humdata_27x29.card.md"

APPEND_TO_DAILY = False
PLUS_4D   = CAMS / "merged_plus_static_27x29_4d.npy"
PLUS_FLAT = CAMS / "merged_plus_static_27x29_flat.npy"
PLUS_META = CAMS / "merged_plus_static_27x29_meta.npz"
PLUS_CARD = CAMS / "merged_plus_static_27x29.card.md"

EXACT_INTERSECT = False   # centroid mode is now robust to lat descending
LOG_EVERY = 200000

ROAD_CLASS_BUCKETS = ["motorway","trunk","primary","secondary","tertiary",
                      "residential","service","track"]
PLACE_BUCKETS = ["city","town","village","hamlet","locality","suburb","neighbourhood"]
# -----------------------------------------------------------

def _build_edges_and_maps(lat_centers, lon_centers):
    """
    Return:
      lat_edges_sorted, lon_edges_sorted (ascending)
      lat_sort_idx, lon_sort_idx        (map sorted -> original indices)
      lat_unsort, lon_unsort            (map original -> sorted indices)
    """
    lat_centers = np.asarray(lat_centers).astype(float)
    lon_centers = np.asarray(lon_centers).astype(float)

    lat_sorted_idx = np.argsort(lat_centers)        # ascending
    lon_sorted_idx = np.argsort(lon_centers)

    lat_sorted = lat_centers[lat_sorted_idx]
    lon_sorted = lon_centers[lon_sorted_idx]

    dlat = float(np.mean(np.diff(lat_sorted)))
    dlon = float(np.mean(np.diff(lon_sorted)))
    lat_edges = np.concatenate([lat_sorted - dlat/2, [lat_sorted[-1] + dlat/2]])
    lon_edges = np.concatenate([lon_sorted - dlon/2, [lon_sorted[-1] + dlon/2]])

    # map original index -> sorted index
    lat_unsort = np.empty_like(lat_sorted_idx)
    lon_unsort = np.empty_like(lon_sorted_idx)
    lat_unsort[lat_sorted_idx] = np.arange(lat_sorted_idx.size)
    lon_unsort[lon_sorted_idx] = np.arange(lon_sorted_idx.size)

    return lat_edges, lon_edges, lat_sorted_idx, lon_sorted_idx, lat_unsort, lon_unsort

def _bin_point(lon, lat, lon_edges_sorted, lat_edges_sorted):
    ix = np.searchsorted(lon_edges_sorted, lon, side="right") - 1
    iy = np.searchsorted(lat_edges_sorted, lat, side="right") - 1
    if ix < 0 or ix >= lon_edges_sorted.size-1: return None
    if iy < 0 or iy >= lat_edges_sorted.size-1: return None
    return int(iy), int(ix)

def _ensure_wgs84(gdf):
    if gdf.crs is None:
        warnings.warn("Layer without CRS—assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf

def _metric_length_m(geom):
    gseries = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    return float(gseries.length.iloc[0])

def _metric_area_m2(geom):
    gseries = gpd.GeoSeries([geom], crs=4326).to_crs(3857)
    return float(gseries.area.iloc[0])

def _init_feature_stack(ny, nx):
    acc = {}
    keys = []
    acc["roads_len_m"] = np.zeros((ny, nx), dtype=np.float64); keys.append("roads_len_m")
    for cls in ROAD_CLASS_BUCKETS:
        k = f"roads_len_m:{cls}"
        acc[k] = np.zeros((ny, nx), dtype=np.float64); keys.append(k)
    acc["building_count"] = np.zeros((ny, nx), dtype=np.float64); keys.append("building_count")
    acc["building_area_m2"] = np.zeros((ny, nx), dtype=np.float64); keys.append("building_area_m2")
    for p in PLACE_BUCKETS:
        k = f"places:{p}"
        acc[k] = np.zeros((ny, nx), dtype=np.float64); keys.append(k)
    return acc, keys

def _append_arrays(acc, keys):
    arrs = [acc[k][None, ...] for k in keys]
    return np.concatenate(arrs, axis=0), np.array(keys, dtype=object)

def _log_layer_sums(prefix, static_fx_yx, static_names):
    sums = np.nansum(static_fx_yx, axis=(1,2))
    print(f"[{prefix}] per-feature sums:")
    for nm, v in zip(static_names, sums):
        print(f"  - {nm}: {v:.3f}")
    print(f"[{prefix}] global sum: {float(np.nansum(static_fx_yx)):.3f}")

def build_static():
    meta = np.load(MERGED_META, allow_pickle=True)
    days = meta["days"]; lat = meta["lat"].astype(float); lon = meta["lon"].astype(float)
    D, F_total, NY, NX = map(int, meta["shape_4d"])
    print(f"[grid] days={D}, grid={NY}x{NX}, merged features={F_total}")

    # Build ascending edges + maps
    lat_edges, lon_edges, lat_sorted_idx, lon_sorted_idx, lat_unsort, lon_unsort = _build_edges_and_maps(lat, lon)

    acc, names_order = _init_feature_stack(NY, NX)

    # ROADS
    inside_count = 0
    if ROADS_PATH.exists():
        print(f"[roads] reading {ROADS_PATH} :: layer={ROADS_LAYER}")
        g = gpd.read_file(ROADS_PATH, layer=ROADS_LAYER)
        print(f"[roads] features: {len(g)}")
        g = _ensure_wgs84(g)
        hcol = next((c for c in g.columns if c.lower() == "highway"), None)
        for i, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            lonc, latc = geom.representative_point().x, geom.representative_point().y
            bin_idx = _bin_point(lonc, latc, lon_edges, lat_edges)
            if bin_idx is None: continue
            inside_count += 1
            iy_sorted, ix_sorted = bin_idx
            # map back to original grid order
            iy = int(lat_sorted_idx[iy_sorted])
            ix = int(lon_sorted_idx[ix_sorted])

            L = _metric_length_m(geom)
            acc["roads_len_m"][iy, ix] += L
            if hcol is not None:
                cls = str(row[hcol]).lower() if pd.notna(row[hcol]) else ""
                k = f"roads_len_m:{cls}"
                if k in acc: acc[k][iy, ix] += L
            if (i+1) % LOG_EVERY == 0:
                print(f"  roads processed: {i+1}/{len(g)}")
        print(f"  roads centroids inside grid: {inside_count}/{len(g)}")
    else:
        print(f"[roads] file not found: {ROADS_PATH}")

    # BUILDINGS
    inside_count = 0
    if BUILDINGS_PATH.exists():
        print(f"[buildings] reading {BUILDINGS_PATH} :: layer={BUILDINGS_LAYER}")
        g = gpd.read_file(BUILDINGS_PATH, layer=BUILDINGS_LAYER)
        print(f"[buildings] features: {len(g)}")
        g = _ensure_wgs84(g)
        for i, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            lonc, latc = geom.representative_point().x, geom.representative_point().y
            bin_idx = _bin_point(lonc, latc, lon_edges, lat_edges)
            if bin_idx is None: continue
            inside_count += 1
            iy_sorted, ix_sorted = bin_idx
            iy = int(lat_sorted_idx[iy_sorted])
            ix = int(lon_sorted_idx[ix_sorted])

            A = _metric_area_m2(geom)
            acc["building_count"][iy, ix] += 1.0
            acc["building_area_m2"][iy, ix] += A
            if (i+1) % (LOG_EVERY*4) == 0:
                print(f"  buildings processed: {i+1}/{len(g)}")
        print(f"  building centroids inside grid: {inside_count}/{len(g)}")
    else:
        print(f"[buildings] file not found: {BUILDINGS_PATH}")

    # PLACES
    inside_count = 0
    if PLACES_PATH.exists():
        print(f"[places] reading {PLACES_PATH} :: layer={PLACES_LAYER}")
        g = gpd.read_file(PLACES_PATH, layer=PLACES_LAYER)
        print(f"[places] features: {len(g)}")
        g = _ensure_wgs84(g)
        pcol = next((c for c in g.columns if c.lower() == "place"), None)
        for i, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type not in ("Point","MultiPoint"):
                geom = geom.representative_point()
            lonc, latc = geom.x, geom.y
            bin_idx = _bin_point(lonc, latc, lon_edges, lat_edges)
            if bin_idx is None: continue
            inside_count += 1
            iy_sorted, ix_sorted = bin_idx
            iy = int(lat_sorted_idx[iy_sorted])
            ix = int(lon_sorted_idx[ix_sorted])

            key = None
            if pcol is not None and pd.notna(row[pcol]):
                cls = str(row[pcol]).lower()
                if cls in ["city","town","village","hamlet","locality","suburb","neighbourhood"]:
                    key = f"places:{cls}"
            if key is None:
                key = "places:locality"
            acc[key][iy, ix] += 1.0
        print(f"  places inside grid: {inside_count}/{len(g)}")
    else:
        print(f"[places] file not found: {PLACES_PATH}")

    # Assemble + write
    static_fx_yx, static_keys = _append_arrays(acc, names_order)   # (F_static, NY, NX)
    static_names = np.array([f"static:{k}" for k in static_keys], dtype=object)

    print(f"[static] shape = (F_static={static_fx_yx.shape[0]}, {NY}, {NX})")
    _log_layer_sums("static", static_fx_yx, static_names)

    np.save(STATIC_3D, static_fx_yx.astype(np.float32))
    np.save(STATIC_FLAT, static_fx_yx.transpose(1,2,0).reshape(NY*NX, static_fx_yx.shape[0]).astype(np.float32))
    np.savez_compressed(
        STATIC_META,
        lat=lat, lon=lon, feature_names=static_names,
        shape_3d=np.array([static_fx_yx.shape[0], NY, NX], dtype=np.int64),
        notes=np.array({
            "mode": "centroid-bin (sorted-edge fix)",
            "roads": "meters in EPSG:3857; summed per cell",
            "buildings": "area m2 and count per cell",
            "places": "point counts per class per cell"
        }, dtype=object)
    )

    approx_gb = (static_fx_yx.shape[0] * NY * NX * 4) / (1024**3)
    card = []
    card += [ "# HOTOSM → Static grid (27×29)", "" ]
    card += [ f"- Static features: `{static_fx_yx.shape[0]}`", f"- Grid: `{NY} × {NX}`", f"- Approx raw size: ~{approx_gb:.2f} GB", "" ]
    card += [ "## Feature list" ]
    for nm in static_names:
        card.append(f"- {nm}")
    Path(STATIC_CARD).write_text("\n".join(card), encoding="utf-8")

    return static_fx_yx, static_names, (NY, NX), int(D), int(F_total)

def append_to_daily(static_fx_yx, static_names, grid_shape, D, F_total):
    NY, NX = grid_shape
    F_static = static_fx_yx.shape[0]
    cells = NY * NX

    X4 = np.memmap(MERGED_4D, dtype=np.float32, mode="r", shape=(D, F_total, NY, NX))
    OUT4 = np.memmap(PLUS_4D, dtype=np.float32, mode="w+", shape=(D, F_total + F_static, NY, NX))
    OUT4[:, :F_total, :, :] = X4
    OUT4[:, F_total:, :, :] = static_fx_yx[None, :, :, :]
    del OUT4

    X2 = np.memmap(MERGED_FLAT, dtype=np.float32, mode="r", shape=(D * cells, F_total))
    OUT2 = np.memmap(PLUS_FLAT, dtype=np.float32, mode="w+", shape=(D * cells, F_total + F_static))
    OUT2[:, :F_total] = X2
    static_flat = static_fx_yx.transpose(1,2,0).reshape(cells, F_static).astype(np.float32)
    OUT2[:, F_total:] = np.tile(static_flat, (D, 1))
    del OUT2

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
            "static_3d": str(STATIC_3D),
            "static_flat": str(STATIC_FLAT),
            "static_meta": str(STATIC_META),
        }, dtype=object),
        notes=np.array({
            "merge": "Concatenated along feature axis; static features broadcast across days."
        }, dtype=object)
    )

def main():
    static_fx_yx, static_names, grid_shape, D, F_total = build_static()
    if float(np.nansum(static_fx_yx)) == 0.0:
        print("⚠️  Still zero after fix — try EXACT_INTERSECT=True for robust apportionment.")
    if APPEND_TO_DAILY:
        append_to_daily(static_fx_yx, static_names, grid_shape, D, F_total)
    print("Done.")

if __name__ == "__main__":
    main()
