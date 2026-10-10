#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 -build_and_append_static_humdata_27x29_fixed_v3.py
Build gridded static roads, buildings and populated-place features.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from shapely.geometry import box

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

LOG_EVERY = 200000
MAJOR = {"motorway","trunk","primary"}
SECONDARY = {"secondary","tertiary"}
LOCAL = {"residential","service","track"}
PLACE_KEEP = {"city","town","village"}

def _build_edges_and_maps(lat_centers, lon_centers):
    lat_centers = np.asarray(lat_centers).astype(float)
    lon_centers = np.asarray(lon_centers).astype(float)
    lat_sorted_idx = np.argsort(lat_centers)
    lon_sorted_idx = np.argsort(lon_centers)
    lat_sorted = lat_centers[lat_sorted_idx]
    lon_sorted = lon_centers[lon_sorted_idx]
    dlat = float(np.mean(np.diff(lat_sorted)))
    dlon = float(np.mean(np.diff(lon_sorted)))
    lat_edges = np.concatenate([lat_sorted - dlat/2, [lat_sorted[-1] + dlat/2]])
    lon_edges = np.concatenate([lon_sorted - dlon/2, [lon_sorted[-1] + dlon/2]])
    return lat_edges, lon_edges, lat_sorted_idx, lon_sorted_idx, dlat, dlon

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

def _cell_area_m2(lat_centers, dlat_deg, dlon_deg):
    R = 6371008.8
    dlat = np.deg2rad(abs(dlat_deg))
    dlon = np.deg2rad(abs(dlon_deg))
    lat_rad = np.deg2rad(lat_centers.astype(float))
    height_m = R * dlat
    width_m = R * dlon * np.cos(lat_rad)
    area = height_m * width_m
    return area

def build_static():
    meta = np.load(MERGED_META, allow_pickle=True)
    days = meta["days"]; lat = meta["lat"].astype(float); lon = meta["lon"].astype(float)
    D, F_total, NY, NX = map(int, meta["shape_4d"])
    print(f"[grid] days={D}, grid={NY}x{NX}, merged features={F_total}")
    lat_edges, lon_edges, lat_sorted_idx, lon_sorted_idx, dlat_deg, dlon_deg = _build_edges_and_maps(lat, lon)
    roads_major_m = np.zeros((NY, NX), dtype=np.float64)
    roads_secondary_m = np.zeros((NY, NX), dtype=np.float64)
    roads_local_m = np.zeros((NY, NX), dtype=np.float64)
    building_area_m2 = np.zeros((NY, NX), dtype=np.float64)
    places_city = np.zeros((NY, NX), dtype=np.float64)
    places_town = np.zeros((NY, NX), dtype=np.float64)
    places_village = np.zeros((NY, NX), dtype=np.float64)

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
            iy_sorted, ix_sorted = bin_idx
            iy = int(lat_sorted_idx[iy_sorted]); ix = int(lon_sorted_idx[ix_sorted])
            L = _metric_length_m(geom)
            cls = str(row[hcol]).lower() if (hcol is not None and pd.notna(row[hcol])) else ""
            if cls in MAJOR: roads_major_m[iy, ix] += L
            elif cls in SECONDARY: roads_secondary_m[iy, ix] += L
            elif cls in LOCAL: roads_local_m[iy, ix] += L
            if (i+1) % LOG_EVERY == 0: print(f"  roads processed: {i+1}/{len(g)}")

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
            iy_sorted, ix_sorted = bin_idx
            iy = int(lat_sorted_idx[iy_sorted]); ix = int(lon_sorted_idx[ix_sorted])
            A = _metric_area_m2(geom)
            building_area_m2[iy, ix] += A
            if (i+1) % (LOG_EVERY*4) == 0: print(f"  buildings processed: {i+1}/{len(g)}")

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
            iy_sorted, ix_sorted = bin_idx
            iy = int(lat_sorted_idx[iy_sorted]); ix = int(lon_sorted_idx[ix_sorted])
            key = str(row[pcol]).lower() if (pcol is not None and pd.notna(row[pcol])) else None
            if key == "city": places_city[iy, ix] += 1.0
            elif key == "town": places_town[iy, ix] += 1.0
            elif key == "village": places_village[iy, ix] += 1.0

    # Convert to densities
    area_strip = _cell_area_m2(lat, dlat_deg, dlon_deg)   # (NY,)
    area_m2 = np.repeat(area_strip[:, None], NX, axis=1)  # (NY, NX)
    with np.errstate(divide='ignore', invalid='ignore'):
        roads_major = roads_major_m * (1000.0 / area_m2)
        roads_secondary = roads_secondary_m * (1000.0 / area_m2)
        roads_local = roads_local_m * (1000.0 / area_m2)

    layers = [
        ("static:roads_major_km_per_km2", roads_major),
        ("static:roads_secondary_km_per_km2", roads_secondary),
        ("static:roads_local_km_per_km2", roads_local),
        ("static:building_area_m2", building_area_m2),
        ("static:places:city", places_city),
        ("static:places:town", places_town),
        ("static:places:village", places_village),
    ]
    names = [nm for nm, _ in layers]
    cube = np.stack([arr for _, arr in layers], axis=0).astype(np.float32)

    print(f"[static] shape = (F={cube.shape[0]}, {cube.shape[1]}, {cube.shape[2]})")
    sums = np.nansum(cube, axis=(1,2))
    for nm, v in zip(names, sums):
        print(f"  - {nm}: {v:.3f}")
    print(f"[static] global sum: {float(np.nansum(cube)):.3f}")

    np.save(STATIC_3D, cube)
    np.save(STATIC_FLAT, cube.transpose(1,2,0).reshape(NY*NX, cube.shape[0]))
    np.savez_compressed(
        STATIC_META,
        lat=lat, lon=lon, feature_names=np.array(names, dtype=object),
        shape_3d=np.array([cube.shape[0], NY, NX], dtype=np.int64),
        notes=np.array({
            "roads": "density km/km^2 using grouped OSM classes",
            "buildings": "sum of footprint area per cell (m^2)",
            "places": "counts per cell (city, town, village)",
            "binning": "centroid with sorted-edge fix",
        }, dtype=object)
    )

    approx_gb = (cube.shape[0] * NY * NX * 4) / (1024**3)
    card = []
    card += [ "# HOTOSM → Static grid (27×29) — grouped & densities", "" ]
    card += [ f"- Static features: `{cube.shape[0]}`", f"- Grid: `{NY} × {NX}`", f"- Approx raw size: ~{approx_gb:.2f} GB", "" ]
    card += [ "## Feature list" ]
    for nm in names:
        card.append(f"- {nm}")
    Path(STATIC_CARD).write_text("\n".join(card), encoding="utf-8")

    return cube, np.array(names, dtype=object), (NY, NX), int(D), int(F_total)

def append_to_daily(cube, names, grid_shape, D, F_total):
    NY, NX = grid_shape
    F_static = cube.shape[0]
    cells = NY * NX
    X4 = np.memmap(MERGED_4D, dtype=np.float32, mode="r", shape=(D, F_total, NY, NX))
    OUT4 = np.memmap(PLUS_4D, dtype=np.float32, mode="w+", shape=(D, F_total + F_static, NY, NX))
    OUT4[:, :F_total, :, :] = X4
    OUT4[:, F_total:, :, :] = cube[None, :, :, :]
    del OUT4
    X2 = np.memmap(MERGED_FLAT, dtype=np.float32, mode="r", shape=(D * cells, F_total))
    OUT2 = np.memmap(PLUS_FLAT, dtype=np.float32, mode="w+", shape=(D * cells, F_total + F_static))
    OUT2[:, :F_total] = X2
    static_flat = cube.transpose(1,2,0).reshape(cells, F_static).astype(np.float32)
    OUT2[:, F_total:] = np.tile(static_flat, (D, 1))
    del OUT2
    merged_meta = np.load(MERGED_META, allow_pickle=True)
    lat, lon = merged_meta["lat"], merged_meta["lon"]
    days = merged_meta["days"]
    names_merged = merged_meta["feature_names"].astype(object)
    names_plus = np.concatenate([names_merged, names])
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
    cube, names, grid_shape, D, F_total = build_static()
    if APPEND_TO_DAILY:
        append_to_daily(cube, names, grid_shape, D, F_total)
    print("Done.")

if __name__ == "__main__":
    main()
