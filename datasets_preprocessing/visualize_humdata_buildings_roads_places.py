#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_humdata_buildings_roads_places.py

Visualize HOTOSM layers in ./datasets/humdata:
- Vector quicklooks (sampled rows)
- Grid heatmaps on the 195x210 CAMS/EAC4 grid:
    buildings: count per cell
    roads:     total length per cell (meters)
    places:    count per cell

Inputs (hardcoded):
  ./datasets/humdata/hotosm_dza_buildings_polygons_gpkg.gpkg
  ./datasets/humdata/hotosm_dza_roads_lines_gpkg.gpkg
  ./datasets/humdata/hotosm_dza_populated_places_points_gpkg.gpkg
  ./datasets/CAMS/merged_daily_meta.npz  (lat, lon for the 195x210 grid)

Outputs:
  ./datasets/humdata/viz/
      buildings_quicklook.png
      roads_quicklook.png
      places_quicklook.png
      buildings_count_heatmap.png
      roads_length_m_heatmap.png
      places_count_heatmap.png

Dependencies:
  pip install geopandas shapely pyogrio pandas numpy
  # (optional) cartopy for nicer basemaps
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

# Optional: Cartopy for georeferenced plotting
HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

# ------------------- Hardcoded paths -------------------
ROOT = Path("./datasets")
HUM = ROOT / "humdata"
CAMS = ROOT / "CAMS"

GPKG_BUILDINGS = HUM / "hotosm_dza_buildings_polygons_gpkg.gpkg"
GPKG_ROADS     = HUM / "hotosm_dza_roads_lines_gpkg.gpkg"
GPKG_PLACES    = HUM / "hotosm_dza_populated_places_points_gpkg.gpkg"

MERGED_META = CAMS / "merged_daily_meta.npz"  # provides lat/lon grid (195x210)

OUT = HUM / "viz"
OUT.mkdir(parents=True, exist_ok=True)

# Sampling for vector quicklooks (to keep plots light)
MAX_ROWS_QUICKLOOK = 5000

# -------------------------------------------------------

def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn("Layer without CRS—assuming EPSG:4326")
        return gdf.set_crs(4326)
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(4326)
    return gdf

def _mesh_edges_from_centers(lat1d, lon1d):
    """Return (lat_edges, lon_edges) for pcolormesh from 1D centers (assumes regular spacing)."""
    dlat = float(np.mean(np.diff(np.sort(lat1d))))
    dlon = float(np.mean(np.diff(np.sort(lon1d))))
    lat_edges = np.concatenate([lat1d - dlat/2, [lat1d[-1] + dlat/2]])
    lon_edges = np.concatenate([lon1d - dlon/2, [lon1d[-1] + dlon/2]])
    return lat_edges, lon_edges

def _cell_index_from_lonlat(lon, lat, lon_edges, lat_edges):
    ix = np.searchsorted(lon_edges, lon, side="right") - 1
    iy = np.searchsorted(lat_edges, lat, side="right") - 1
    if ix < 0 or ix >= lon_edges.size-1: return None
    if iy < 0 or iy >= lat_edges.size-1: return None
    return int(iy), int(ix)

def _plot_pcolormesh(field2d, lat, lon, title, out_png, cmap="viridis"):
    if HAS_CARTOPY:
        # Build edges
        lat_edges, lon_edges = _mesh_edges_from_centers(lat, lon)
        LON, LAT = np.meshgrid(lon_edges, lat_edges)
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)

        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None

        pm = ax.pcolormesh(LON, LAT, field2d, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(pm, ax=ax, shrink=0.92, pad=0.03)
        cb.set_label("value")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None
        im = ax.imshow(field2d, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
        cb.set_label("value")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

def _plot_vectors_quicklook(gdf, title, out_png, sample=MAX_ROWS_QUICKLOOK, color="#204a87"):
    gdf = _ensure_wgs84(gdf)
    if len(gdf) > sample:
        gdf = gdf.sample(sample, random_state=42)
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())
        try:
            bounds = gdf.total_bounds
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())
        except Exception:
            pass
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=0.4, markersize=3, color=color)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        gdf.plot(ax=ax, linewidth=0.4, markersize=3, color=color)
        ax.set_title(title)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

def _metric_length_m(geom) -> float:
    # project to WebMercator for length in meters (fast & good enough)
    return float(gpd.GeoSeries([geom], crs=4326).to_crs(3857).length.iloc[0])

def main():
    # 1) Load grid from merged meta
    m = np.load(MERGED_META, allow_pickle=True)
    lat = m["lat"].astype(float)    # (195,)
    lon = m["lon"].astype(float)    # (210,)
    NY, NX = lat.shape[0], lon.shape[0]
    lat_edges, lon_edges = _mesh_edges_from_centers(lat, lon)
    print(f"[grid] Grid loaded: {NY}x{NX} (WGS84)")

    # 2) Vector quicklooks
    if GPKG_BUILDINGS.exists():
        print("[vector] Buildings quicklook…")
        gb = gpd.read_file(GPKG_BUILDINGS, rows=slice(0, max(MAX_ROWS_QUICKLOOK, 10000)))
        _plot_vectors_quicklook(gb, "Buildings (sample quicklook)", OUT/"buildings_quicklook.png", color="#8f5902")
    if GPKG_ROADS.exists():
        print("[vector] Roads quicklook…")
        gr = gpd.read_file(GPKG_ROADS, rows=slice(0, max(MAX_ROWS_QUICKLOOK, 10000)))
        _plot_vectors_quicklook(gr, "Roads (sample quicklook)", OUT/"roads_quicklook.png", color="#204a87")
    if GPKG_PLACES.exists():
        print("[vector] Places quicklook…")
        gp = gpd.read_file(GPKG_PLACES, rows=slice(0, max(MAX_ROWS_QUICKLOOK, 10000)))
        _plot_vectors_quicklook(gp, "Populated places (sample quicklook)", OUT/"places_quicklook.png", color="#a40000")

    # 3) Grid heatmaps (centroid binning for speed)
    # Buildings: count per cell
    if GPKG_BUILDINGS.exists():
        print("[grid] Buildings → count per cell…")
        gb = _ensure_wgs84(gpd.read_file(GPKG_BUILDINGS))
        bldg_count = np.zeros((NY, NX), dtype=np.float64)
        for geom in gb.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                rp = geom.representative_point()
                iy_ix = _cell_index_from_lonlat(rp.x, rp.y, lon_edges, lat_edges)
            except Exception:
                iy_ix = None
            if iy_ix is None:
                continue
            iy, ix = iy_ix
            bldg_count[iy, ix] += 1.0
        _plot_pcolormesh(bldg_count, lat, lon, "Buildings — count per cell", OUT/"buildings_count_heatmap.png", cmap="YlOrBr")

    # Roads: total length per cell (meters)
    if GPKG_ROADS.exists():
        print("[grid] Roads → total length per cell (meters)…")
        gr = _ensure_wgs84(gpd.read_file(GPKG_ROADS))
        roads_len = np.zeros((NY, NX), dtype=np.float64)
        for geom in gr.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                rp = geom.representative_point()
                iy_ix = _cell_index_from_lonlat(rp.x, rp.y, lon_edges, lat_edges)
            except Exception:
                iy_ix = None
            if iy_ix is None:
                continue
            iy, ix = iy_ix
            roads_len[iy, ix] += _metric_length_m(geom)
        _plot_pcolormesh(roads_len, lat, lon, "Roads — total length per cell (m)", OUT/"roads_length_m_heatmap.png", cmap="PuBu")

    # Places: count per cell
    if GPKG_PLACES.exists():
        print("[grid] Places → count per cell…")
        gp = _ensure_wgs84(gpd.read_file(GPKG_PLACES))
        places_count = np.zeros((NY, NX), dtype=np.float64)
        for geom in gp.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type not in ("Point", "MultiPoint"):
                try:
                    geom = geom.representative_point()
                except Exception:
                    continue
            iy_ix = _cell_index_from_lonlat(geom.x, geom.y, lon_edges, lat_edges)
            if iy_ix is None:
                continue
            iy, ix = iy_ix
            places_count[iy, ix] += 1.0
        _plot_pcolormesh(places_count, lat, lon, "Populated places — count per cell", OUT/"places_count_heatmap.png", cmap="Greens")

    print(f"✅ Figures saved to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
