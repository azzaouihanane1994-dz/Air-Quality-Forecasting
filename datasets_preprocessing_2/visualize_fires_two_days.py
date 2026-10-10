#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_fires_two_days_maplines_bold.py

Same as the previous map visualizer, but with **bold, high-contrast map lines**:
- Thick coastlines & borders (linewidth=1.2–1.6, edgecolor="#111")
- Light land/ocean fills for contrast
- Optional graticule (lat/lon gridlines)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

CAMS = Path("./datasets/CAMS")
FIRE_4D = CAMS / "FIRE_ALG_27x29_4d.npy"
FIRE_META = CAMS / "FIRE_ALG_27x29_meta.npz"
OUT = CAMS / "fire_vis_map"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 7

def _mesh_edges_from_centers(lat_1d, lon_1d):
    lat_1d = np.asarray(lat_1d, dtype=float)
    lon_1d = np.asarray(lon_1d, dtype=float)
    dlat = float(np.mean(np.diff(np.sort(lat_1d))))
    dlon = float(np.mean(np.diff(np.sort(lon_1d))))
    lat_edges = np.concatenate([lat_1d - dlat/2, [lat_1d[-1] + dlat/2]])
    lon_edges = np.concatenate([lon_1d - dlon/2, [lon_1d[-1] + dlon/2]])
    return lat_edges, lon_edges

def _plot_map(field2d, lat, lon, title, out_png, vmin=None, vmax=None):
    lat_edges, lon_edges = _mesh_edges_from_centers(lat, lon)
    LON, LAT = np.meshgrid(lon_edges, lat_edges)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(9, 7), dpi=140)
        ax = plt.axes(projection=ccrs.PlateCarree())
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Light background for contrast
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e9f2ff", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"),  facecolor="#faf9f5", edgecolor="none", zorder=0)

        # Draw data first, underneath the lines, so lines stay crisp on top
        pm = ax.pcolormesh(LON, LAT, field2d, shading="auto", vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), zorder=3)

        # Bold coastlines/borders on top
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#111", linewidth=1.6, zorder=8)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"),   edgecolor="#111", linewidth=1.2, zorder=8)

        # Optional graticule
        gl = ax.gridlines(draw_labels=False, linewidth=0.6, color="#555", alpha=0.4, linestyle="--", zorder=7)
        cb = fig.colorbar(pm, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_title(title)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        # Fallback without cartopy (no political outlines available)
        fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
        pm = ax.pcolormesh(LON, LAT, field2d, shading="auto", vmin=vmin, vmax=vmax, zorder=1)
        cb = fig.colorbar(pm, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_title(title)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

def main():
    if not (FIRE_4D.exists() and FIRE_META.exists()):
        raise SystemExit("Missing fire files. Expected FIRE_ALG_27x29_4d.npy and FIRE_ALG_27x29_meta.npz in ./datasets/CAMS/")

    X = np.load(FIRE_4D, mmap_mode="r")  # (D, 2, 27, 29)
    m = np.load(FIRE_META, allow_pickle=True)
    days = m["days"].astype("datetime64[ns]")
    lat = m["lat"].astype(float)
    lon = m["lon"].astype(float)

    D = int(X.shape[0]); F = int(X.shape[1])
    assert F == 2, "Expected 2 features: [fire_label, fire_frp]"
    print(f"[meta] days={D}, grid={lat.size}x{lon.size}, cartopy={'yes' if HAS_CARTOPY else 'no'}")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(D, size=2, replace=False); idx.sort()
    print("[pick] day indices:", idx, "=>", [str(days[i])[:10] for i in idx])

    for di in idx:
        date_str = str(days[di])[:10]
        label = np.array(X[di, 0])
        frp   = np.array(X[di, 1])

        _plot_map(label, lat, lon, f"Fires present (0/1) — {date_str}",
                  OUT / f"fire_map_{di:04d}_{date_str}_label.png", vmin=0, vmax=1)

        finite = np.isfinite(frp) & (frp > 0)
        if finite.any():
            vmin = float(np.nanpercentile(frp[finite], 5))
            vmax = float(np.nanpercentile(frp[finite], 95))
        else:
            vmin = None; vmax = None
        _plot_map(frp, lat, lon, f"FRP (sum) — {date_str}",
                  OUT / f"fire_map_{di:04d}_{date_str}_frp.png", vmin=vmin, vmax=vmax)

    print(f"✅ Saved figures in {OUT.resolve()}")

if __name__ == "__main__":
    main()
