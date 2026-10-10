#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_static_humdata_27x29_winfix.py
- Same as visualize_static_humdata_27x29.py, but with Windows-safe filenames.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

ROOT = Path("./datasets")
HUM = ROOT / "humdata"
CAMS = ROOT / "CAMS"

STATIC_3D  = HUM / "static_humdata_27x29_3d.npy"
STATIC_META= HUM / "static_humdata_27x29_meta.npz"
MERGED_META= CAMS / "merged_27x29_meta.npz"

OUT = HUM / "viz_27x29"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------- helpers -------------------------

def _mesh_edges_from_centers(lat1d, lon1d):
    dlat = float(np.mean(np.diff(np.sort(lat1d))))
    dlon = float(np.mean(np.diff(np.sort(lon1d))))
    lat_edges = np.concatenate([lat1d - dlat/2, [lat1d[-1] + dlat/2]])
    lon_edges = np.concatenate([lon1d - dlon/2, [lon1d[-1] + dlon/2]])
    return lat_edges, lon_edges

def _sanitize(name: str) -> str:
    """
    Make a filesystem-safe filename component (Windows + Unix):
    - Replace invalid characters <>:\"/\\|?* and spaces/colons with '_'
    - Collapse repeated underscores and strip
    - Keep at most 100 chars
    """
    s = str(name)
    s = s.replace(" ", "_")
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)   # Windows-forbidden
    s = re.sub(r"[:]+", "_", s)            # also replace ':'
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:100] if len(s) > 100 else s

def plot_field(field2d, lat, lon, title, out_png, cmap="viridis"):
    if HAS_CARTOPY:
        lat_edges, lon_edges = _mesh_edges_from_centers(lat, lon)
        LON, LAT = np.meshgrid(lon_edges, lat_edges)
        fig = plt.figure(figsize=(8, 6), dpi=140)
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
        try:
            fig.savefig(out_png, bbox_inches="tight")
        except OSError:
            # Fallback with extra-sanitized filename
            out_png = out_png.with_name(_sanitize(out_png.stem) + ".png")
            fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None
        im = ax.imshow(field2d, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
        cb.set_label("value")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title)
        try:
            fig.savefig(out_png, bbox_inches="tight")
        except OSError:
            out_png = out_png.with_name(_sanitize(out_png.stem) + ".png")
            fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

# --------------------------- main ---------------------------

def main():
    if not STATIC_3D.exists() or not STATIC_META.exists():
        raise SystemExit("Missing static_humdata_27x29 files. Run the builder first.")
    if not MERGED_META.exists():
        raise SystemExit("Missing merged_27x29_meta.npz (to get lat/lon).")

    X = np.load(STATIC_3D)  # (F, 27, 29)
    meta = np.load(STATIC_META, allow_pickle=True)
    names = meta.get("feature_names", np.arange(X.shape[0])).astype(object)

    m = np.load(MERGED_META, allow_pickle=True)
    lat = m["lat"].astype(float)
    lon = m["lon"].astype(float)

    F, NY, NX = X.shape
    print(f"[static] features={F}, grid={NY}x{NX}")
    print(f"[grid] lat range=({lat.min():.3f}, {lat.max():.3f}), lon range=({lon.min():.3f}, {lon.max():.3f})")

    rows = []
    for k in range(F):
        nm = str(names[k])
        safe = _sanitize(nm)
        f2 = X[k]
        finite = np.isfinite(f2)
        s = dict(
            idx=k, name=nm, safe_name=safe,
            count=int(finite.sum()),
            min=float(np.nanmin(f2)) if finite.any() else np.nan,
            p50=float(np.nanpercentile(f2, 50)) if finite.any() else np.nan,
            mean=float(np.nanmean(f2)) if finite.any() else np.nan,
            max=float(np.nanmax(f2)) if finite.any() else np.nan,
            sum=float(np.nansum(f2)) if finite.any() else np.nan,
        )
        rows.append(s)
        png = OUT / f"static_{k:02d}_{safe}.png"
        plot_field(f2, lat, lon, f"Static: {nm}", png)
        if (k+1) % 5 == 0 or k == F-1:
            print(f"  wrote {k+1}/{F} maps…")

    df = pd.DataFrame.from_records(rows).sort_values("sum", ascending=False)
    df.to_csv(OUT / "static_summary.csv", index=False)
    with open(OUT / "static_summary.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print(f"✅ Saved maps and summaries in: {OUT.resolve()}")

if __name__ == "__main__":
    main()
