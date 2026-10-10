#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_merged_daily_4d.py

Visualize samples from the merged daily dataset:
  ./datasets/CAMS/merged_daily_4d.npy      # (days, features, 195, 210)
  ./datasets/CAMS/merged_daily_meta.npz    # lat/lon, feature_names, days

Saves PNGs to: ./datasets/CAMS/merged_vis/

- Picks N random days (configurable)
- Matches features by substring patterns (configurable)
- Uses Cartopy for georeferenced pcolormesh if available; falls back to imshow
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Optional geoviz stack
HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

# ----------------------- hard-coded paths -----------------------
ROOT = Path("./datasets/CAMS")
MERGED_4D   = ROOT / "merged_daily_4d.npy"
MERGED_META = ROOT / "merged_daily_meta.npz"
OUT_DIR     = ROOT / "merged_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature name substring patterns (case-insensitive)
PATTERNS = [
    "eac4:o3", "ozone",
    "eac4:no2", "nitrogen dioxide",
    "eac4:so2", "sulphur dioxide",
    "eac4:co", "carbon monoxide",
    "eac4:pm", "pm2.5", "pm10",
    "emis:nitrogen-oxides", "emis:sulphur-dioxide",
    "emis:black-carbon", "emis:organic-carbon",
    "emis:carbon-monoxide", "emis:methane",
    "emis:isoprene", "emis:monoterpenes",
]

# Colormap hints by keyword (fallback = "Blues")
CMAP_HINTS = {
    "o3": "Blues", "ozone": "Blues",
    "no2": "OrRd", "nitrogen dioxide": "OrRd",
    "so2": "Purples", "sulphur": "Purples",
    "pm": "YlOrBr",
    "co": "PuBu",
    "emis": "Greens",
}

# Random day sampling
RANDOM_SEED = 17
N_RANDOM_DAYS = 6
# ----------------------------------------------------------------


def _pick_cmap(name: str) -> str:
    low = name.lower()
    for k, cmap in CMAP_HINTS.items():
        if k in low:
            return cmap
    return "Blues"


def _mesh_edges_from_centers(lat_1d: np.ndarray, lon_1d: np.ndarray):
    """
    Build 2D lon/lat edge grids (nlat+1, nlon+1) for pcolormesh from 1D centers.
    Assumes regular spacing.
    """
    dlat = float(np.mean(np.diff(np.sort(lat_1d))))
    dlon = float(np.mean(np.diff(np.sort(lon_1d))))
    lat_edges = np.concatenate([lat_1d - dlat/2, [lat_1d[-1] + dlat/2]])
    lon_edges = np.concatenate([lon_1d - dlon/2, [lon_1d[-1] + dlon/2]])
    LON, LAT = np.meshgrid(lon_edges, lat_edges)  # (nlat+1, nlon+1)
    return LAT, LON


def _plot_map(field2d, lat, lon, title, out_png, cmap="Blues"):
    if HAS_CARTOPY:
        LAT, LON = _mesh_edges_from_centers(lat, lon)
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Background + extent
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)

        # Robust color stretch
        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None

        pm = ax.pcolormesh(LON, LAT, field2d, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(pm, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        # Fallback: imshow with geographic extent
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None
        im = ax.imshow(field2d, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)


def main():
    if not (MERGED_4D.exists() and MERGED_META.exists()):
        raise SystemExit("Missing merged dataset files. Expected merged_daily_4d.npy and merged_daily_meta.npz.")

    m = np.load(MERGED_META, allow_pickle=True)
    days = m["days"].astype("datetime64[ns]")
    lat  = m["lat"].astype(float)    # (195,)
    lon  = m["lon"].astype(float)    # (210,)
    names = m["feature_names"].astype(str)
    D, F, Y, X = map(int, m["shape_4d"])

    print(f"[meta] days={D}, features={F}, grid={Y}x{X}")
    X4 = np.memmap(MERGED_4D, dtype=np.float32, mode="r", shape=(D, F, Y, X))

    # resolve feature indices by substring
    wanted = []
    for pat in PATTERNS:
        pl = pat.lower()
        for j, nm in enumerate(names):
            if pl in nm.lower():
                wanted.append(j)
    wanted = sorted(set(wanted))
    if not wanted:
        print("[warn] No features matched PATTERNS. Showing first 6 features instead.")
        wanted = list(range(min(6, F)))

    # pick random days
    rng = np.random.default_rng(RANDOM_SEED)
    day_idx = np.arange(D) if D <= N_RANDOM_DAYS else np.sort(rng.choice(D, N_RANDOM_DAYS, replace=False))

    print(f"[plan] Visualizing {len(wanted)} features on {len(day_idx)} random days.")
    print(f"[out]  {OUT_DIR}")

    count = 0
    for di in day_idx:
        date_str = str(days[di])[:10]
        for j in wanted:
            field = np.asarray(X4[di, j], dtype=np.float32)  # (Y, X)
            title = f"Merged — {date_str} — {names[j]}"
            safe = names[j].replace(":", "_").replace("/", "-")
            out = OUT_DIR / f"merged_{di:04d}_{date_str}__f{j:03d}_{safe}.png"
            _plot_map(field, lat, lon, title, out, cmap=_pick_cmap(names[j]))
            count += 1
            if count % 10 == 0:
                print(f"  wrote {count} figures…")

    print(f"✅ Done. {count} figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
