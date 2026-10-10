#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_alg_datasets_all_features.py

Visualize **all features** from the merged daily dataset on the 27×29 grid:

  ./datasets/CAMS/merged_27x29_4d.npy      # (days, features, 27, 29)
  ./datasets/CAMS/merged_27x29_meta.npz    # lat/lon, feature_names, days

Saves PNGs to: ./datasets/CAMS/merged_vis/

- Picks N random days (configurable) to limit output size
- For each selected day, plots **every feature**
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
MERGED_4D   = ROOT / "merged_27x29_4d.npy"
MERGED_META = ROOT / "merged_27x29_meta.npz"
OUT_DIR     = ROOT / "merged_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Random day sampling
RANDOM_SEED = 17
N_RANDOM_DAYS = 2  # Set to D to visualize all days
# ----------------------------------------------------------------


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


def _plot_map(field2d, lat, lon, title, out_png, cmap="viridis"):
    if HAS_CARTOPY:
        LAT, LON = _mesh_edges_from_centers(lat, lon)
        fig = plt.figure(figsize=(9, 7), dpi=140)
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
        cb = fig.colorbar(pm, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    else:
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
        finite = np.isfinite(field2d)
        vmin = np.nanpercentile(field2d[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(field2d[finite], 95) if finite.any() else None
        im = ax.imshow(field2d, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
        cb.set_label("value")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)


def main():
    if not (MERGED_4D.exists() and MERGED_META.exists()):
        raise SystemExit("Missing merged dataset files. Expected merged_27x29_4d.npy and merged_27x29_meta.npz.")

    m = np.load(MERGED_META, allow_pickle=True)
    days = m["days"].astype("datetime64[ns]")
    lat  = m["lat"].astype(float)    # (27,)
    lon  = m["lon"].astype(float)    # (29,)
    names = m["feature_names"].astype(str)
    D, F, Y, X = map(int, m["shape_4d"])

    print(f"[meta] days={D}, features={F}, grid={Y}x{X}")
    X4 = np.memmap(MERGED_4D, dtype=np.float32, mode="r", shape=(D, F, Y, X))

    # Use all features
    wanted = list(range(F))

    # pick random days (or all if N_RANDOM_DAYS >= D)
    rng = np.random.default_rng(RANDOM_SEED)
    day_idx = np.arange(D) if D <= N_RANDOM_DAYS else np.sort(rng.choice(D, N_RANDOM_DAYS, replace=False))

    print(f"[plan] Visualizing ALL {F} features on {len(day_idx)} day(s).")
    print(f"[out]  {OUT_DIR}")

    count = 0
    for di in day_idx:
        date_str = str(days[di])[:10]
        for j in wanted:
            field = np.asarray(X4[di, j], dtype=np.float32)  # (Y, X)
            safe = names[j].replace(":", "_").replace("/", "-").replace(" ", "_")
            title = f"Merged — {date_str} — {names[j]}"
            out = OUT_DIR / f"merged_{di:04d}_{date_str}__f{j:03d}_{safe}.png"
            _plot_map(field, lat, lon, title, out)
            count += 1
            if count % 50 == 0:
                print(f"  wrote {count} figures…")

    print(f"✅ Done. {count} figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
