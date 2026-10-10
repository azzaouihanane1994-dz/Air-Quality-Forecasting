#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def edges_from_centers(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"Expected 1D centers with len>=2, got {x.shape}")
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges

def main():
    # ---- CHANGE THIS to your dataset NPZ that contains lat/lon ----
    dataset_npz = Path("./datasets/global_airpollution_2015_2024_4d.npz")
    # --------------------------------------------------------------
    out_dir = Path("./ML_3/results/windno_2")
    out_path = out_dir / "region_map_grid.png"

    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_npz.resolve()}")

    data = np.load(dataset_npz, allow_pickle=True)
    if "lat" not in data or "lon" not in data:
        raise KeyError(f"'lat'/'lon' not found in {dataset_npz}. Keys: {list(data.keys())}")

    lat_1d = np.asarray(data["lat"], dtype=float).squeeze()
    lon_1d = np.asarray(data["lon"], dtype=float).squeeze()

    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    fig = plt.figure(figsize=(7, 5.5), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333333", zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333333", zorder=10)

    # draw grid mesh
    lat_e = edges_from_centers(lat_1d)
    lon_e = edges_from_centers(lon_1d)
    LonE, LatE = np.meshgrid(lon_e, lat_e)

    ax.pcolormesh(
        LonE, LatE, np.zeros((lat_1d.size, lon_1d.size)),
        transform=ccrs.PlateCarree(),
        shading="auto",
        facecolor="none",
        edgecolor="k",
        linewidth=0.25,
        zorder=5,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    ax.set_title("Model domain and grid")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()