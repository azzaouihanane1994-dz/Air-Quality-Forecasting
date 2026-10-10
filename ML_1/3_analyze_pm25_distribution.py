#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze PM2.5 distribution and pollution fractions from
global_airpollution_2015_2024_flat.npz (no train/test split).

Outputs a text summary:
    pm25_analysis_summary.txt

Generates plots:
    pm25_hist_all.png
    pm25_daily_fraction_all.png
    Random-day pollution maps (PM2.5 + polluted cells) in ./datasets/pollution_samples
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional geoviz stack (same idea as visualize_alg_datasets.py)
HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

NPZ_FLAT = Path("./datasets/global_airpollution_2015_2024_flat.npz")
SUMMARY_TXT = Path("./datasets/pm25_analysis_summary.txt")

# thresholds in µg/m³
THRESHOLDS_UGM3 = [15.0, 25.0, 35.0]
MAIN_THRESHOLD = 25.0  # main "polluted" threshold

# Plots
MAKE_PLOTS = True
RANDOM_SEED = 17
N_RANDOM_DAYS = 5  # how many random days to plot

# Directory to store pollution sample maps
POLLUTION_SAMPLE_DIR = Path("./datasets/pollution_samples")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def add(lines, text=""):
    lines.append(str(text))


def auto_convert_to_ugm3(pm25_values, target_units_hint="auto"):
    """
    Try to decide whether PM2.5 is in kg/m³ or µg/m³.

    Heuristic:
      - If 95th percentile < 1e-3, assume kg/m³ (values ~1e-7 etc.) and convert to µg/m³.
      - Otherwise assume already in µg/m³.

    Returns:
      pm25_ug, units_string
    """
    pm25 = np.asarray(pm25_values).astype(float)
    q95 = np.nanpercentile(pm25, 95)

    if target_units_hint == "kg/m3":
        factor = 1e9
        return pm25 * factor, "kg/m³ -> µg/m³ (forced)"

    if target_units_hint == "ug/m3":
        return pm25, "µg/m³ (forced)"

    # auto
    if q95 < 1e-3:
        # Almost certainly kg/m³
        factor = 1e9
        return pm25 * factor, f"kg/m³ -> µg/m³ (auto, q95={q95:.2e})"
    else:
        return pm25, f"µg/m³ (auto, q95={q95:.2f})"


def daily_stats(pm25_vals, dates, thr):
    """
    pm25_vals: [N_samples] in µg/m³
    dates:     [N_samples] datetime64[D], aligned with pm25_vals
    Returns:
      daily_mean: Series(date -> mean PM2.5)
      daily_frac: Series(date -> fraction of polluted cells)
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pm25": pm25_vals
    })
    df["polluted"] = (df["pm25"] >= thr).astype(int)

    grp = df.groupby("date")
    daily_mean = grp["pm25"].mean()
    daily_frac = grp["polluted"].mean()
    return daily_mean, daily_frac


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


def _plot_pollution_map(pm25_map, polluted_mask, lat_1d, lon_1d,
                        label_name, date_str, frac, thr_main, out_png):
    """
    Plot PM2.5 map + polluted mask (PM2.5 >= thr_main) with coastlines/borders if Cartopy is available.
    """
    H, W = pm25_map.shape

    if HAS_CARTOPY:
        LAT, LON = _mesh_edges_from_centers(lat_1d, lon_1d)
        fig = plt.figure(figsize=(10, 4), dpi=140)

        # PM2.5 map
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        extent = [float(lon_1d.min()), float(lon_1d.max()),
                  float(lat_1d.min()), float(lat_1d.max())]
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb",
                        edgecolor="none", zorder=0)
        ax1.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5",
                        edgecolor="none", zorder=0)
        ax1.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6,
                        edgecolor="#333", zorder=10)
        ax1.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6,
                        edgecolor="#333", zorder=10)

        finite = np.isfinite(pm25_map)
        vmin = np.nanpercentile(pm25_map[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(pm25_map[finite], 95) if finite.any() else None

        pm = ax1.pcolormesh(LON, LAT, pm25_map, cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            shading="auto", transform=ccrs.PlateCarree(), zorder=5)
        cb1 = fig.colorbar(pm, ax=ax1, shrink=0.9, pad=0.03)
        cb1.set_label("PM2.5 [µg/m³]")
        ax1.set_title(f"{label_name} {date_str}\nPM2.5")

        # Polluted mask
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb",
                        edgecolor="none", zorder=0)
        ax2.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5",
                        edgecolor="none", zorder=0)
        ax2.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6,
                        edgecolor="#333", zorder=10)
        ax2.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6,
                        edgecolor="#333", zorder=10)

        mask_plot = ax2.pcolormesh(LON, LAT, polluted_mask, cmap="Reds",
                                   vmin=0, vmax=1,
                                   shading="auto", transform=ccrs.PlateCarree(), zorder=5)
        cb2 = fig.colorbar(mask_plot, ax=ax2, shrink=0.9, pad=0.03)
        cb2.set_label("Polluted (1) / Clean (0)")
        ax2.set_title(f"Polluted cells (PM2.5 ≥ {thr_main} µg/m³)\nfrac={frac:.3f}")

        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    else:
        # Fallback without Cartopy (simple imshow)
        extent = [float(lon_1d.min()), float(lon_1d.max()),
                  float(lat_1d.min()), float(lat_1d.max())]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)

        finite = np.isfinite(pm25_map)
        vmin = np.nanpercentile(pm25_map[finite], 5) if finite.any() else None
        vmax = np.nanpercentile(pm25_map[finite], 95) if finite.any() else None

        im1 = axes[0].imshow(pm25_map, origin="lower", extent=extent,
                             cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        cb1 = fig.colorbar(im1, ax=axes[0], shrink=0.9, pad=0.03)
        cb1.set_label("PM2.5 [µg/m³]")
        axes[0].set_title(f"{label_name} {date_str}\nPM2.5")
        axes[0].set_xlabel("lon")
        axes[0].set_ylabel("lat")

        im2 = axes[1].imshow(polluted_mask, origin="lower", extent=extent,
                             cmap="Reds", vmin=0, vmax=1, aspect="auto")
        cb2 = fig.colorbar(im2, ax=axes[1], shrink=0.9, pad=0.03)
        cb2.set_label("Polluted (1) / Clean (0)")
        axes[1].set_title(f"Polluted cells (PM2.5 ≥ {thr_main} µg/m³)\nfrac={frac:.3f}")
        axes[1].set_xlabel("lon")
        axes[1].set_ylabel("lat")

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def generate_random_pollution_maps(pm25_vals, dates, daily_frac,
                                   lat_1d, lon_1d,
                                   label_name,
                                   thr_main, H, W, out_dir,
                                   n_random_days=5, lines=None, rng=None):
    """
    Generate maps of PM2.5 and polluted cells for random days.

    pm25_vals:  flat array [N_samples] (µg/m³)
    dates:      flat array [N_samples] (datetime64[D]) aligned with pm25_vals
    daily_frac: pd.Series indexed by Timestamp, giving fraction of polluted cells
    lat_1d, lon_1d: 1D arrays of grid centers
    label_name: e.g. "All"
    thr_main:   threshold in µg/m³
    H, W:       grid dimensions
    out_dir:    directory to save images
    n_random_days: how many random days to choose
    rng:        numpy random Generator
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    all_days = daily_frac.index.to_numpy()
    if len(all_days) == 0:
        if lines is not None:
            add(lines, f"No days available in daily_frac; skipping maps.")
        return

    n_choose = min(n_random_days, len(all_days))
    chosen_days = np.sort(rng.choice(all_days, size=n_choose, replace=False))

    if lines is not None:
        add(lines, f"\nRandom {n_choose} days for pollution maps ({label_name}, thr={thr_main} µg/m³):")
        for d in chosen_days:
            frac = float(daily_frac.loc[d])
            add(lines, f"  {pd.Timestamp(d).date()} -> polluted fraction = {frac:.3f}")

    for d in chosen_days:
        ts = pd.Timestamp(d)
        d_date = ts.date()
        d_np = np.datetime64(d_date)

        mask_day = (dates == d_np)
        pm25_day = pm25_vals[mask_day]

        if pm25_day.size != H * W:
            if lines is not None:
                add(lines, f"  WARNING: for {label_name} {d_date}, "
                           f"expected {H*W} cells, got {pm25_day.size}. Skipping map.")
            continue

        pm25_map = pm25_day.reshape(H, W)
        polluted_mask = (pm25_map >= thr_main).astype(int)
        frac = float(daily_frac.loc[d]) if d in daily_frac.index else polluted_mask.mean()

        date_str = d_date.isoformat()
        fname = f"{label_name.lower()}_polluted_map_{date_str}.png"
        fpath = out_dir / fname

        _plot_pollution_map(pm25_map, polluted_mask, lat_1d, lon_1d,
                            label_name, date_str, frac, thr_main, fpath)

        if lines is not None:
            add(lines, f"  Saved pollution map: {fpath}")


# ---------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------

def analyze_pm25():
    lines = []
    add(lines, "PM2.5 ANALYSIS (NO TRAIN/TEST SPLIT)")
    add(lines, "=" * 80)
    add(lines)

    if not NPZ_FLAT.exists():
        add(lines, f"File not found: {NPZ_FLAT}")
        report = "\n".join(lines)
        print(report)
        SUMMARY_TXT.write_text(report, encoding="utf-8")
        return

    data = np.load(NPZ_FLAT, allow_pickle=True)

    # New flat arrays (no split)
    X_flat = data["X_flat"]
    y_reg_flat = data["y_reg_flat"]
    y_class_flat = data["y_class_flat"]
    dates_flat = data["dates_flat"].astype("datetime64[D]")
    lat_flat = data["lat_flat"]
    lon_flat = data["lon_flat"]

    feature_names = [str(f) for f in data["feature_names"]]
    target_names = [str(t) for t in data["target_names"]]
    stored_threshold = float(data["pm25_threshold"])
    grid_shape_4d = data["grid_shape_4d"]  # [T, F, H, W]
    H, W = int(grid_shape_4d[-2]), int(grid_shape_4d[-1])

    # Recover 1D lat/lon from flat metadata
    lat_1d = np.unique(lat_flat)
    lon_1d = np.unique(lon_flat)
    assert lat_1d.size == H and lon_1d.size == W, \
        f"Recovered lat/lon sizes ({lat_1d.size}, {lon_1d.size}) != (H,W)=({H},{W})"

    add(lines, f"Loaded: {NPZ_FLAT.resolve()}")
    add(lines, f"Targets: {target_names}")
    add(lines, f"Stored pm25_threshold in file: {stored_threshold} (unit depends on prep)")
    add(lines, f"Grid: H={H}, W={W}, cells={H*W}")
    add(lines)
    add(lines, f"Cartopy available: {HAS_CARTOPY}")
    add(lines)

    # -----------------------------------------------------------------
    # Extract PM2.5 from regression targets
    # -----------------------------------------------------------------
    try:
        pm25_idx = target_names.index("eac4:pm2p5")
    except ValueError:
        raise SystemExit("Target 'eac4:pm2p5' not found in target_names")

    pm25_raw = y_reg_flat[:, pm25_idx]

    pm25_ug, units_info = auto_convert_to_ugm3(pm25_raw, target_units_hint="auto")

    add(lines, "PM2.5 units inference:")
    add(lines, f"  All samples: {units_info}")
    add(lines)

    # -----------------------------------------------------------------
    # Basic distribution stats
    # -----------------------------------------------------------------
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    q_all = np.nanpercentile(pm25_ug, qs)

    add(lines, "PM2.5 distribution (all samples) [µg/m³]:")
    for p, qv in zip(qs, q_all):
        add(lines, f"  {p:>3}th percentile: {qv:8.3f}")
    add(lines)

    # -----------------------------------------------------------------
    # Classification stats for multiple thresholds (using pm25_ug)
    # -----------------------------------------------------------------
    def cls_stats_for_threshold(pm25_vals, thr, label_name):
        y = (pm25_vals >= thr).astype(int)
        n = y.size
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return f"{label_name}: thr={thr} µg/m³ -> polluted={pos} ({pos/n:.4f}), clean={neg} ({neg/n:.4f})"

    add(lines, "Classification stats (recomputed from PM2.5 in µg/m³):")
    for thr in THRESHOLDS_UGM3:
        add(lines, "  " + cls_stats_for_threshold(pm25_ug, thr, "All"))
    add(lines)

    # -----------------------------------------------------------------
    # Daily stats: mean PM2.5 and fraction of polluted cells per day
    # -----------------------------------------------------------------
    thr_main = MAIN_THRESHOLD

    daily_mean_all, daily_frac_all = daily_stats(pm25_ug, dates_flat, thr_main)

    add(lines, f"Daily mean PM2.5 [µg/m³] (All, threshold={thr_main}):")
    add(lines, f"  mean of daily means: {daily_mean_all.mean():.3f}")
    add(lines, f"  min/median/max: {daily_mean_all.min():.3f} / "
               f"{daily_mean_all.median():.3f} / {daily_mean_all.max():.3f}")
    add(lines)

    add(lines, f"Daily fraction of polluted cells (All, thr={thr_main} µg/m³):")
    add(lines, f"  mean fraction:       {daily_frac_all.mean():.3f}")
    add(lines, f"  min/median/max:      {daily_frac_all.min():.3f} / "
               f"{daily_frac_all.median():.3f} / {daily_frac_all.max():.3f}")
    add(lines)

    # -----------------------------------------------------------------
    # Compare with existing binary labels in file (sanity)
    # -----------------------------------------------------------------
    y_class_existing = np.asarray(y_class_flat).astype(int)

    def compare_with_existing_labels(y_existing, pm25_vals, thr):
        new_labels = (pm25_vals >= thr).astype(int)
        if y_existing.shape != new_labels.shape:
            return f"shape mismatch between existing and new labels: " \
                   f"{y_existing.shape} vs {new_labels.shape}"
        same = (new_labels == y_existing).sum()
        n = new_labels.size
        diff = n - same
        return f"agreement with existing labels at thr={thr} µg/m³: " \
               f"{same}/{n} ({same/n:.4f}), different={diff}"

    add(lines)
    add(lines, f"Agreement with existing classification in file (assuming thr={thr_main} µg/m³):")
    add(lines, "  " + compare_with_existing_labels(y_class_existing, pm25_ug, thr_main))

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    if MAKE_PLOTS:
        # Histogram of PM2.5 (all samples)
        plt.figure()
        plt.hist(pm25_ug, bins=100)
        plt.xlabel("PM2.5 [µg/m³]")
        plt.ylabel("Frequency")
        plt.title("PM2.5 distribution (all samples)")
        plt.tight_layout()
        plt.savefig("pm25_hist_all.png", dpi=150)
        plt.close()

        # Daily fraction of polluted cells (all)
        plt.figure()
        daily_frac_all.sort_index().plot()
        plt.ylabel(f"Fraction of cells PM2.5 ≥ {thr_main} µg/m³")
        plt.xlabel("Date")
        plt.title("Daily fraction of polluted cells (all)")
        plt.tight_layout()
        plt.savefig("pm25_daily_fraction_all.png", dpi=150)
        plt.close()

        add(lines)
        add(lines, "Plots saved:")
        add(lines, "  pm25_hist_all.png")
        add(lines, "  pm25_daily_fraction_all.png")

        rng = np.random.default_rng(RANDOM_SEED)

        # Random-day pollution maps for the whole dataset
        generate_random_pollution_maps(
            pm25_vals=pm25_ug,
            dates=dates_flat,
            daily_frac=daily_frac_all,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            label_name="All",
            thr_main=thr_main,
            H=H,
            W=W,
            out_dir=POLLUTION_SAMPLE_DIR,
            n_random_days=N_RANDOM_DAYS,
            lines=lines,
            rng=rng,
        )

        add(lines, f"Random pollution sample maps saved under: {POLLUTION_SAMPLE_DIR.resolve()}")

    # -----------------------------------------------------------------
    # Write + print report
    # -----------------------------------------------------------------
    report = "\n".join(lines)
    print(report)
    SUMMARY_TXT.write_text(report, encoding="utf-8")
    print(f"\nSummary written to: {SUMMARY_TXT.resolve()}")


if __name__ == "__main__":
    analyze_pm25()
