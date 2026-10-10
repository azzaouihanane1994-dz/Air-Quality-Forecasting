#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WindNO v2: Wind-conditioned Neural Operator + Hierarchical AQI + Full Artifacts
==============================================================================
Backbone novelty:
  - ConvGRU temporal encoder on grid (no node-flatten GRU)
  - latent semi-Lagrangian advection via grid_sample conditioned on wind
  - learnable diffusion in latent space
  - Fourier Neural Operator (FNO) blocks for global mixing

Tasks:
  - Regression: PM2.5, PM10, O3, NO2 (4 channels)
  - Binary: PM2.5 exceedance
  - AQI: hierarchical 2-stage with weighted focal / label smoothing + ordinal consistency (asymmetric)

Artifacts (similar to your old script):
  - best checkpoint + training log JSON
  - loss curves
  - regression scatter + error hist
  - binary confusion + ROC + PR
  - AQI confusion + per-class metrics JSON
  - region map + random geo maps
  - test_predictions_grid.npz
  - multi_horizon_metrics_{mode}.csv/.json + summary plots + per-h confusion matrices
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Optional Cartopy for geo maps
HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    HAS_CARTOPY = False

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
})


# ============================== CONFIG ===================================== #

@dataclass
class Config:
    # Data
    npz_path_4d: str = "./datasets/global_airpollution_2015_2024_4d.npz"
    train_end_year: int = 2023

    # History & horizon
    history: int = 7
    horizon: int = 1

    # Multi-horizon evaluation (rollout with h=1 model)
    eval_horizons: Tuple[int, ...] = (1, 3, 7, 14, 30)
    multi_horizon_exo_modes: Tuple[str, ...] = ("oracle", "persist")
    multi_horizon_max_samples: Optional[int] = None
    multi_horizon_batch_size: Optional[int] = None

    # Model
    hidden_dim: int = 64
    convgru_kernel: int = 3

    # Advection/Diffusion
    adv_scale: float = 0.15      # converts wind to pixel-ish displacement
    wind_speed_scale: float = 10.0  # only for optional normalization (kept for parity)
    nu_max: float = 0.25

    # FNO
    fno_blocks: int = 4
    fno_modes1: int = 16
    fno_modes2: int = 16

    # Loss weights
    lambda_reg: float = 1.0
    lambda_cls_bin: float = 1.0
    lambda_aqi: float = 1.0

    # AQI stage weights (inside lambda_aqi)
    lambda_aqi_group: float = 1.0
    lambda_aqi_fine: float = 2.0
    lambda_aqi_ordinal: float = 0.5

    # AQI loss configuration (stage losses)
    # options: "ce", "weighted_ce", "focal", "weighted_focal", "label_smoothing", "weighted_label_smoothing"
    aqi_loss_type: str = "weighted_focal"
    focal_gamma: float = 2.0
    label_smoothing_eps: float = 0.1

    # Fine(2,3) bias multipliers (mapped local 0->AQI2, 1->AQI3)
    fine23_weight_mult_class2: float = 1.2
    fine23_weight_mult_class3: float = 1.0

    # Ordinal asymmetry for threshold k=2 (y>2)
    ordinal_asym_2_to_3: float = 1.15
    ordinal_asym_3_to_2: float = 1.0

    # Inference calibration
    coarse_temperature: float = 1.5

    # Post-hoc margins
    aqi_posthoc_margin_down_32: float = 0.0      # if pred=3 but barely above 2 -> set 2
    aqi_posthoc_margin_23_up: float = 0.02       # if pred=2 but 3 is close -> set 3
    aqi_posthoc_margin_10_down: float = 0.0075   # if pred=1 but 0 is close -> set 0

    # Optional Kendall uncertainty weighting (reg/bin/aqi)
    use_uncertainty_weighting: bool = True

    # Training
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0

    # Checkpoint selection
    best_checkpoint_metric: str = "aqi_macro_f1"  # or "test_loss"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Output dirs
    ckpt_dir: Path = Path("./ML_3/checkpoints/windno_2")
    log_dir: Path = Path("./ML_3/logs/windno_2")
    plot_dir: Path = Path("./ML_3/results/windno_2")

    # Geo sampling
    n_random_test_days: int = 10
    random_seed: int = 42


cfg = Config()
print("Device:", cfg.device)
print("Cartopy available:", HAS_CARTOPY)


# ============================== UTILS ====================================== #

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_wind_indices(feature_names: List[str]) -> Dict[str, Optional[int]]:
    fn = list(feature_names)

    def find_any(names: List[str]) -> Optional[int]:
        for n in names:
            if n in fn:
                return fn.index(n)
        return None

    u10_idx = find_any(["eac4:u10", "u10"])
    v10_idx = find_any(["eac4:v10", "v10"])
    u100_idx = find_any(["u100", "eac4:u100"])
    v100_idx = find_any(["v100", "eac4:v100"])

    if (u10_idx is None or v10_idx is None) and (u100_idx is None or v100_idx is None):
        raise ValueError(
            "Could not find any 10 m or 100 m wind components in feature_names. "
            "Expected eac4:u10/eac4:v10 or u100/v100."
        )

    return {"u10": u10_idx, "v10": v10_idx, "u100": u100_idx, "v100": v100_idx}


def mesh_edges_from_centers(lat_1d: np.ndarray, lon_1d: np.ndarray):
    lat_1d = np.asarray(lat_1d, dtype=float)
    lon_1d = np.asarray(lon_1d, dtype=float)

    dlat = float(np.mean(np.diff(np.sort(lat_1d)))) if lat_1d.size > 1 else 0.5
    dlon = float(np.mean(np.diff(np.sort(lon_1d)))) if lon_1d.size > 1 else 0.5

    lat_edges = np.concatenate([lat_1d - dlat / 2, [lat_1d[-1] + dlat / 2]])
    lon_edges = np.concatenate([lon_1d - dlon / 2, [lon_1d[-1] + dlon / 2]])
    LON, LAT = np.meshgrid(lon_edges, lat_edges)
    return LAT, LON


# ============================== DATASET ==================================== #

class SpatioTemporalAirDataset(Dataset):
    """
    Same IO format as your old model:
      Input:  X_seq_full [L, F+4, H, W]  (exo + past pollution)
      Targets:
        y_reg [4,H,W]
        y_cls [1,H,W]
        y_aqi [1,H,W] int 0..3
    """

    def __init__(self, npz_path: str, split: str, history: int, horizon: int, train_end_year: int):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)

        self.X_4d = data["X_4d"].astype(np.float32)
        self.y_reg_4d = data["y_reg_4d"].astype(np.float32)
        self.y_cls_4d = data["y_class_4d"].astype(np.float32)
        self.y_aqi_4d = data["y_aqi_4d"].astype(np.int64)
        self.dates_4d = data["dates_4d"]
        self.feature_names = list(data["feature_names"])
        self.target_names = list(data["target_names"])
        self.aqi_class_names = list(data["aqi_class_names"]) if "aqi_class_names" in data.files else [
            "good", "moderate", "unhealthy", "very_unhealthy_or_worse"
        ]

        self.lat_1d = data["lat"] if "lat" in data.files else None
        self.lon_1d = data["lon"] if "lon" in data.files else None

        self.T, self.F, self.H, self.W = self.X_4d.shape
        self.C = self.y_reg_4d.shape[1]
        self.history = history
        self.horizon = horizon
        self.train_end_year = train_end_year

        years = pd.to_datetime(self.dates_4d).year.values
        all_t_in_end = np.arange(history - 1, self.T - horizon, dtype=int)
        t_out = all_t_in_end + horizon

        if split == "train":
            mask = years[t_out] <= train_end_year
        elif split == "test":
            mask = years[t_out] > train_end_year
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.t_in_end_indices = all_t_in_end[mask]
        self.t_out_indices = t_out[mask]

        if len(self.t_in_end_indices) == 0:
            raise ValueError(f"No {split} samples found. Check split config.")

    def __len__(self) -> int:
        return len(self.t_in_end_indices)

    def __getitem__(self, idx: int):
        t_in_end = self.t_in_end_indices[idx]
        t_out = self.t_out_indices[idx]
        t_in_start = t_in_end - self.history + 1

        X_seq = self.X_4d[t_in_start:t_in_end + 1]          # [L,F,H,W]
        y_reg_seq = self.y_reg_4d[t_in_start:t_in_end + 1]  # [L,4,H,W]
        X_seq_full = np.concatenate([X_seq, y_reg_seq], axis=1).astype(np.float32)  # [L,F+4,H,W]

        y_reg = self.y_reg_4d[t_out].astype(np.float32)
        y_cls = self.y_cls_4d[t_out].astype(np.float32)
        y_aqi = self.y_aqi_4d[t_out].astype(np.int64)
        date_out = np.datetime_as_string(self.dates_4d[t_out], unit="D")

        X_seq_full = np.nan_to_num(X_seq_full, nan=0.0, posinf=0.0, neginf=0.0)
        y_reg = np.nan_to_num(y_reg, nan=0.0, posinf=0.0, neginf=0.0)
        y_cls = np.nan_to_num(y_cls, nan=0.0, posinf=0.0, neginf=0.0)
        y_aqi = np.nan_to_num(y_aqi, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(X_seq_full).float(),
            torch.from_numpy(y_reg).float(),
            torch.from_numpy(y_cls).float(),
            torch.from_numpy(y_aqi).long(),
            date_out
        )


# ============================== AQI WEIGHTS/LOSSES ========================== #

def compute_aqi_class_weights(y_aqi_4d, dates_4d, train_end_year, num_classes):
    years = pd.to_datetime(dates_4d).year.values
    mask = years <= train_end_year
    y_train = y_aqi_4d[mask].reshape(-1)
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        freq = np.maximum(counts / total, 1e-6)
        inv = 1.0 / freq
        weights = (inv / inv.mean()).astype(np.float32)
    return weights, counts.astype(np.int64)


def build_weights_from_counts(counts_sub):
    counts_sub = np.asarray(counts_sub, dtype=np.float64)
    total = counts_sub.sum()
    if total <= 0:
        return np.ones_like(counts_sub, dtype=np.float32)
    freq = np.maximum(counts_sub / total, 1e-6)
    inv = 1.0 / freq
    w = inv / inv.mean()
    return w.astype(np.float32)


def focal_loss_multi_class(logits, targets, alpha=None, gamma=2.0):
    B, C, H, W = logits.shape
    logits_2d = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets_1d = targets.reshape(-1)

    logp = F.log_softmax(logits_2d, dim=1)
    p = logp.exp()
    idx = torch.arange(logits_2d.size(0), device=logits.device)

    logpt = logp[idx, targets_1d]
    pt = p[idx, targets_1d]

    if alpha is not None:
        at = alpha[targets_1d]
        loss = -at * (1 - pt) ** gamma * logpt
    else:
        loss = -(1 - pt) ** gamma * logpt
    return loss.mean()


def label_smoothing_ce(logits, targets, class_weights=None, smoothing=0.1):
    B, C, H, W = logits.shape
    logits_2d = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets_1d = targets.reshape(-1)
    logp = F.log_softmax(logits_2d, dim=1)

    smooth = torch.zeros_like(logp)
    for c in range(C):
        m = targets_1d == c
        if not m.any():
            continue
        n_neigh = (1 if c > 0 else 0) + (1 if c < C - 1 else 0)
        if n_neigh == 0:
            smooth[m, c] = 1.0
        else:
            smooth[m, c] = 1.0 - smoothing
            share = smoothing / n_neigh
            if c > 0:
                smooth[m, c - 1] += share
            if c < C - 1:
                smooth[m, c + 1] += share

    loss = -(smooth * logp).sum(dim=1)
    if class_weights is not None:
        loss = loss * class_weights[targets_1d]
    return loss.mean()


def make_multi_class_loss_fn(aqi_loss_type, weights_np, focal_gamma, smoothing_eps, device):
    if weights_np is not None:
        cw = torch.tensor(weights_np, device=device, dtype=torch.float32)
        alpha = cw / cw.mean()
    else:
        cw = None
        alpha = None

    def loss_fn(logits, targets):
        if aqi_loss_type == "ce":
            return F.cross_entropy(logits, targets)
        if aqi_loss_type == "weighted_ce":
            return F.cross_entropy(logits, targets, weight=cw)
        if aqi_loss_type == "focal":
            return focal_loss_multi_class(logits, targets, alpha=None, gamma=focal_gamma)
        if aqi_loss_type == "weighted_focal":
            return focal_loss_multi_class(logits, targets, alpha=alpha, gamma=focal_gamma)
        if aqi_loss_type == "label_smoothing":
            return label_smoothing_ce(logits, targets, class_weights=None, smoothing=smoothing_eps)
        if aqi_loss_type == "weighted_label_smoothing":
            return label_smoothing_ce(logits, targets, class_weights=cw, smoothing=smoothing_eps)
        raise ValueError(f"Unknown aqi_loss_type={aqi_loss_type!r}")
    return loss_fn


def ordinal_consistency_loss_from_probs(p4, y_true, *, w_2_to_3=1.0, w_3_to_2=1.0):
    eps = 1e-6
    p4 = torch.clamp(p4, eps, 1.0 - eps)

    q0 = p4[:, 1:].sum(dim=1, keepdim=True)  # y>0
    q1 = p4[:, 2:].sum(dim=1, keepdim=True)  # y>1
    q2 = p4[:, 3:].sum(dim=1, keepdim=True)  # y>2
    q = torch.cat([q0, q1, q2], dim=1)       # [B,3,H,W]

    y = y_true.unsqueeze(1)  # [B,1,H,W]
    t0 = (y > 0).float()
    t1 = (y > 1).float()
    t2 = (y > 2).float()
    t = torch.cat([t0, t1, t2], dim=1)

    bce = -(t * torch.log(q) + (1.0 - t) * torch.log(1.0 - q))  # [B,3,H,W]

    if (w_2_to_3 != 1.0) or (w_3_to_2 != 1.0):
        w = torch.ones_like(bce)
        # apply asymmetry only to k=2 threshold channel
        w[:, 2:3] = w[:, 2:3] * torch.where(
            (y == 2),
            torch.tensor(float(w_2_to_3), device=bce.device),
            torch.tensor(1.0, device=bce.device),
        )
        w[:, 2:3] = w[:, 2:3] * torch.where(
            (y == 3),
            torch.tensor(float(w_3_to_2), device=bce.device),
            torch.tensor(1.0, device=bce.device),
        )
        bce = bce * w

    return bce.mean()


# ============================== METRICS ==================================== #

def compute_regression_metrics(y_true, y_pred):
    eps = 1e-8
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0) + eps)
    return mae, rmse


def compute_binary_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    return float(acc), float(prec), float(rec), float(f1), cm


def compute_multiclass_confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> float:
    eps = 1e-8
    f1s = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
    return float(np.mean(f1s)) if len(f1s) else 0.0


# ============================== PLOTTING =================================== #

def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path, title: str, normalize: bool = False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_plot = cm.astype(np.float64)
    if normalize:
        cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 5), dpi=140)
    plt.imshow(cm_plot, origin="upper", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_region_outline(lat_1d, lon_1d, out_path: Path):
    lat_1d = np.asarray(lat_1d, dtype=float)
    lon_1d = np.asarray(lon_1d, dtype=float)
    extent = [lon_1d.min(), lon_1d.max(), lat_1d.min(), lat_1d.max()]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(6, 5), dpi=140)
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
        ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
        ax.set_title("Model domain & grid")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Model domain (no Cartopy)")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_geo_example(true_pm25, pred_pm25,
                     true_cls, pred_cls_prob,
                     true_aqi, pred_aqi_class, aqi_class_names,
                     lat_1d, lon_1d, date_str, idx, out_dir: Path):
    H, W = true_pm25.shape
    extent = [float(lon_1d.min()), float(lon_1d.max()), float(lat_1d.min()), float(lat_1d.max())]

    polluted_true = true_cls.astype(int)
    polluted_pred = (pred_cls_prob >= 0.5).astype(int)

    vmin = float(min(true_pm25.min(), pred_pm25.min()))
    vmax = float(max(true_pm25.max(), pred_pm25.max()))

    n_aqi = len(aqi_class_names)
    cmap_aqi = "tab10"

    if HAS_CARTOPY:
        LAT, LON = mesh_edges_from_centers(lat_1d, lon_1d)
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(11, 10), dpi=140)

        def ax_geo(pos, title):
            ax = plt.subplot(3, 2, pos, projection=proj)
            ax.set_extent(extent, crs=proj)
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1eb", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dfe8f5", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#333", zorder=10)
            ax.set_title(title)
            return ax

        ax1 = ax_geo(1, f"True PM2.5\n{date_str}")
        m1 = ax1.pcolormesh(LON, LAT, true_pm25, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto", transform=proj)
        fig.colorbar(m1, ax=ax1, shrink=0.9, pad=0.03)

        ax2 = ax_geo(2, "Predicted PM2.5")
        m2 = ax2.pcolormesh(LON, LAT, pred_pm25, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto", transform=proj)
        fig.colorbar(m2, ax=ax2, shrink=0.9, pad=0.03)

        ax3 = ax_geo(3, "True polluted mask")
        m3 = ax3.pcolormesh(LON, LAT, polluted_true, cmap="Reds", vmin=0, vmax=1, shading="auto", transform=proj)
        fig.colorbar(m3, ax=ax3, shrink=0.9, pad=0.03)

        ax4 = ax_geo(4, "Pred polluted mask")
        m4 = ax4.pcolormesh(LON, LAT, polluted_pred, cmap="Reds", vmin=0, vmax=1, shading="auto", transform=proj)
        fig.colorbar(m4, ax=ax4, shrink=0.9, pad=0.03)

        ax5 = ax_geo(5, "True AQI")
        m5 = ax5.pcolormesh(LON, LAT, true_aqi, cmap=cmap_aqi, vmin=-0.5, vmax=n_aqi - 0.5, shading="auto", transform=proj)
        cb5 = fig.colorbar(m5, ax=ax5, shrink=0.9, pad=0.03, ticks=range(n_aqi))
        cb5.ax.set_yticklabels(aqi_class_names)

        ax6 = ax_geo(6, "Pred AQI")
        m6 = ax6.pcolormesh(LON, LAT, pred_aqi_class, cmap=cmap_aqi, vmin=-0.5, vmax=n_aqi - 0.5, shading="auto", transform=proj)
        cb6 = fig.colorbar(m6, ax=ax6, shrink=0.9, pad=0.03, ticks=range(n_aqi))
        cb6.ax.set_yticklabels(aqi_class_names)

        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(11, 10), dpi=140)
        ax1 = plt.subplot(3, 2, 1); im1 = ax1.imshow(true_pm25, origin="lower"); plt.colorbar(im1, ax=ax1); ax1.set_title(f"True PM2.5\n{date_str}")
        ax2 = plt.subplot(3, 2, 2); im2 = ax2.imshow(pred_pm25, origin="lower"); plt.colorbar(im2, ax=ax2); ax2.set_title("Predicted PM2.5")
        ax3 = plt.subplot(3, 2, 3); im3 = ax3.imshow(polluted_true, origin="lower", vmin=0, vmax=1); plt.colorbar(im3, ax=ax3); ax3.set_title("True polluted mask")
        ax4 = plt.subplot(3, 2, 4); im4 = ax4.imshow(polluted_pred, origin="lower", vmin=0, vmax=1); plt.colorbar(im4, ax=ax4); ax4.set_title("Pred polluted mask")
        ax5 = plt.subplot(3, 2, 5); im5 = ax5.imshow(true_aqi, origin="lower", vmin=-0.5, vmax=n_aqi - 0.5, cmap=cmap_aqi); cb5 = plt.colorbar(im5, ax=ax5, ticks=range(n_aqi)); cb5.ax.set_yticklabels(aqi_class_names); ax5.set_title("True AQI")
        ax6 = plt.subplot(3, 2, 6); im6 = ax6.imshow(pred_aqi_class, origin="lower", vmin=-0.5, vmax=n_aqi - 0.5, cmap=cmap_aqi); cb6 = plt.colorbar(im6, ax=ax6, ticks=range(n_aqi)); cb6.ax.set_yticklabels(aqi_class_names); ax6.set_title("Pred AQI")
        fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"example_geo_maps_sample_{idx}_{date_str}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================== MODEL (WindNO) ============================== #

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.hidden_dim = hidden_dim
        self.conv_zr = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel, padding=pad)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel, padding=pad)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        cat = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(cat))
        z, r = torch.chunk(zr, 2, dim=1)

        cat_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(cat_r))
        return (1 - z) * h + z * h_tilde


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weights_ri: torch.Tensor) -> torch.Tensor:
        w = torch.view_as_complex(weights_ri)
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x) + self.w(x)
        x = self.norm(x)
        return F.gelu(x)


class AdvectionDiffusionBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, nu_max: float):
        super().__init__()
        self.fno = FNOBlock(width, modes1, modes2)

        self.nu_max = nu_max
        self.nu_logit = nn.Parameter(torch.tensor(-2.0))

        lap = torch.tensor([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)

        self.post = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.norm = nn.GroupNorm(8, width)

    def _advect(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, adv_scale: float) -> torch.Tensor:
        B, C, H, W = h.shape
        device = h.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

        dx = adv_scale * u.squeeze(1)
        dy = adv_scale * v.squeeze(1)
        dx_norm = 2.0 * dx / max(W - 1, 1)
        dy_norm = 2.0 * dy / max(H - 1, 1)

        grid = base.clone()
        grid[..., 0] = grid[..., 0] - dx_norm
        grid[..., 1] = grid[..., 1] - dy_norm

        return F.grid_sample(h, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _diffuse(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        nu = torch.sigmoid(self.nu_logit) * self.nu_max
        k = self.lap_kernel.repeat(C, 1, 1, 1)
        lap_h = F.conv2d(h, k, padding=1, groups=C)
        return h + nu * lap_h

    def forward(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, adv_scale: float) -> torch.Tensor:
        h_adv = self._advect(h, u, v, adv_scale=adv_scale)
        h_diff = self._diffuse(h_adv)
        h_fno = self.fno(h_diff)
        out = h + self.post(h_fno)
        out = self.norm(out)
        return F.gelu(out)


class WindNO(nn.Module):
    """
    Outputs:
      y_reg_pred: [B,4,H,W]
      y_cls_logits: [B,1,H,W]
      y_grp_logits: [B,2,H,W]
      y_fine_g0_logits: [B,2,H,W]
      y_fine_g1_logits: [B,2,H,W]
    """
    def __init__(self, F_exo: int, C_targets: int, hidden_dim: int, wind_idx: Dict[str, Optional[int]]):
        super().__init__()
        self.F_exo = F_exo
        self.C_targets = C_targets
        self.wind_idx = wind_idx

        self.in_proj = nn.Sequential(
            nn.Conv2d(F_exo + C_targets, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.gru = ConvGRUCell(hidden_dim, hidden_dim, kernel=cfg.convgru_kernel)

        self.blocks = nn.ModuleList([
            AdvectionDiffusionBlock(hidden_dim, cfg.fno_modes1, cfg.fno_modes2, nu_max=cfg.nu_max)
            for _ in range(cfg.fno_blocks)
        ])

        self.head_reg = nn.Conv2d(hidden_dim, C_targets, 1)
        self.head_cls_bin = nn.Conv2d(hidden_dim, 1, 1)

        self.head_aqi_group = nn.Conv2d(hidden_dim, 2, 1)
        self.head_aqi_fine_g0 = nn.Conv2d(hidden_dim, 2, 1)
        self.head_aqi_fine_g1 = nn.Conv2d(hidden_dim, 2, 1)

    def extract_combined_wind(self, X_seq_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X_seq_full: [B,L,F_exo+C_targets,H,W]
        use wind from the last day exogenous channels only.
        """
        B, L, Ffull, H, W = X_seq_full.shape
        X_last_exo = X_seq_full[:, -1, :self.F_exo]  # [B,F_exo,H,W]

        u_stack, v_stack = [], []

        def try_add(iu: Optional[int], iv: Optional[int]):
            if iu is not None and iv is not None:
                u_stack.append(X_last_exo[:, iu:iu+1])
                v_stack.append(X_last_exo[:, iv:iv+1])

        try_add(self.wind_idx.get("u10"), self.wind_idx.get("v10"))
        try_add(self.wind_idx.get("u100"), self.wind_idx.get("v100"))

        if len(u_stack) == 0:
            raise RuntimeError("No wind components available.")

        u = torch.stack(u_stack, dim=0).mean(dim=0)
        v = torch.stack(v_stack, dim=0).mean(dim=0)
        return u, v

    def forward(self, X_seq_full: torch.Tensor):
        B, L, Ffull, H, W = X_seq_full.shape
        h = None
        for t in range(L):
            z = self.in_proj(X_seq_full[:, t])
            h = self.gru(z, h)

        u, v = self.extract_combined_wind(X_seq_full)
        for blk in self.blocks:
            h = blk(h, u, v, adv_scale=cfg.adv_scale)

        y_reg_pred = self.head_reg(h)
        y_cls_logits = self.head_cls_bin(h)
        y_grp_logits = self.head_aqi_group(h)
        y_fine_g0_logits = self.head_aqi_fine_g0(h)
        y_fine_g1_logits = self.head_aqi_fine_g1(h)

        return y_reg_pred, y_cls_logits, y_grp_logits, y_fine_g0_logits, y_fine_g1_logits


# ============================== AQI INFERENCE =============================== #

def hier_aqi_probs_and_pred(y_grp_logits, y_fine_g0_logits, y_fine_g1_logits, temp: float):
    temp = float(max(temp, 1e-6))
    p_grp = torch.softmax(y_grp_logits / temp, dim=1)
    p_g0 = torch.softmax(y_fine_g0_logits, dim=1)
    p_g1 = torch.softmax(y_fine_g1_logits, dim=1)

    p0 = p_grp[:, 0:1] * p_g0[:, 0:1]
    p1 = p_grp[:, 0:1] * p_g0[:, 1:2]
    p2 = p_grp[:, 1:2] * p_g1[:, 0:1]
    p3 = p_grp[:, 1:2] * p_g1[:, 1:2]
    p4 = torch.cat([p0, p1, p2, p3], dim=1)
    pred = torch.argmax(p4, dim=1)

    # Post-hoc margins
    if cfg.aqi_posthoc_margin_down_32 > 0:
        diff32 = p4[:, 3] - p4[:, 2]
        pred = torch.where((pred == 3) & (diff32 < cfg.aqi_posthoc_margin_down_32),
                           torch.full_like(pred, 2), pred)

    if cfg.aqi_posthoc_margin_23_up > 0:
        diff23 = p4[:, 2] - p4[:, 3]
        pred = torch.where((pred == 2) & (diff23 < cfg.aqi_posthoc_margin_23_up),
                           torch.full_like(pred, 3), pred)

    if cfg.aqi_posthoc_margin_10_down > 0:
        diff10 = p4[:, 1] - p4[:, 0]
        pred = torch.where((pred == 1) & (diff10 < cfg.aqi_posthoc_margin_10_down),
                           torch.full_like(pred, 0), pred)

    return p4, pred.long()


# ============================== TRAIN/EVAL ================================= #

def train_and_evaluate():
    set_seed(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    meta = np.load(cfg.npz_path_4d, allow_pickle=True)
    feature_names = list(meta["feature_names"])
    target_names = list(meta["target_names"])
    aqi_class_names = list(meta["aqi_class_names"]) if "aqi_class_names" in meta.files else [
        "good", "moderate", "unhealthy", "very_unhealthy_or_worse"
    ]
    lat_1d = meta["lat"] if "lat" in meta.files else None
    lon_1d = meta["lon"] if "lon" in meta.files else None

    F_exo = len(feature_names)
    C_targets = len(target_names)
    num_aqi_classes = len(aqi_class_names)

    wind_idx = find_wind_indices(feature_names)
    print("Using wind indices:", {k: (feature_names[v] if v is not None else None) for k, v in wind_idx.items()})

    # AQI weights (train years)
    aqi_w4, aqi_counts = compute_aqi_class_weights(meta["y_aqi_4d"], meta["dates_4d"], cfg.train_end_year, num_aqi_classes)
    print("AQI class counts:", dict(zip(range(num_aqi_classes), aqi_counts)))
    print("AQI class weights:", dict(zip(range(num_aqi_classes), aqi_w4)))

    group_counts = np.array([aqi_counts[0] + aqi_counts[1], aqi_counts[2] + aqi_counts[3]], dtype=np.int64)
    group_w = build_weights_from_counts(group_counts)

    fine01_counts = np.array([aqi_counts[0], aqi_counts[1]], dtype=np.int64)
    fine01_w = build_weights_from_counts(fine01_counts)

    fine23_counts = np.array([aqi_counts[2], aqi_counts[3]], dtype=np.int64)
    fine23_w = build_weights_from_counts(fine23_counts).copy()
    # apply bias multipliers then renormalize
    fine23_w[0] *= float(cfg.fine23_weight_mult_class2)
    fine23_w[1] *= float(cfg.fine23_weight_mult_class3)
    fine23_w = fine23_w / max(fine23_w.mean(), 1e-9)

    loss_grp = make_multi_class_loss_fn(cfg.aqi_loss_type, group_w, cfg.focal_gamma, cfg.label_smoothing_eps, device)
    loss_fine_g0 = make_multi_class_loss_fn(cfg.aqi_loss_type, fine01_w, cfg.focal_gamma, cfg.label_smoothing_eps, device)
    loss_fine_g1 = make_multi_class_loss_fn(cfg.aqi_loss_type, fine23_w, cfg.focal_gamma, cfg.label_smoothing_eps, device)

    train_ds = SpatioTemporalAirDataset(cfg.npz_path_4d, "train", cfg.history, cfg.horizon, cfg.train_end_year)
    test_ds = SpatioTemporalAirDataset(cfg.npz_path_4d, "test", cfg.history, cfg.horizon, cfg.train_end_year)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = WindNO(F_exo=F_exo, C_targets=C_targets, hidden_dim=cfg.hidden_dim, wind_idx=wind_idx).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional Kendall uncertainty weighting params
    if cfg.use_uncertainty_weighting:
        log_vars = nn.Parameter(torch.zeros(3, device=device))
        optimizer.add_param_group({"params": [log_vars]})
    else:
        log_vars = None

    reg_loss_fn = nn.MSELoss()
    bin_loss_fn = nn.BCEWithLogitsLoss()

    history_log = {
        "epoch": [],
        "train_loss": [],
        "train_reg_loss": [],
        "train_cls_bin_loss": [],
        "train_aqi_loss": [],
        "test_loss": [],
        "test_reg_loss": [],
        "test_cls_bin_loss": [],
        "test_aqi_loss": [],
        "aqi_macro_f1": [],
        "target_names": target_names,
        "aqi_class_names": aqi_class_names,
    }

    best_score = -float("inf") if cfg.best_checkpoint_metric == "aqi_macro_f1" else float("inf")
    best_ckpt_path = cfg.ckpt_dir / "windno_v2_best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        tr_reg_sum = tr_bin_sum = tr_aqi_sum = 0.0
        tr_n = 0

        for X_seq, y_reg, y_cls, y_aqi, _ in train_loader:
            X_seq = torch.nan_to_num(X_seq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            y_reg = torch.nan_to_num(y_reg.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            y_cls = torch.nan_to_num(y_cls.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            y_aqi = torch.nan_to_num(y_aqi.to(device), nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)

            y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(X_seq)

            loss_reg = reg_loss_fn(y_reg_pred, y_reg)
            loss_bin = bin_loss_fn(y_cls_logits, y_cls)

            y_aqi_t = y_aqi.squeeze(1).long()
            y_grp_t = (y_aqi_t >= 2).long()

            loss_grp_val = loss_grp(y_grp_logits, y_grp_t)

            # fine g0 (0 vs 1)
            mask0 = (y_grp_t == 0)
            if mask0.any():
                logits0 = y_g0_logits.permute(0, 2, 3, 1)[mask0].unsqueeze(-1).unsqueeze(-1)
                targ0 = y_aqi_t[mask0].unsqueeze(-1).unsqueeze(-1)
                loss_g0 = loss_fine_g0(logits0, targ0)
            else:
                loss_g0 = torch.tensor(0.0, device=device)

            # fine g1 (2 vs 3) mapped to (0 vs 1)
            mask1 = (y_grp_t == 1)
            if mask1.any():
                logits1 = y_g1_logits.permute(0, 2, 3, 1)[mask1].unsqueeze(-1).unsqueeze(-1)
                targ1 = (y_aqi_t[mask1] - 2).clamp(0, 1).unsqueeze(-1).unsqueeze(-1)
                loss_g1 = loss_fine_g1(logits1, targ1)
            else:
                loss_g1 = torch.tensor(0.0, device=device)

            # final probs for ordinal loss
            p4, _ = hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=cfg.coarse_temperature)

            fine_mean = 0.5 * (loss_g0 + loss_g1)
            loss_aqi_core = (
                cfg.lambda_aqi_group * loss_grp_val
                + cfg.lambda_aqi_fine * fine_mean
            ) / max(cfg.lambda_aqi_group + cfg.lambda_aqi_fine, 1e-6)

            loss_ord = ordinal_consistency_loss_from_probs(
                p4, y_aqi_t,
                w_2_to_3=float(cfg.ordinal_asym_2_to_3),
                w_3_to_2=float(cfg.ordinal_asym_3_to_2),
            )
            loss_aqi = loss_aqi_core + cfg.lambda_aqi_ordinal * loss_ord

            if log_vars is not None:
                w_reg = torch.exp(-log_vars[0])
                w_bin = torch.exp(-log_vars[1])
                w_aqi = torch.exp(-log_vars[2])
                loss = (
                    w_reg * loss_reg + log_vars[0]
                    + w_bin * loss_bin + log_vars[1]
                    + w_aqi * loss_aqi + log_vars[2]
                )
            else:
                loss = cfg.lambda_reg * loss_reg + cfg.lambda_cls_bin * loss_bin + cfg.lambda_aqi * loss_aqi

            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bsz = X_seq.size(0)
            tr_reg_sum += loss_reg.item() * bsz
            tr_bin_sum += loss_bin.item() * bsz
            tr_aqi_sum += loss_aqi.item() * bsz
            tr_n += bsz

        train_reg = tr_reg_sum / max(tr_n, 1)
        train_bin = tr_bin_sum / max(tr_n, 1)
        train_aqi = tr_aqi_sum / max(tr_n, 1)
        train_total = cfg.lambda_reg * train_reg + cfg.lambda_cls_bin * train_bin + cfg.lambda_aqi * train_aqi

        # ---------------- EVAL ----------------
        model.eval()
        te_reg_sum = te_bin_sum = te_aqi_sum = te_total_sum = 0.0
        te_n = 0
        cm_aqi_epoch = np.zeros((num_aqi_classes, num_aqi_classes), dtype=np.int64)

        with torch.no_grad():
            for X_seq, y_reg, y_cls, y_aqi, _ in test_loader:
                X_seq = torch.nan_to_num(X_seq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                y_reg = torch.nan_to_num(y_reg.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                y_cls = torch.nan_to_num(y_cls.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                y_aqi = torch.nan_to_num(y_aqi.to(device), nan=0.0, posinf=0.0, neginf=0.0)

                y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(X_seq)

                loss_reg = reg_loss_fn(y_reg_pred, y_reg)
                loss_bin = bin_loss_fn(y_cls_logits, y_cls)

                y_aqi_t = y_aqi.squeeze(1).long()
                y_grp_t = (y_aqi_t >= 2).long()

                loss_grp_val = loss_grp(y_grp_logits, y_grp_t)

                mask0 = (y_grp_t == 0)
                if mask0.any():
                    logits0 = y_g0_logits.permute(0, 2, 3, 1)[mask0].unsqueeze(-1).unsqueeze(-1)
                    targ0 = y_aqi_t[mask0].unsqueeze(-1).unsqueeze(-1)
                    loss_g0 = loss_fine_g0(logits0, targ0)
                else:
                    loss_g0 = torch.tensor(0.0, device=device)

                mask1 = (y_grp_t == 1)
                if mask1.any():
                    logits1 = y_g1_logits.permute(0, 2, 3, 1)[mask1].unsqueeze(-1).unsqueeze(-1)
                    targ1 = (y_aqi_t[mask1] - 2).clamp(0, 1).unsqueeze(-1).unsqueeze(-1)
                    loss_g1 = loss_fine_g1(logits1, targ1)
                else:
                    loss_g1 = torch.tensor(0.0, device=device)

                p4, pred_aqi = hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=cfg.coarse_temperature)

                fine_mean = 0.5 * (loss_g0 + loss_g1)
                loss_aqi_core = (
                    cfg.lambda_aqi_group * loss_grp_val
                    + cfg.lambda_aqi_fine * fine_mean
                ) / max(cfg.lambda_aqi_group + cfg.lambda_aqi_fine, 1e-6)

                loss_ord = ordinal_consistency_loss_from_probs(
                    p4, y_aqi_t,
                    w_2_to_3=float(cfg.ordinal_asym_2_to_3),
                    w_3_to_2=float(cfg.ordinal_asym_3_to_2),
                )
                loss_aqi = loss_aqi_core + cfg.lambda_aqi_ordinal * loss_ord

                if log_vars is not None:
                    w_reg = torch.exp(-log_vars[0])
                    w_bin = torch.exp(-log_vars[1])
                    w_aqi = torch.exp(-log_vars[2])
                    loss_total = (
                        w_reg * loss_reg + log_vars[0]
                        + w_bin * loss_bin + log_vars[1]
                        + w_aqi * loss_aqi + log_vars[2]
                    )
                else:
                    loss_total = cfg.lambda_reg * loss_reg + cfg.lambda_cls_bin * loss_bin + cfg.lambda_aqi * loss_aqi

                # Update AQI confusion (flatten)
                y_true_flat = y_aqi_t.detach().cpu().numpy().reshape(-1)
                y_pred_flat = pred_aqi.detach().cpu().numpy().reshape(-1)
                for t, p in zip(y_true_flat, y_pred_flat):
                    if 0 <= t < num_aqi_classes and 0 <= p < num_aqi_classes:
                        cm_aqi_epoch[t, p] += 1

                bsz = X_seq.size(0)
                te_reg_sum += loss_reg.item() * bsz
                te_bin_sum += loss_bin.item() * bsz
                te_aqi_sum += loss_aqi.item() * bsz
                te_total_sum += loss_total.item() * bsz
                te_n += bsz

        test_reg = te_reg_sum / max(te_n, 1)
        test_bin = te_bin_sum / max(te_n, 1)
        test_aqi = te_aqi_sum / max(te_n, 1)
        test_total = te_total_sum / max(te_n, 1) if log_vars is not None else (
            cfg.lambda_reg * test_reg + cfg.lambda_cls_bin * test_bin + cfg.lambda_aqi * test_aqi
        )

        aqi_macro_f1 = macro_f1_from_cm(cm_aqi_epoch)

        print(
            f"Epoch {epoch:03d} | Train Loss={train_total:.4f} (reg={train_reg:.4f}, bin={train_bin:.4f}, aqi={train_aqi:.4f}) | "
            f"Test Loss={test_total:.4f} (reg={test_reg:.4f}, bin={test_bin:.4f}, aqi={test_aqi:.4f}) | "
            f"AQI macro-F1={aqi_macro_f1:.3f}"
        )

        history_log["epoch"].append(epoch)
        history_log["train_loss"].append(train_total)
        history_log["train_reg_loss"].append(train_reg)
        history_log["train_cls_bin_loss"].append(train_bin)
        history_log["train_aqi_loss"].append(train_aqi)
        history_log["test_loss"].append(test_total)
        history_log["test_reg_loss"].append(test_reg)
        history_log["test_cls_bin_loss"].append(test_bin)
        history_log["test_aqi_loss"].append(test_aqi)
        history_log["aqi_macro_f1"].append(aqi_macro_f1)

        # checkpoint
        if cfg.best_checkpoint_metric == "aqi_macro_f1":
            score = aqi_macro_f1
            is_better = score > best_score
        else:
            score = test_total
            is_better = score < best_score

        if is_better:
            best_score = score
            torch.save(model.state_dict(), best_ckpt_path)

    # Save training log
    log_path = cfg.log_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history_log, f, indent=2)
    print("\nTraining log saved to:", log_path.resolve())
    print("Best checkpoint saved to:", best_ckpt_path.resolve())

    # Loss curves
    epochs = history_log["epoch"]
    plt.figure()
    plt.plot(epochs, history_log["train_loss"], label="Train Loss")
    plt.plot(epochs, history_log["test_loss"], label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "loss_curves_total.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(epochs, history_log["train_reg_loss"], label="Train Reg")
    plt.plot(epochs, history_log["test_reg_loss"], label="Test Reg")
    plt.plot(epochs, history_log["train_cls_bin_loss"], label="Train Bin")
    plt.plot(epochs, history_log["test_cls_bin_loss"], label="Test Bin")
    plt.plot(epochs, history_log["train_aqi_loss"], label="Train AQI")
    plt.plot(epochs, history_log["test_aqi_loss"], label="Test AQI")
    plt.xlabel("Epoch"); plt.ylabel("Loss (unweighted)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "loss_curves_components.png", dpi=150); plt.close()

    # Extra artifacts for best model
    generate_additional_artifacts(
        model=model,
        best_ckpt_path=best_ckpt_path,
        test_loader=DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True),
        target_names=target_names,
        aqi_class_names=aqi_class_names,
        lat_1d=lat_1d,
        lon_1d=lon_1d,
    )


# ========================== ARTIFACT GENERATION ============================ #

def _find_target_index(target_names, key="pm25"):
    key = str(key).lower()
    for i, n in enumerate(target_names):
        if key in str(n).lower():
            return i
    return 0


@torch.no_grad()
def generate_additional_artifacts(model, best_ckpt_path: Path, test_loader: DataLoader,
                                  target_names, aqi_class_names, lat_1d, lon_1d):
    device = torch.device(cfg.device)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.to(device).eval()

    num_targets = len(target_names)
    num_aqi_classes = len(aqi_class_names)

    all_y_reg_true_grid = []
    all_y_reg_pred_grid = []
    all_y_cls_true_grid = []
    all_y_cls_prob_grid = []
    all_y_aqi_true_grid = []
    all_y_aqi_prob_grid = []
    all_dates = []

    for X_seq, y_reg, y_cls, y_aqi, dates in test_loader:
        X_seq = torch.nan_to_num(X_seq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y_reg = torch.nan_to_num(y_reg.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y_cls = torch.nan_to_num(y_cls.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y_aqi = torch.nan_to_num(y_aqi.to(device), nan=0.0, posinf=0.0, neginf=0.0)

        y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(X_seq)
        y_cls_prob = torch.sigmoid(y_cls_logits)

        p4, _ = hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=cfg.coarse_temperature)

        all_y_reg_true_grid.append(y_reg.cpu().numpy())
        all_y_reg_pred_grid.append(y_reg_pred.cpu().numpy())
        all_y_cls_true_grid.append(y_cls.cpu().numpy())
        all_y_cls_prob_grid.append(y_cls_prob.cpu().numpy())
        all_y_aqi_true_grid.append(y_aqi.cpu().numpy())
        all_y_aqi_prob_grid.append(p4.cpu().numpy())
        all_dates.extend(list(dates))

    all_y_reg_true_grid = np.concatenate(all_y_reg_true_grid, axis=0)
    all_y_reg_pred_grid = np.concatenate(all_y_reg_pred_grid, axis=0)
    all_y_cls_true_grid = np.concatenate(all_y_cls_true_grid, axis=0)
    all_y_cls_prob_grid = np.concatenate(all_y_cls_prob_grid, axis=0)
    all_y_aqi_true_grid = np.concatenate(all_y_aqi_true_grid, axis=0)
    all_y_aqi_prob_grid = np.concatenate(all_y_aqi_prob_grid, axis=0)
    N = all_y_reg_true_grid.shape[0]

    cfg.plot_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    pred_npz_path = cfg.plot_dir / "test_predictions_grid.npz"
    np.savez_compressed(
        pred_npz_path,
        y_reg_true_grid=all_y_reg_true_grid,
        y_reg_pred_grid=all_y_reg_pred_grid,
        y_cls_true_grid=all_y_cls_true_grid,
        y_cls_prob_grid=all_y_cls_prob_grid,
        y_aqi_true_grid=all_y_aqi_true_grid,
        y_aqi_prob_grid=all_y_aqi_prob_grid,
        dates=np.array(all_dates),
    )
    print("Test predictions saved to:", pred_npz_path.resolve())

    # Flatten for metrics
    y_reg_true_flat = all_y_reg_true_grid.reshape(N, num_targets, -1).transpose(0, 2, 1).reshape(-1, num_targets)
    y_reg_pred_flat = all_y_reg_pred_grid.reshape(N, num_targets, -1).transpose(0, 2, 1).reshape(-1, num_targets)

    y_cls_true_flat = all_y_cls_true_grid.reshape(-1)
    y_cls_prob_flat = all_y_cls_prob_grid.reshape(-1)

    y_aqi_true_flat = all_y_aqi_true_grid.reshape(-1).astype(int)
    y_aqi_prob_flat = all_y_aqi_prob_grid.reshape(N, num_aqi_classes, -1).transpose(0, 2, 1).reshape(-1, num_aqi_classes)
    y_aqi_pred_flat = np.argmax(y_aqi_prob_flat, axis=1).astype(int)

    # Apply same post-hoc margins on flattened probs
    if cfg.aqi_posthoc_margin_down_32 > 0:
        diff32 = y_aqi_prob_flat[:, 3] - y_aqi_prob_flat[:, 2]
        mask = (y_aqi_pred_flat == 3) & (diff32 < cfg.aqi_posthoc_margin_down_32)
        y_aqi_pred_flat = y_aqi_pred_flat.copy()
        y_aqi_pred_flat[mask] = 2

    if cfg.aqi_posthoc_margin_23_up > 0:
        diff23 = y_aqi_prob_flat[:, 2] - y_aqi_prob_flat[:, 3]
        mask = (y_aqi_pred_flat == 2) & (diff23 < cfg.aqi_posthoc_margin_23_up)
        y_aqi_pred_flat = y_aqi_pred_flat.copy()
        y_aqi_pred_flat[mask] = 3

    if cfg.aqi_posthoc_margin_10_down > 0:
        diff10 = y_aqi_prob_flat[:, 1] - y_aqi_prob_flat[:, 0]
        mask = (y_aqi_pred_flat == 1) & (diff10 < cfg.aqi_posthoc_margin_10_down)
        y_aqi_pred_flat = y_aqi_pred_flat.copy()
        y_aqi_pred_flat[mask] = 0

    # Metrics
    reg_mae, reg_rmse = compute_regression_metrics(y_reg_true_flat, y_reg_pred_flat)
    bin_acc, bin_prec, bin_rec, bin_f1, cm_bin = compute_binary_metrics(y_cls_true_flat, y_cls_prob_flat, thr=0.5)
    cm_aqi = compute_multiclass_confusion(y_aqi_true_flat, y_aqi_pred_flat, num_aqi_classes)
    aqi_acc = float(np.trace(cm_aqi) / max(cm_aqi.sum(), 1))
    aqi_macro_f1 = macro_f1_from_cm(cm_aqi)

    print("\n[WindNO v2 test metrics]")
    print("  Regression MAE:", dict(zip(target_names, reg_mae)))
    print("  Regression RMSE:", dict(zip(target_names, reg_rmse)))
    print(f"  Binary PM2.5: acc={bin_acc:.3f}, prec={bin_prec:.3f}, rec={bin_rec:.3f}, f1={bin_f1:.3f}")
    print(f"  AQI: acc={aqi_acc:.3f}, macro-F1={aqi_macro_f1:.3f}")

    # Save confusion matrices
    plot_confusion_matrix(cm_bin, ["0", "1"], cfg.plot_dir / "cls_bin_confusion_matrix.png", "Binary PM2.5 Confusion (Test)", normalize=False)
    plot_confusion_matrix(cm_aqi, aqi_class_names, cfg.plot_dir / "aqi_confusion_matrix.png", "AQI Confusion (Test)", normalize=False)

    # AQI per-class metrics JSON
    eps = 1e-8
    aqi_stats = {}
    for c in range(num_aqi_classes):
        tp = cm_aqi[c, c]
        fp = cm_aqi[:, c].sum() - tp
        fn = cm_aqi[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        aqi_stats[c] = {"name": str(aqi_class_names[c]), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    with open(cfg.log_dir / "aqi_class_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"overall_acc": float(aqi_acc), "overall_macro_f1": float(aqi_macro_f1), "per_class": aqi_stats}, f, indent=2)

    # Regression scatter + error hist
    for c, name in enumerate(target_names):
        y_true_c = y_reg_true_flat[:, c]
        y_pred_c = y_reg_pred_flat[:, c]
        err = y_pred_c - y_true_c

        plt.figure(figsize=(5, 5))
        plt.scatter(y_true_c, y_pred_c, s=1, alpha=0.3)
        lims = [min(y_true_c.min(), y_pred_c.min()), max(y_true_c.max(), y_pred_c.max())]
        plt.plot(lims, lims, "r--", linewidth=1)
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel(f"True {name}"); plt.ylabel(f"Pred {name}")
        plt.title(f"{name} True vs Pred (Test)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(cfg.plot_dir / f"reg_scatter_{str(name).replace(':','_')}.png", dpi=150); plt.close()

        plt.figure()
        plt.hist(err, bins=100)
        plt.xlabel(f"Error {name} (pred-true)"); plt.ylabel("Freq")
        plt.title(f"{name} error distribution (Test)")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / f"reg_error_hist_{str(name).replace(':','_')}.png", dpi=150); plt.close()

    # ROC + PR for binary
    thresholds = np.linspace(0.0, 1.0, 51)
    tprs, fprs, precisions, recalls = [], [], [], []
    yt = y_cls_true_flat.astype(int)
    yp = y_cls_prob_flat.astype(float)

    for thr in thresholds:
        yhat = (yp >= thr).astype(int)
        tp = ((yhat == 1) & (yt == 1)).sum()
        fp = ((yhat == 1) & (yt == 0)).sum()
        fn = ((yhat == 0) & (yt == 1)).sum()
        tn = ((yhat == 0) & (yt == 0)).sum()
        tpr = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        tprs.append(tpr); fprs.append(fpr); precisions.append(prec); recalls.append(rec)

    plt.figure()
    plt.plot(fprs, tprs, marker=".")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Binary PM2.5 ROC (Test)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "cls_bin_roc_curve.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(recalls, precisions, marker=".")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Binary PM2.5 PR (Test)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "cls_bin_pr_curve.png", dpi=150); plt.close()

    # Region map + random geo maps
    if lat_1d is not None and lon_1d is not None:
        plot_region_outline(lat_1d, lon_1d, cfg.plot_dir / "region_map_grid.png")

        pm25_idx = target_names.index("eac4:pm2p5") if "eac4:pm2p5" in target_names else _find_target_index(target_names, "pm25")
        rng = np.random.default_rng(cfg.random_seed)
        n_to_plot = min(cfg.n_random_test_days, N)
        sample_idx = rng.choice(N, size=n_to_plot, replace=False)

        for k, i in enumerate(sample_idx):
            true_reg = all_y_reg_true_grid[i]
            pred_reg = all_y_reg_pred_grid[i]
            true_cls = all_y_cls_true_grid[i, 0]
            pred_cls_prob = all_y_cls_prob_grid[i, 0]
            true_aqi = all_y_aqi_true_grid[i, 0]
            pred_aqi_cls = np.argmax(all_y_aqi_prob_grid[i], axis=0)
            date_str = str(all_dates[i])

            plot_geo_example(
                true_reg[pm25_idx], pred_reg[pm25_idx],
                true_cls, pred_cls_prob,
                true_aqi, pred_aqi_cls, aqi_class_names,
                lat_1d, lon_1d, date_str, k, cfg.plot_dir
            )

    # Multi-horizon evaluation
    try:
        horizons = list(cfg.eval_horizons)
        modes = list(cfg.multi_horizon_exo_modes)
        for mode in modes:
            df = evaluate_multi_horizon_rollout(model, test_loader.dataset, horizons, target_names, aqi_class_names, cfg.plot_dir, exo_mode=mode)
            df.to_csv(cfg.plot_dir / f"multi_horizon_metrics_{mode}.csv", index=False)
            with open(cfg.plot_dir / f"multi_horizon_metrics_{mode}.json", "w", encoding="utf-8") as f:
                json.dump(df.to_dict(orient="records"), f, indent=2)
            plot_multi_horizon_summary(df, target_names, cfg.plot_dir / f"multi_horizon_summary_{mode}.png")
        print("Multi-horizon artifacts saved to:", cfg.plot_dir.resolve())
    except Exception as e:
        print("[Warn] Multi-horizon evaluation failed:", repr(e))


@torch.no_grad()
def evaluate_multi_horizon_rollout(model, dataset: SpatioTemporalAirDataset, horizons, target_names, aqi_class_names, out_dir: Path, exo_mode: str):
    device = torch.device(cfg.device)
    model.to(device).eval()

    exo_mode = str(exo_mode).lower().strip()
    if exo_mode not in ("oracle", "persist"):
        raise ValueError("exo_mode must be 'oracle' or 'persist'")

    horizons = [int(h) for h in horizons]
    batch_size = int(cfg.multi_horizon_batch_size or cfg.batch_size)
    max_samples = cfg.multi_horizon_max_samples

    X_4d = dataset.X_4d
    y_reg_4d = dataset.y_reg_4d
    y_cls_4d = dataset.y_cls_4d
    y_aqi_4d = dataset.y_aqi_4d
    dates = dataset.dates_4d
    years = pd.to_datetime(dates).year.values

    T = dataset.T
    H = dataset.H
    W = dataset.W
    history = int(dataset.history)
    num_targets = y_reg_4d.shape[1]
    num_aqi_classes = len(aqi_class_names)

    t_in_end_all = dataset.t_in_end_indices

    h_max = int(max(horizons))
    base_valid = (t_in_end_all - history + 1 >= 0) & (t_in_end_all + h_max < T)
    if np.any(base_valid):
        idx_base = np.where(base_valid)[0]
        year_ok = np.ones_like(idx_base, dtype=bool)
        for _h in horizons:
            t_out = t_in_end_all[idx_base] + int(_h)
            year_ok &= (years[t_out] > dataset.train_end_year)
        base_valid[idx_base] = year_ok
    shared_idx = np.where(base_valid)[0]
    print(f"[Multi-horizon][{exo_mode}] shared anchors n={len(shared_idx)} horizons={horizons}")

    rows = []
    for h in horizons:
        valid_idx = shared_idx
        if max_samples is not None:
            valid_idx = valid_idx[: int(max_samples)]
        n_samples = int(len(valid_idx))
        if n_samples == 0:
            rows.append({"exo_mode": exo_mode, "horizon": h, "n_samples": 0})
            continue

        y_reg_true_parts, y_reg_pred_parts = [], []
        y_cls_true_parts, y_cls_prob_parts = [], []
        y_aqi_true_parts, y_aqi_pred_parts = [], []

        for s0 in range(0, n_samples, batch_size):
            batch_ids = valid_idx[s0:s0 + batch_size]
            t_in_end = t_in_end_all[batch_ids]
            B = t_in_end.shape[0]

            t_hist = (t_in_end[:, None] - history + 1) + np.arange(history)[None, :]
            X_seq = X_4d[t_hist]
            y_reg_seq = y_reg_4d[t_hist]
            X_full = np.concatenate([X_seq, y_reg_seq], axis=2)
            X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
            X_full_t = torch.from_numpy(X_full).float().to(device)

            y_reg_pred = None
            y_cls_logits = None
            y_grp_logits = None
            y_g0_logits = None
            y_g1_logits = None

            for step in range(1, h + 1):
                y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(X_full_t)
                if step < h:
                    t_new = t_in_end + step
                    if exo_mode == "oracle":
                        X_new = X_4d[t_new]
                    else:
                        X_new = X_4d[t_in_end]
                    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
                    X_new_t = torch.from_numpy(X_new).float().to(device)

                    new_last = torch.cat([X_new_t, y_reg_pred], dim=1)
                    X_full_t = torch.cat([X_full_t[:, 1:], new_last.unsqueeze(1)], dim=1)

            y_cls_prob = torch.sigmoid(y_cls_logits)
            _, aqi_pred = hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=cfg.coarse_temperature)

            t_out = t_in_end + h
            y_reg_true = np.nan_to_num(y_reg_4d[t_out], nan=0.0)
            y_cls_true = np.nan_to_num(y_cls_4d[t_out], nan=0.0)
            y_aqi_true = np.nan_to_num(y_aqi_4d[t_out], nan=0.0)

            y_reg_true_flat = y_reg_true.reshape(B, num_targets, -1).transpose(0, 2, 1).reshape(-1, num_targets)
            y_reg_pred_flat = y_reg_pred.detach().cpu().numpy().reshape(B, num_targets, -1).transpose(0, 2, 1).reshape(-1, num_targets)

            y_reg_true_parts.append(y_reg_true_flat)
            y_reg_pred_parts.append(y_reg_pred_flat)
            y_cls_true_parts.append(y_cls_true.reshape(-1))
            y_cls_prob_parts.append(y_cls_prob.detach().cpu().numpy().reshape(-1))
            y_aqi_true_parts.append(y_aqi_true.reshape(-1).astype(int))
            y_aqi_pred_parts.append(aqi_pred.detach().cpu().numpy().reshape(-1).astype(int))

        y_reg_true_all = np.concatenate(y_reg_true_parts, axis=0)
        y_reg_pred_all = np.concatenate(y_reg_pred_parts, axis=0)
        y_cls_true_all = np.concatenate(y_cls_true_parts, axis=0).astype(int)
        y_cls_prob_all = np.concatenate(y_cls_prob_parts, axis=0)
        y_aqi_true_all = np.concatenate(y_aqi_true_parts, axis=0).astype(int)
        y_aqi_pred_all = np.concatenate(y_aqi_pred_parts, axis=0).astype(int)

        reg_mae, reg_rmse = compute_regression_metrics(y_reg_true_all, y_reg_pred_all)
        bin_acc, bin_prec, bin_rec, bin_f1, cm_bin = compute_binary_metrics(y_cls_true_all, y_cls_prob_all, thr=0.5)
        cm_aqi = compute_multiclass_confusion(y_aqi_true_all, y_aqi_pred_all, num_aqi_classes)
        aqi_acc = float(np.trace(cm_aqi) / max(cm_aqi.sum(), 1))
        aqi_macro_f1 = macro_f1_from_cm(cm_aqi)

        plot_confusion_matrix(cm_bin, ["0", "1"], out_dir / f"multi_horizon_{exo_mode}_bin_confusion_h{h}.png",
                              f"Binary PM2.5 Confusion ({exo_mode}, h={h})", normalize=False)
        plot_confusion_matrix(cm_aqi, aqi_class_names, out_dir / f"multi_horizon_{exo_mode}_aqi_confusion_h{h}.png",
                              f"AQI Confusion ({exo_mode}, h={h})", normalize=False)

        row = {"exo_mode": exo_mode, "horizon": h, "n_samples": n_samples,
               "aqi_acc": aqi_acc, "aqi_macro_f1": aqi_macro_f1,
               "bin_acc": bin_acc, "bin_prec": bin_prec, "bin_rec": bin_rec, "bin_f1": bin_f1}
        for name, mae, rmse in zip(target_names, reg_mae, reg_rmse):
            row[f"reg_mae_{name}"] = float(mae)
            row[f"reg_rmse_{name}"] = float(rmse)
        row["reg_mae_mean"] = float(np.mean(reg_mae)) if len(reg_mae) else np.nan
        row["reg_rmse_mean"] = float(np.mean(reg_rmse)) if len(reg_rmse) else np.nan
        rows.append(row)

        print(f"[Multi-horizon][{exo_mode}] h={h}: n={n_samples} | binF1={bin_f1:.3f} | AQI macroF1={aqi_macro_f1:.3f}")

    return pd.DataFrame(rows).sort_values(["exo_mode", "horizon"]).reset_index(drop=True)


def plot_multi_horizon_summary(df: pd.DataFrame, target_names, out_path: Path):
    if df is None or len(df) == 0:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pm25_idx = _find_target_index(target_names, key="pm25")
    pm25_name = target_names[pm25_idx] if len(target_names) else "pm25"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    def _plot_group(gdf: pd.DataFrame, label_prefix: str = ""):
        gdf = gdf.sort_values("horizon")
        h = gdf["horizon"].values
        rmse_pm25 = gdf.get(f"reg_rmse_{pm25_name}", pd.Series([np.nan] * len(gdf))).values
        rmse_mean = gdf.get("reg_rmse_mean", pd.Series([np.nan] * len(gdf))).values
        bin_f1 = gdf.get("bin_f1", pd.Series([np.nan] * len(gdf))).values
        aqi_f1 = gdf.get("aqi_macro_f1", pd.Series([np.nan] * len(gdf))).values
        aqi_acc = gdf.get("aqi_acc", pd.Series([np.nan] * len(gdf))).values

        axes[0].plot(h, rmse_pm25, marker="o", label=f"{label_prefix}{pm25_name} RMSE")
        axes[0].plot(h, rmse_mean, marker="o", label=f"{label_prefix}Mean RMSE")
        axes[0].set_xlabel("Horizon (days)"); axes[0].set_ylabel("RMSE"); axes[0].set_title("Regression vs Horizon"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(h, bin_f1, marker="o", label=f"{label_prefix}PM2.5 bin F1")
        axes[1].set_xlabel("Horizon (days)"); axes[1].set_ylabel("F1"); axes[1].set_title("Binary vs Horizon"); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)

        axes[2].plot(h, aqi_f1, marker="o", label=f"{label_prefix}AQI macro-F1")
        axes[2].plot(h, aqi_acc, marker="o", label=f"{label_prefix}AQI acc")
        axes[2].set_xlabel("Horizon (days)"); axes[2].set_ylabel("Score"); axes[2].set_title("AQI vs Horizon"); axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1)

    if "exo_mode" in df.columns:
        for mode, g in df.groupby("exo_mode"):
            _plot_group(g, label_prefix=f"{mode}: ")
    else:
        _plot_group(df, label_prefix="")

    axes[0].legend(); axes[1].legend(); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    train_and_evaluate()
