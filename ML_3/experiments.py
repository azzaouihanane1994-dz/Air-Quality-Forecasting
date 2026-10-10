#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WindNO benchmark experiments.

Train and evaluate neural and deterministic baselines using a chronological
split, train-only normalization, physical-unit regression metrics and
classification diagnostics. Outputs are stored in ML_3/benchmark_results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
WINDNO_PATH = Path(__file__).resolve().with_name("WindNO_2.py")
DATASET_PATH = ROOT / "datasets" / "global_airpollution_2015_2024_4d.npz"
OUT_DIR = Path(__file__).resolve().parent / "benchmark_results"
UGM3_FACTOR = 1.0e9


@dataclass
class ExperimentConfig:
    train_end_year: int = 2022
    val_year: int = 2023
    test_year: int = 2024
    history: int = 7
    horizon: int = 1
    batch_size: int = 64
    num_epochs: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    seed: int = 42
    max_norm_clip: float = 10.0
    run_training: bool = False


MODEL_SPECS = {
    "windno": {"label": "WindNO", "slug": "windno"},
    "convlstm": {"label": "ConvLSTM", "slug": "convlstm"},
    "unet": {"label": "U-Net", "slug": "unet"},
    "plain_fno": {"label": "Plain FNO", "slug": "plain_fno"},
}


def load_windno_module():
    spec = importlib.util.spec_from_file_location("windno2_experiment_import", WINDNO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["windno2_experiment_import"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.cfg.device = "cpu"
    module.cfg.batch_size = 64
    module.cfg.num_workers = 0
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def years_from_dates(dates: np.ndarray) -> np.ndarray:
    return pd.to_datetime(dates).year.values


def split_output_indices(dates: np.ndarray, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    years = years_from_dates(dates)
    t_in_end = np.arange(cfg.history - 1, len(dates) - cfg.horizon, dtype=int)
    t_out = t_in_end + cfg.horizon
    return {
        "train": t_in_end[years[t_out] <= cfg.train_end_year],
        "val": t_in_end[years[t_out] == cfg.val_year],
        "test": t_in_end[years[t_out] == cfg.test_year],
    }


def split_target_indices(dates: np.ndarray, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    splits = split_output_indices(dates, cfg)
    return {k: v + cfg.horizon for k, v in splits.items()}


def finite_train_stats(arr: np.ndarray, time_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    subset = arr[time_indices].astype(np.float64)
    subset = np.where(np.isfinite(subset), subset, np.nan)
    mean = np.nanmean(subset, axis=(0, 2, 3)).astype(np.float32)
    std = np.nanstd(subset, axis=(0, 2, 3)).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1.0e-12, 1.0, std).astype(np.float32)
    return mean, std


def compute_normalization(data: np.lib.npyio.NpzFile, cfg: ExperimentConfig, wind_idx: Dict[str, int | None]):
    dates = data["dates_4d"]
    train_t_out = split_target_indices(dates, cfg)["train"]
    x_mean, x_std = finite_train_stats(data["X_4d"], train_t_out)
    y_ugm3 = data["y_reg_4d"].astype(np.float32) * UGM3_FACTOR
    y_mean, y_std = finite_train_stats(y_ugm3, train_t_out)

    wind_feature_indices = {i for i in wind_idx.values() if i is not None}
    for i in wind_feature_indices:
        x_mean[i] = 0.0
        x_std[i] = 1.0
    return x_mean, x_std, y_mean, y_std, sorted(wind_feature_indices)


class NormalizedAirDataset(Dataset):
    def __init__(
        self,
        data: np.lib.npyio.NpzFile,
        split: str,
        cfg: ExperimentConfig,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ):
        self.X = data["X_4d"].astype(np.float32)
        self.y_reg = data["y_reg_4d"].astype(np.float32) * UGM3_FACTOR
        self.y_cls = data["y_class_4d"].astype(np.float32)
        self.y_aqi = data["y_aqi_4d"].astype(np.int64)
        self.dates = data["dates_4d"]
        self.cfg = cfg
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)
        self.t_in_end = split_output_indices(self.dates, cfg)[split]
        if len(self.t_in_end) == 0:
            raise ValueError(f"No samples found for split={split}")

    def __len__(self) -> int:
        return len(self.t_in_end)

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self.x_mean[None, :, None, None]) / self.x_std[None, :, None, None]
        return np.clip(x, -self.cfg.max_norm_clip, self.cfg.max_norm_clip).astype(np.float32)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = (y - self.y_mean[:, None, None]) / self.y_std[:, None, None]
        return np.clip(y, -self.cfg.max_norm_clip, self.cfg.max_norm_clip).astype(np.float32)

    def __getitem__(self, idx: int):
        t_end = int(self.t_in_end[idx])
        t_out = t_end + self.cfg.horizon
        t_start = t_end - self.cfg.history + 1
        x_exo = self._normalize_x(self.X[t_start : t_end + 1])
        y_hist = self._normalize_y(self.y_reg[t_start : t_end + 1])
        x_full = np.concatenate([x_exo, y_hist], axis=1).astype(np.float32)
        y_reg_norm = self._normalize_y(self.y_reg[t_out])
        y_cls = np.nan_to_num(self.y_cls[t_out], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_aqi = np.nan_to_num(self.y_aqi[t_out], nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
        date_out = np.datetime_as_string(self.dates[t_out], unit="D")
        return (
            torch.from_numpy(x_full),
            torch.from_numpy(y_reg_norm),
            torch.from_numpy(y_cls),
            torch.from_numpy(y_aqi),
            date_out,
        )


def denorm_reg(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    y = y_norm * y_std[None, :, None, None] + y_mean[None, :, None, None]
    return np.maximum(y, 0.0)


def compute_regression(y_true: np.ndarray, y_pred: np.ndarray, names: Iterable[str]) -> Tuple[List[dict], float]:
    rows = []
    rmses = []
    for i, name in enumerate(names):
        err = y_pred[:, i] - y_true[:, i]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({"target": str(name), "mae_ugm3": mae, "rmse_ugm3": rmse})
        rmses.append(rmse)
    return rows, float(np.mean(rmses))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    t = y_true.reshape(-1).astype(int)
    p = y_pred.reshape(-1).astype(int)
    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    acc = (tp + tn) / max(len(t), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1.0e-12)
    return {"bin_acc": acc, "bin_prec": prec, "bin_rec": rec, "bin_f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def aqi_from_reg(y_ugm3: np.ndarray, data: np.lib.npyio.NpzFile) -> np.ndarray:
    bounds = [
        data["aqi_bounds_pm25_ugm3"],
        data["aqi_bounds_pm10_ugm3"],
        data["aqi_bounds_go3_ugm3"],
        data["aqi_bounds_no2_ugm3"],
    ]
    classes = []
    for c, b in enumerate(bounds):
        classes.append(np.digitize(y_ugm3[:, c], b, right=False))
    return np.max(np.stack(classes, axis=1), axis=1)[:, None].astype(np.int64)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    t = y_true.reshape(-1).astype(int)
    p = y_pred.reshape(-1).astype(int)
    f1s = []
    for c in range(num_classes):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1.0e-12))
    return float(np.mean(f1s))


def classification_metrics(y_cls_true, y_aqi_true, y_reg_pred_ugm3, data) -> dict:
    y_bin_pred = (y_reg_pred_ugm3[:, 0:1] >= float(data["pm25_threshold"])).astype(np.int64)
    y_aqi_pred = aqi_from_reg(y_reg_pred_ugm3, data)
    out = binary_metrics(y_cls_true, y_bin_pred)
    out["aqi_acc"] = float((y_aqi_pred.reshape(-1) == y_aqi_true.reshape(-1)).mean())
    out["aqi_macro_f1"] = macro_f1(y_aqi_true, y_aqi_pred, len(data["aqi_class_names"]))
    return out


def make_baseline_predictions(data: np.lib.npyio.NpzFile, cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = data["dates_4d"]
    years = years_from_dates(dates)
    doy = pd.to_datetime(dates).dayofyear.values
    splits_out = split_target_indices(dates, cfg)
    test_out = splits_out["test"]
    test_in_end = test_out - cfg.horizon
    train_days = np.where(years <= cfg.train_end_year)[0]

    y = data["y_reg_4d"].astype(np.float64) * UGM3_FACTOR
    y_true = y[test_out]
    y_cls_true = data["y_class_4d"][test_out]
    y_aqi_true = data["y_aqi_4d"][test_out]
    names = list(data["target_names"])

    annual = y[train_days].mean(axis=0)
    doy_cache = {}
    for d in np.unique(doy[train_days]):
        idx = train_days[doy[train_days] == d]
        doy_cache[int(d)] = y[idx].mean(axis=0)

    baselines = {
        "Persistence": y[test_in_end],
        "7-day mean": np.stack([y[t - cfg.history + 1 : t + 1].mean(axis=0) for t in test_in_end]),
        "Annual climatology": np.broadcast_to(annual, y_true.shape),
        "Day-of-year climatology": np.stack([doy_cache.get(int(doy[t]), annual) for t in test_out]),
    }

    reg_rows = []
    cls_rows = []
    for name, pred in baselines.items():
        reg_metrics, mean_rmse = compute_regression(y_true, pred, names)
        for row in reg_metrics:
            row["model"] = name
            reg_rows.append(row)
        cls = classification_metrics(y_cls_true, y_aqi_true, pred, data)
        cls["model"] = name
        cls_rows.append(cls)
        reg_rows.append({"model": name, "target": "mean", "mae_ugm3": np.nan, "rmse_ugm3": mean_rmse})

    return pd.DataFrame(reg_rows), pd.DataFrame(cls_rows)


def class_frequency_table(data: np.lib.npyio.NpzFile, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    splits = split_target_indices(data["dates_4d"], cfg)
    class_names = list(data["aqi_class_names"])
    for split, idx in splits.items():
        y_bin = data["y_class_4d"][idx].reshape(-1).astype(int)
        rows.append({
            "split": split,
            "label": "PM2.5 exceedance=1",
            "count": int(y_bin.sum()),
            "fraction": float(y_bin.mean()),
        })
        y_aqi = data["y_aqi_4d"][idx].reshape(-1).astype(int)
        counts = np.bincount(y_aqi, minlength=len(class_names))
        total = counts.sum()
        for c, n in enumerate(counts):
            rows.append({
                "split": split,
                "label": f"AQI {c}: {class_names[c]}",
                "count": int(n),
                "fraction": float(n / max(total, 1)),
            })
    return pd.DataFrame(rows)


def feature_inventory(data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    features = [str(f) for f in data["feature_names"]]
    rows = []
    for i, name in enumerate(features):
        if name.startswith("eac4:"):
            source = "CAMS/EAC4 and emissions merge"
        elif name.startswith("static:"):
            source = "HOTOSM static covariate"
        elif name in {"u100", "v100", "tcc", "blh", "skt", "ws100", "vent100"}:
            source = "ERA5"
        elif name in {"fire_label", "fire_frp"}:
            source = "NASA FIRMS"
        else:
            source = "Merged predictor"
        rows.append({"index": i, "feature": name, "source_group": source})
    return pd.DataFrame(rows)


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, c_targets: int):
        super().__init__()
        self.head_reg = nn.Conv2d(hidden_dim, c_targets, 1)
        self.head_cls_bin = nn.Conv2d(hidden_dim, 1, 1)
        self.head_aqi_group = nn.Conv2d(hidden_dim, 2, 1)
        self.head_aqi_fine_g0 = nn.Conv2d(hidden_dim, 2, 1)
        self.head_aqi_fine_g1 = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, h: torch.Tensor):
        return (
            self.head_reg(h),
            self.head_cls_bin(h),
            self.head_aqi_group(h),
            self.head_aqi_fine_g0(h),
            self.head_aqi_fine_g1(h),
        )


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, state):
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMBaseline(nn.Module):
    def __init__(self, input_channels: int, c_targets: int, hidden_dim: int = 64):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size=3)
        self.refine = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.heads = MultiTaskHeads(hidden_dim, c_targets)

    def forward(self, x_seq_full: torch.Tensor):
        state = None
        for t in range(x_seq_full.size(1)):
            state = self.cell(x_seq_full[:, t], state)
        h, _ = state
        return self.heads(self.refine(h))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    def __init__(self, history: int, input_channels: int, c_targets: int, base_dim: int = 32):
        super().__init__()
        in_channels = history * input_channels
        self.enc1 = ConvBlock(in_channels, base_dim)
        self.enc2 = ConvBlock(base_dim, base_dim * 2)
        self.bottleneck = ConvBlock(base_dim * 2, base_dim * 4)
        self.dec2 = ConvBlock(base_dim * 4 + base_dim * 2, base_dim * 2)
        self.dec1 = ConvBlock(base_dim * 2 + base_dim, base_dim)
        self.heads = MultiTaskHeads(base_dim, c_targets)

    def forward(self, x_seq_full: torch.Tensor):
        b, t, c, h, w = x_seq_full.shape
        x = x_seq_full.reshape(b, t * c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        z = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.heads(d1)


class PlainFNOBaseline(nn.Module):
    def __init__(
        self,
        history: int,
        input_channels: int,
        c_targets: int,
        fno_block_cls,
        hidden_dim: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(history * input_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([fno_block_cls(hidden_dim, modes1, modes2) for _ in range(num_blocks)])
        self.heads = MultiTaskHeads(hidden_dim, c_targets)

    def forward(self, x_seq_full: torch.Tensor):
        b, t, c, h, w = x_seq_full.shape
        h_state = self.stem(x_seq_full.reshape(b, t * c, h, w))
        for block in self.blocks:
            h_state = block(h_state)
        return self.heads(h_state)


def build_model(model_key: str, windno, feature_count: int, target_count: int, wind_idx: Dict[str, int | None], cfg: ExperimentConfig) -> nn.Module:
    input_channels = feature_count + target_count
    if model_key == "windno":
        return windno.WindNO(
            F_exo=feature_count,
            C_targets=target_count,
            hidden_dim=windno.cfg.hidden_dim,
            wind_idx=wind_idx,
        )
    if model_key == "convlstm":
        return ConvLSTMBaseline(input_channels=input_channels, c_targets=target_count, hidden_dim=64)
    if model_key == "unet":
        return UNetBaseline(history=cfg.history, input_channels=input_channels, c_targets=target_count, base_dim=32)
    if model_key == "plain_fno":
        return PlainFNOBaseline(
            history=cfg.history,
            input_channels=input_channels,
            c_targets=target_count,
            fno_block_cls=windno.FNOBlock,
            hidden_dim=64,
            modes1=windno.cfg.fno_modes1,
            modes2=windno.cfg.fno_modes2,
            num_blocks=4,
        )
    raise ValueError(f"Unknown model_key={model_key!r}")


def train_one_seed(cfg: ExperimentConfig, data: np.lib.npyio.NpzFile, model_key: str = "windno") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model_key={model_key!r}. Expected one of {sorted(MODEL_SPECS)}")
    model_label = MODEL_SPECS[model_key]["label"]
    model_slug = MODEL_SPECS[model_key]["slug"]
    windno = load_windno_module()
    set_seed(cfg.seed)
    feature_names = list(data["feature_names"])
    target_names = list(data["target_names"])
    aqi_class_names = list(data["aqi_class_names"])
    wind_idx = windno.find_wind_indices(feature_names)
    x_mean, x_std, y_mean, y_std, wind_features = compute_normalization(data, cfg, wind_idx)

    train_ds = NormalizedAirDataset(data, "train", cfg, x_mean, x_std, y_mean, y_std)
    val_ds = NormalizedAirDataset(data, "val", cfg, x_mean, x_std, y_mean, y_std)
    test_ds = NormalizedAirDataset(data, "test", cfg, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = build_model(model_key, windno, len(feature_names), len(target_names), wind_idx, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    reg_loss_fn = nn.MSELoss()
    bin_loss_fn = nn.BCEWithLogitsLoss()

    train_t_out = split_target_indices(data["dates_4d"], cfg)["train"]
    aqi_w4, aqi_counts = windno.compute_aqi_class_weights(
        data["y_aqi_4d"], data["dates_4d"], cfg.train_end_year, len(aqi_class_names)
    )
    group_counts = np.array([aqi_counts[0] + aqi_counts[1], aqi_counts[2] + aqi_counts[3]], dtype=np.int64)
    fine01_counts = np.array([aqi_counts[0], aqi_counts[1]], dtype=np.int64)
    fine23_counts = np.array([aqi_counts[2], aqi_counts[3]], dtype=np.int64)
    loss_grp = windno.make_multi_class_loss_fn(windno.cfg.aqi_loss_type, windno.build_weights_from_counts(group_counts), windno.cfg.focal_gamma, windno.cfg.label_smoothing_eps, "cpu")
    loss_g0 = windno.make_multi_class_loss_fn(windno.cfg.aqi_loss_type, windno.build_weights_from_counts(fine01_counts), windno.cfg.focal_gamma, windno.cfg.label_smoothing_eps, "cpu")
    fine23_w = windno.build_weights_from_counts(fine23_counts).copy()
    fine23_w[0] *= float(windno.cfg.fine23_weight_mult_class2)
    fine23_w[1] *= float(windno.cfg.fine23_weight_mult_class3)
    fine23_w = fine23_w / max(fine23_w.mean(), 1.0e-9)
    loss_g1 = windno.make_multi_class_loss_fn(windno.cfg.aqi_loss_type, fine23_w, windno.cfg.focal_gamma, windno.cfg.label_smoothing_eps, "cpu")

    def aqi_loss(y_grp_logits, y_g0_logits, y_g1_logits, y_aqi):
        y_aqi_t = y_aqi.squeeze(1).long()
        y_grp_t = (y_aqi_t >= 2).long()
        lg = loss_grp(y_grp_logits, y_grp_t)
        mask0 = y_grp_t == 0
        if mask0.any():
            l0 = loss_g0(y_g0_logits.permute(0, 2, 3, 1)[mask0].unsqueeze(-1).unsqueeze(-1), y_aqi_t[mask0].unsqueeze(-1).unsqueeze(-1))
        else:
            l0 = torch.tensor(0.0)
        mask1 = y_grp_t == 1
        if mask1.any():
            l1 = loss_g1(y_g1_logits.permute(0, 2, 3, 1)[mask1].unsqueeze(-1).unsqueeze(-1), (y_aqi_t[mask1] - 2).clamp(0, 1).unsqueeze(-1).unsqueeze(-1))
        else:
            l1 = torch.tensor(0.0)
        p4, _ = windno.hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=windno.cfg.coarse_temperature)
        lord = windno.ordinal_consistency_loss_from_probs(
            p4, y_aqi_t, w_2_to_3=float(windno.cfg.ordinal_asym_2_to_3), w_3_to_2=float(windno.cfg.ordinal_asym_3_to_2)
        )
        core = (windno.cfg.lambda_aqi_group * lg + windno.cfg.lambda_aqi_fine * 0.5 * (l0 + l1)) / max(
            windno.cfg.lambda_aqi_group + windno.cfg.lambda_aqi_fine, 1.0e-6
        )
        return core + windno.cfg.lambda_aqi_ordinal * lord

    def evaluate(loader: DataLoader) -> dict:
        model.eval()
        all_true_norm, all_pred_norm, all_cls_true, all_cls_prob, all_aqi_true, all_aqi_pred = [], [], [], [], [], []
        with torch.no_grad():
            for x, y_reg, y_cls, y_aqi, _ in loader:
                y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(x.float())
                p4, pred_aqi = windno.hier_aqi_probs_and_pred(y_grp_logits, y_g0_logits, y_g1_logits, temp=windno.cfg.coarse_temperature)
                all_true_norm.append(y_reg.numpy())
                all_pred_norm.append(y_reg_pred.numpy())
                all_cls_true.append(y_cls.numpy())
                all_cls_prob.append(torch.sigmoid(y_cls_logits).numpy())
                all_aqi_true.append(y_aqi.numpy())
                all_aqi_pred.append(pred_aqi.unsqueeze(1).numpy())
        true_norm = np.concatenate(all_true_norm)
        pred_norm = np.concatenate(all_pred_norm)
        true_ugm3 = denorm_reg(true_norm, y_mean, y_std)
        pred_ugm3 = denorm_reg(pred_norm, y_mean, y_std)
        reg_rows, mean_rmse = compute_regression(true_ugm3, pred_ugm3, target_names)
        y_cls_true = np.concatenate(all_cls_true)
        y_cls_pred = (np.concatenate(all_cls_prob) >= 0.5).astype(np.int64)
        y_aqi_true = np.concatenate(all_aqi_true)
        y_aqi_pred = np.concatenate(all_aqi_pred)
        cls = binary_metrics(y_cls_true, y_cls_pred)
        cls["aqi_acc"] = float((y_aqi_pred.reshape(-1) == y_aqi_true.reshape(-1)).mean())
        cls["aqi_macro_f1"] = macro_f1(y_aqi_true, y_aqi_pred, len(aqi_class_names))
        cls["reg_rmse_mean_ugm3"] = mean_rmse
        cls["negative_regression_fraction_after_clip"] = 0.0
        return {"reg_rows": reg_rows, "cls": cls}

    best_state = None
    best_val = -math.inf
    history_rows = []
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y_reg, y_cls, y_aqi, _ in train_loader:
            opt.zero_grad(set_to_none=True)
            y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(x.float())
            loss_reg = reg_loss_fn(y_reg_pred, y_reg.float())
            loss_bin = bin_loss_fn(y_cls_logits, y_cls.float())
            loss_aqi = aqi_loss(y_grp_logits, y_g0_logits, y_g1_logits, y_aqi)
            loss = loss_reg + loss_bin + loss_aqi
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
        val_metrics = evaluate(val_loader)["cls"]
        history_rows.append({"epoch": epoch, "train_loss": loss_sum / max(n, 1), **val_metrics})
        if val_metrics["aqi_macro_f1"] > best_val:
            best_val = val_metrics["aqi_macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"model={model_label} seed={cfg.seed} epoch={epoch:02d} "
            f"val_aqi_macro_f1={val_metrics['aqi_macro_f1']:.4f} val_bin_f1={val_metrics['bin_f1']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_eval = evaluate(test_loader)
    reg_df = pd.DataFrame(test_eval["reg_rows"])
    reg_df["model"] = model_label
    cls_row = {"model": model_label, **test_eval["cls"]}
    cls_df = pd.DataFrame([cls_row])
    pd.DataFrame(history_rows).to_csv(OUT_DIR / f"{model_slug}_seed{cfg.seed}_val_history.csv", index=False)
    torch.save(best_state, OUT_DIR / f"{model_slug}_seed{cfg.seed}_best.pt")
    with open(OUT_DIR / f"{model_slug}_seed{cfg.seed}_normalization.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "config": asdict(cfg),
            "model_key": model_key,
            "model_label": model_label,
            "target_mean_ugm3": dict(zip(target_names, y_mean)),
            "target_std_ugm3": dict(zip(target_names, y_std)),
            "wind_features_not_standardized": [feature_names[i] for i in wind_features],
            "train_target_index_count": int(len(train_t_out)),
        }), f, indent=2)
    return reg_df, cls_df


def collect_existing_windno_categorical() -> pd.DataFrame:
    metrics_path = ROOT / "ML_3" / "logs" / "windno_2" / "aqi_class_metrics.json"
    pred_path = ROOT / "ML_3" / "results" / "windno_2" / "test_predictions_grid.npz"
    if not (metrics_path.exists() and pred_path.exists()):
        return pd.DataFrame()
    with open(metrics_path, "r", encoding="utf-8") as f:
        aqi = json.load(f)
    p = np.load(pred_path, allow_pickle=True)
    bin_pred = (p["y_cls_prob_grid"] >= 0.5).astype(np.int64)
    bm = binary_metrics(p["y_cls_true_grid"], bin_pred)
    return pd.DataFrame([{
        "model": "WindNO reference categorical checkpoint",
        **bm,
        "aqi_acc": float(aqi["overall_acc"]),
        "aqi_macro_f1": float(aqi["overall_macro_f1"]),
    }])


def existing_windno_regression_audit(data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    pred_path = ROOT / "ML_3" / "results" / "windno_2" / "test_predictions_grid.npz"
    if not pred_path.exists():
        return pd.DataFrame()
    p = np.load(pred_path, allow_pickle=True)
    true_ugm3 = p["y_reg_true_grid"].astype(np.float64) * UGM3_FACTOR
    raw_pred_ugm3 = p["y_reg_pred_grid"].astype(np.float64) * UGM3_FACTOR
    clipped_pred_ugm3 = np.maximum(raw_pred_ugm3, 0.0)
    rows = []
    for label, pred in [("raw", raw_pred_ugm3), ("nonnegative_clipped", clipped_pred_ugm3)]:
        reg_rows, mean_rmse = compute_regression(true_ugm3, pred, data["target_names"])
        for row in reg_rows:
            row["audit"] = label
            row["negative_fraction"] = float((raw_pred_ugm3 < 0).mean())
            rows.append(row)
        rows.append({"audit": label, "target": "mean", "mae_ugm3": np.nan, "rmse_ugm3": mean_rmse, "negative_fraction": float((raw_pred_ugm3 < 0).mean())})
    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    path.write_text(df.to_latex(index=False, escape=False, float_format=float_format), encoding="utf-8")


def aggregate_seed_outputs() -> None:
    cls_files = sorted(OUT_DIR.glob("*_seed*_classification.csv"))
    reg_files = sorted(OUT_DIR.glob("*_seed*_regression_ugm3.csv"))
    if cls_files:
        cls_raw = []
        for path in cls_files:
            seed = int(path.stem.split("seed", 1)[1].split("_", 1)[0])
            df = pd.read_csv(path)
            df["seed"] = seed
            cls_raw.append(df)
        cls_raw_df = pd.concat(cls_raw, ignore_index=True)
        metric_cols = [
            "bin_acc", "bin_prec", "bin_rec", "bin_f1",
            "aqi_acc", "aqi_macro_f1", "reg_rmse_mean_ugm3",
        ]
        cls_agg = cls_raw_df.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
        cls_agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in cls_agg.columns]
        cls_seed_counts = cls_raw_df.groupby("model")["seed"].nunique().rename("seed_count").reset_index()
        cls_agg = cls_agg.merge(cls_seed_counts, on="model", how="left")
        cls_raw_df.to_csv(OUT_DIR / "learned_model_classification_seed_summary_raw.csv", index=False)
        cls_agg.to_csv(OUT_DIR / "learned_model_classification_seed_summary_agg.csv", index=False)
        cls_raw_df.to_csv(OUT_DIR / "windno_classification_seed_summary_raw.csv", index=False)
        cls_agg.to_csv(OUT_DIR / "windno_classification_seed_summary_agg.csv", index=False)

        baseline_path = OUT_DIR / "deterministic_baseline_classification.csv"
        if baseline_path.exists():
            base = pd.read_csv(baseline_path)
            base["seed_count"] = 0
            for col in metric_cols:
                base[f"{col}_mean"] = base[col] if col in base.columns else np.nan
                base[f"{col}_std"] = np.nan
            wind = cls_agg.copy()
            keep_cols = ["model", "seed_count"] + [f"{c}_mean" for c in metric_cols] + [f"{c}_std" for c in metric_cols]
            comparison = pd.concat([base[keep_cols], wind[keep_cols]], ignore_index=True)
            comparison.to_csv(OUT_DIR / "categorical_comparison_agg.csv", index=False)

    if reg_files:
        reg_raw = []
        for path in reg_files:
            seed = int(path.stem.split("seed", 1)[1].split("_", 1)[0])
            df = pd.read_csv(path)
            df["seed"] = seed
            reg_raw.append(df)
        reg_raw_df = pd.concat(reg_raw, ignore_index=True)
        reg_agg = reg_raw_df.groupby(["model", "target"])[["mae_ugm3", "rmse_ugm3"]].agg(["mean", "std"]).reset_index()
        reg_agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in reg_agg.columns]
        mean_rows = []
        for model_name, g in reg_raw_df.groupby("model"):
            per_seed = g.groupby("seed")["rmse_ugm3"].mean()
            mean_rows.append({
                "model": model_name,
                "target": "mean",
                "mae_ugm3_mean": np.nan,
                "mae_ugm3_std": np.nan,
                "rmse_ugm3_mean": float(per_seed.mean()),
                "rmse_ugm3_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
            })
        if mean_rows:
            reg_agg = pd.concat([reg_agg, pd.DataFrame(mean_rows)], ignore_index=True)
        reg_raw_df.to_csv(OUT_DIR / "learned_model_regression_seed_summary_raw.csv", index=False)
        reg_agg.to_csv(OUT_DIR / "learned_model_regression_seed_summary_agg.csv", index=False)
        reg_raw_df.to_csv(OUT_DIR / "windno_regression_seed_summary_raw.csv", index=False)
        reg_agg.to_csv(OUT_DIR / "windno_regression_seed_summary_agg.csv", index=False)

        baseline_path = OUT_DIR / "deterministic_baseline_regression_ugm3.csv"
        if baseline_path.exists():
            base = pd.read_csv(baseline_path)
            base["seed_count"] = 0
            base["mae_ugm3_mean"] = base["mae_ugm3"]
            base["mae_ugm3_std"] = np.nan
            base["rmse_ugm3_mean"] = base["rmse_ugm3"]
            base["rmse_ugm3_std"] = np.nan
            reg_seed_counts = reg_raw_df.groupby("model")["seed"].nunique().rename("seed_count").reset_index()
            reg_agg = reg_agg.merge(reg_seed_counts, on="model", how="left")
            keep_cols = ["model", "target", "seed_count", "mae_ugm3_mean", "mae_ugm3_std", "rmse_ugm3_mean", "rmse_ugm3_std"]
            comparison = pd.concat([base[keep_cols], reg_agg[keep_cols]], ignore_index=True)
            comparison.to_csv(OUT_DIR / "regression_comparison_ugm3_agg.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-training", action="store_true", help="Train the selected learned model on CPU.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="windno", help="Learned model to train when --run-training is set.")
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.num_epochs)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    args = parser.parse_args()

    cfg = ExperimentConfig(run_training=args.run_training, num_epochs=args.epochs, seed=args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATASET_PATH, allow_pickle=True)

    feature_df = feature_inventory(data)
    freq_df = class_frequency_table(data, cfg)
    baseline_reg_df, baseline_cls_df = make_baseline_predictions(data, cfg)
    reference_cls_df = collect_existing_windno_categorical()
    audit_df = existing_windno_regression_audit(data)

    feature_df.to_csv(OUT_DIR / "feature_inventory.csv", index=False)
    freq_df.to_csv(OUT_DIR / "class_frequencies_by_split.csv", index=False)
    baseline_reg_df.to_csv(OUT_DIR / "deterministic_baseline_regression_ugm3.csv", index=False)
    baseline_cls_df.to_csv(OUT_DIR / "deterministic_baseline_classification.csv", index=False)
    if not reference_cls_df.empty:
        reference_cls_df.to_csv(OUT_DIR / "reference_windno_categorical_metrics.csv", index=False)
    if not audit_df.empty:
        audit_df.to_csv(OUT_DIR / "reference_windno_regression_unit_audit.csv", index=False)

    combined_cls = pd.concat([baseline_cls_df, reference_cls_df], ignore_index=True)
    combined_cls.to_csv(OUT_DIR / "categorical_baseline_comparison.csv", index=False)

    if cfg.run_training:
        model_slug = MODEL_SPECS[args.model]["slug"]
        reg_df, cls_df = train_one_seed(cfg, data, args.model)
        reg_df.to_csv(OUT_DIR / f"{model_slug}_seed{cfg.seed}_regression_ugm3.csv", index=False)
        cls_df.to_csv(OUT_DIR / f"{model_slug}_seed{cfg.seed}_classification.csv", index=False)
        combined_reg = pd.concat([baseline_reg_df, reg_df], ignore_index=True)
        combined_cls = pd.concat([baseline_cls_df, cls_df], ignore_index=True)
        combined_reg.to_csv(OUT_DIR / "regression_comparison_ugm3.csv", index=False)
        combined_cls.to_csv(OUT_DIR / "categorical_comparison.csv", index=False)

    aggregate_seed_outputs()

    summary = {
        "config": asdict(cfg),
        "last_requested_model": args.model,
        "dataset": {
            "path": str(DATASET_PATH),
            "shape_X": list(data["X_4d"].shape),
            "shape_y_reg": list(data["y_reg_4d"].shape),
            "date_range": [str(np.datetime_as_string(data["dates_4d"][0], unit="D")), str(np.datetime_as_string(data["dates_4d"][-1], unit="D"))],
            "target_names": [str(x) for x in data["target_names"]],
            "pm25_threshold_ugm3": float(data["pm25_threshold"]),
            "aqi_class_names": [str(x) for x in data["aqi_class_names"]],
            "aqi_bounds_ugm3": {
                "pm25": data["aqi_bounds_pm25_ugm3"].tolist(),
                "pm10": data["aqi_bounds_pm10_ugm3"].tolist(),
                "go3": data["aqi_bounds_go3_ugm3"].tolist(),
                "no2": data["aqi_bounds_no2_ugm3"].tolist(),
            },
        },
        "split_counts": {k: int(len(v)) for k, v in split_target_indices(data["dates_4d"], cfg).items()},
        "written_outputs": sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.glob("*")),
    }
    with open(OUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("experiment artifacts written to:", OUT_DIR.resolve())
    print("Split counts:", summary["split_counts"])


if __name__ == "__main__":
    main()
