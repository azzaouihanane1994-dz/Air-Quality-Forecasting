#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event-oriented diagnostics for the benchmark WindNO checkpoints.

The diagnostics are intentionally simple and reproducible:
  - high-PM10 days: top 10% of test days by true domain-mean PM10
  - active-fire cells: target-day FIRMS fire_label > 0
  - strong-wind days: top quartile by input-day 10 m wind speed

These diagnostics are not independent attribution studies; they are targeted
checks that the benchmark model maintains skill during transport-relevant and
episodic-source regimes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import experiments as exp


OUT_DIR = exp.OUT_DIR
DEVICE = "cpu"


def _load_norm(path: Path, target_names, feature_names):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    y_mean = np.array([payload["target_mean_ugm3"][str(t)] for t in target_names], dtype=np.float32)
    y_std = np.array([payload["target_std_ugm3"][str(t)] for t in target_names], dtype=np.float32)

    data = np.load(exp.DATASET_PATH, allow_pickle=True)
    windno = exp.load_windno_module()
    wind_idx = windno.find_wind_indices(feature_names)
    x_mean, x_std, _, _, _ = exp.compute_normalization(data, exp.ExperimentConfig(seed=int(payload["config"]["seed"])), wind_idx)
    return x_mean, x_std, y_mean, y_std


def _predict_seed(seed: int):
    data = np.load(exp.DATASET_PATH, allow_pickle=True)
    feature_names = list(data["feature_names"])
    target_names = list(data["target_names"])
    norm_path = OUT_DIR / f"windno_seed{seed}_normalization.json"
    ckpt_path = OUT_DIR / f"windno_seed{seed}_best.pt"
    x_mean, x_std, y_mean, y_std = _load_norm(norm_path, target_names, feature_names)

    cfg = exp.ExperimentConfig(seed=seed)
    ds = exp.NormalizedAirDataset(data, "test", cfg, x_mean, x_std, y_mean, y_std)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    windno = exp.load_windno_module()
    wind_idx = windno.find_wind_indices(feature_names)
    model = windno.WindNO(
        F_exo=len(feature_names),
        C_targets=len(target_names),
        hidden_dim=windno.cfg.hidden_dim,
        wind_idx=wind_idx,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    pred_norm, true_norm, y_cls_prob, y_cls_true, y_aqi_pred, y_aqi_true = [], [], [], [], [], []
    with torch.no_grad():
        for x, y_reg, y_cls, y_aqi, _ in loader:
            y_reg_pred, y_cls_logits, y_grp_logits, y_g0_logits, y_g1_logits = model(x.float())
            _, pred_aqi = windno.hier_aqi_probs_and_pred(
                y_grp_logits, y_g0_logits, y_g1_logits, temp=windno.cfg.coarse_temperature
            )
            pred_norm.append(y_reg_pred.numpy())
            true_norm.append(y_reg.numpy())
            y_cls_prob.append(torch.sigmoid(y_cls_logits).numpy())
            y_cls_true.append(y_cls.numpy())
            y_aqi_pred.append(pred_aqi.unsqueeze(1).numpy())
            y_aqi_true.append(y_aqi.numpy())

    pred_ugm3 = exp.denorm_reg(np.concatenate(pred_norm), y_mean, y_std)
    true_ugm3 = exp.denorm_reg(np.concatenate(true_norm), y_mean, y_std)
    return {
        "seed": seed,
        "pred_ugm3": pred_ugm3,
        "true_ugm3": true_ugm3,
        "y_cls_prob": np.concatenate(y_cls_prob),
        "y_cls_true": np.concatenate(y_cls_true),
        "y_aqi_pred": np.concatenate(y_aqi_pred),
        "y_aqi_true": np.concatenate(y_aqi_true),
    }


def _subset_binary_metrics(y_true, y_prob, mask):
    y_t = y_true[mask[:, None, :, :]].reshape(-1)
    y_p = (y_prob[mask[:, None, :, :]] >= 0.5).astype(int).reshape(-1)
    return exp.binary_metrics(y_t, y_p)


def _subset_aqi_macro(y_true, y_pred, mask):
    t = y_true[mask[:, None, :, :]].reshape(-1)
    p = y_pred[mask[:, None, :, :]].reshape(-1)
    return exp.macro_f1(t, p, 4)


def _subset_rmse(true_ugm3, pred_ugm3, mask, target_idx):
    err = pred_ugm3[:, target_idx][mask] - true_ugm3[:, target_idx][mask]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan


def build_event_masks(data, cfg: exp.ExperimentConfig):
    target_idx = exp.split_target_indices(data["dates_4d"], cfg)["test"]
    input_idx = target_idx - cfg.horizon
    y_true = data["y_reg_4d"][target_idx].astype(np.float64) * exp.UGM3_FACTOR

    pm10_daily = y_true[:, 1].mean(axis=(1, 2))
    high_pm10_day = pm10_daily >= np.quantile(pm10_daily, 0.90)
    masks = {
        "all_test_cells": np.ones((len(target_idx), y_true.shape[2], y_true.shape[3]), dtype=bool),
        "high_pm10_days_top10pct": np.broadcast_to(high_pm10_day[:, None, None], (len(target_idx), y_true.shape[2], y_true.shape[3])).copy(),
    }

    features = [str(f) for f in data["feature_names"]]
    if "fire_label" in features:
        fire_idx = features.index("fire_label")
        masks["active_fire_cells"] = data["X_4d"][target_idx, fire_idx] > 0

    if "eac4:u10" in features and "eac4:v10" in features:
        u = data["X_4d"][input_idx, features.index("eac4:u10")]
        v = data["X_4d"][input_idx, features.index("eac4:v10")]
        wind_daily = np.sqrt(u ** 2 + v ** 2).mean(axis=(1, 2))
        high_wind = wind_daily >= np.quantile(wind_daily, 0.75)
        masks["strong_wind_days_top25pct"] = np.broadcast_to(high_wind[:, None, None], (len(target_idx), y_true.shape[2], y_true.shape[3])).copy()
    return masks


def main():
    data = np.load(exp.DATASET_PATH, allow_pickle=True)
    cfg = exp.ExperimentConfig()
    masks = build_event_masks(data, cfg)
    seed_files = sorted(OUT_DIR.glob("windno_seed*_best.pt"))
    seeds = [int(p.stem.split("seed", 1)[1].split("_", 1)[0]) for p in seed_files]
    if not seeds:
        raise SystemExit("No benchmark checkpoints found. Run experiments.py --run-training first.")

    raw_rows = []
    for seed in seeds:
        pred = _predict_seed(seed)
        for event, mask in masks.items():
            bm = _subset_binary_metrics(pred["y_cls_true"], pred["y_cls_prob"], mask)
            raw_rows.append({
                "seed": seed,
                "event": event,
                "n_cells": int(mask.sum()),
                "pm25_rmse_ugm3": _subset_rmse(pred["true_ugm3"], pred["pred_ugm3"], mask, 0),
                "pm10_rmse_ugm3": _subset_rmse(pred["true_ugm3"], pred["pred_ugm3"], mask, 1),
                "bin_f1": bm["bin_f1"],
                "aqi_macro_f1": _subset_aqi_macro(pred["y_aqi_true"], pred["y_aqi_pred"], mask),
            })

    raw = pd.DataFrame(raw_rows)
    agg = raw.groupby("event")[["n_cells", "pm25_rmse_ugm3", "pm10_rmse_ugm3", "bin_f1", "aqi_macro_f1"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in agg.columns]
    raw.to_csv(OUT_DIR / "event_diagnostics_raw.csv", index=False)
    agg.to_csv(OUT_DIR / "event_diagnostics_agg.csv", index=False)
    print("Event diagnostics written to:", OUT_DIR.resolve())
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
