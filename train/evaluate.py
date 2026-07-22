from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wide_fov_supervision_v2.config import PipelineConfig
from wide_fov_supervision_v2.datasets.nyu.quad_dataset import NYUQuadCompletionDataset
from wide_fov_supervision_v2.modules.quad_completion.model import QuadRayCompletionModel
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint
from wide_fov_supervision_v2.train.trainer import _move_batch, predict_batch_with_stencils


def depth_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target) & (target > 0.0)
    if not np.any(valid):
        return {"abs_rel": float("nan"), "rmse": float("nan")}
    error = pred[valid] - target[valid]
    return {
        "abs_rel": float(np.mean(np.abs(error) / target[valid].clip(min=1.0e-6))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _classification_counts(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(prediction, dtype=bool)[mask]
    truth = np.asarray(target, dtype=bool)[mask]
    return (
        int(np.sum(pred & truth)),
        int(np.sum(pred & ~truth)),
        int(np.sum(~pred & truth)),
        int(np.sum(~pred & ~truth)),
    )


def _finish_classification(counts: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(value) for value in counts)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1.0e-12, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_cached_predictions(config: PipelineConfig) -> Path:
    """NYU test quad에서 bilinear baseline과 completion을 동일 query로 비교한다."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NYUQuadCompletionDataset(config, "test")
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
    model = QuadRayCompletionModel(config.completion).to(device).eval()
    checkpoint_loaded = False
    if config.paths.checkpoint is not None:
        checkpoint_path = Path(config.paths.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Completion checkpoint가 없습니다: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, map_location=device)
        checkpoint_loaded = True

    store: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "pred_rgb", "base_rgb", "target_rgb", "pred_depth", "base_depth", "target_depth",
            "valid", "confidence", "valid_prob", "confidence_prob", "continuous", "normal_error",
        )
    }
    with torch.no_grad():
        for raw_batch in tqdm(loader, desc="evaluate quad completion"):
            batch = _move_batch(raw_batch, device)
            result, pred_normal, target_normal, normal_mask = predict_batch_with_stencils(
                model, batch, z_eps=config.camera.geometry_z_eps
            )
            cosine = torch.sum(pred_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
            normal_error = torch.rad2deg(torch.acos(cosine))
            values = {
                "pred_rgb": result.rgb,
                "base_rgb": result.base_rgb,
                "target_rgb": batch["target_rgb"],
                "pred_depth": result.depth_z,
                "base_depth": result.base_depth_z,
                "target_depth": batch["target_depth_z"],
                "valid": batch["target_valid"],
                "confidence": batch["target_confidence"],
                "valid_prob": torch.sigmoid(result.valid_logit),
                "confidence_prob": torch.sigmoid(result.confidence_logit),
                "continuous": batch["source_continuous"][:, None].expand_as(batch["query_mask"]),
                "normal_error": torch.where(normal_mask, normal_error, torch.full_like(normal_error, float("nan"))),
            }
            for name, value in values.items():
                store[name].append(value.detach().cpu().numpy())
    arrays = {name: np.concatenate(values, axis=0) for name, values in store.items()}
    valid = arrays["valid"].astype(bool)
    target_rgb = arrays["target_rgb"]
    pred_rgb_error = np.abs(arrays["pred_rgb"] - target_rgb).mean(axis=-1)
    base_rgb_error = np.abs(arrays["base_rgb"] - target_rgb).mean(axis=-1)
    pred_mse = float(np.mean((arrays["pred_rgb"][valid] - target_rgb[valid]) ** 2)) if np.any(valid) else float("nan")
    base_mse = float(np.mean((arrays["base_rgb"][valid] - target_rgb[valid]) ** 2)) if np.any(valid) else float("nan")
    metrics: dict = {
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint": str(config.paths.checkpoint) if config.paths.checkpoint is not None else None,
        "sample_count": len(dataset),
        "rgb": {
            "completion_mae": float(np.mean(pred_rgb_error[valid])) if np.any(valid) else float("nan"),
            "bilinear_mae": float(np.mean(base_rgb_error[valid])) if np.any(valid) else float("nan"),
            "completion_psnr": float(-10.0 * np.log10(max(pred_mse, 1.0e-12))),
            "bilinear_psnr": float(-10.0 * np.log10(max(base_mse, 1.0e-12))),
        },
        "depth": {
            "completion": depth_metrics(arrays["pred_depth"], arrays["target_depth"], valid),
            "bilinear": depth_metrics(arrays["base_depth"], arrays["target_depth"], valid),
        },
        "normal_angular_error_degrees": {
            "mean": float(np.nanmean(arrays["normal_error"])),
            "median": float(np.nanmedian(arrays["normal_error"])),
        },
    }
    query_mask = np.ones_like(valid, dtype=bool)
    valid_counts = np.asarray(_classification_counts(arrays["valid_prob"] >= 0.5, valid, query_mask))
    confidence_counts = np.asarray(
        _classification_counts(arrays["confidence_prob"] >= 0.5, arrays["confidence"].astype(bool), query_mask)
    )
    metrics["valid_classification"] = _finish_classification(valid_counts)
    metrics["confidence_classification"] = _finish_classification(confidence_counts)
    confident = valid & (arrays["valid_prob"] >= 0.5) & (arrays["confidence_prob"] >= 0.5)
    metrics["confidence_filtered_depth"] = depth_metrics(
        arrays["pred_depth"], arrays["target_depth"], confident
    )
    for group, group_mask in {
        "continuous": arrays["continuous"].astype(bool),
        "edge": ~arrays["continuous"].astype(bool),
    }.items():
        metrics[f"{group}_depth"] = depth_metrics(
            arrays["pred_depth"], arrays["target_depth"], valid & group_mask
        )

    out_dir = config.paths.outputs / "eval" / time.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
