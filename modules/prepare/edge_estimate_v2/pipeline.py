"""세 variant의 학습·평가·fisheye 추론·3D polyline·BEV·보고서를 연결한다."""

from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wide_fov_supervision_v2.backbone.depth_anything import DepthAnythingMetricWrapper
from wide_fov_supervision_v2.modules.camera_geometry import (
    build_camera_rays,
    build_fisheye_rays,
    camera_to_world_points,
    cell_angular_gap,
    points_from_z_depth,
    project_camera_rays,
    project_fisheye_rays,
)
from wide_fov_supervision_v2.modules.visualization import save_depth, save_heatmap, save_rgb

from .config import (
    EdgeEstimateConfig,
    Variant,
    VARIANTS,
    config_to_dict,
    ensure_edge_output_roots,
    save_edge_config,
)
from .dataset import (
    EdgePatchDataset,
    _query_grid,
    _sample_bilinear,
    _sample_bilinear_masked,
    _sample_nearest,
    spherical_bilerp,
)
from .edge_prior import cell_edge_prior, estimate_2d_edge_prior
from .losses import EdgeEstimateLoss
from .model import EdgeEstimateModel, EdgeEstimateResult
from .pseudo_labels import EDGE_OCCLUSION, build_pseudo_edges
from .report import generate_edge_report


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward(model: EdgeEstimateModel, batch: dict[str, torch.Tensor]) -> EdgeEstimateResult:
    return model(
        batch["support_rgb"],
        batch["support_ray_dir"],
        batch["support_valid"],
        batch["query_ray_dir"],
        batch["query_relative_uv"],
        da_relative_log_depth=batch.get("da_relative_log_depth"),
        da_valid=batch.get("da_valid"),
        support_edge_2d=batch.get("support_edge_2d"),
        query_prior_depth_z=batch.get("query_prior_depth_z"),
        query_prior_valid=batch.get("query_prior_valid"),
    )


def save_checkpoint(
    path: Path,
    model: EdgeEstimateModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": model.checkpoint_schema,
            "variant": model.variant,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "log_depth_mean": model.log_depth_mean,
            "log_depth_std": model.log_depth_std,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: EdgeEstimateModel,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != model.checkpoint_schema:
        raise RuntimeError(
            f"Edge checkpoint schema 불일치: expected={model.checkpoint_schema!r}, actual={payload.get('schema')!r}"
        )
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def _compatible_training_run(
    config: EdgeEstimateConfig,
    variant: Variant,
) -> tuple[Path, Path | None, bool] | None:
    """동일 설정의 최신 학습을 찾아 완료 재사용 또는 중단 지점 재개에 사용한다.

    반환값은 ``(run_dir, resume_checkpoint, complete)``이다. 설정 전체가 같은
    run만 선택하므로 smoke 설정이나 다른 variant의 checkpoint를 섞지 않는다.
    """

    variant_root = config.output_root / "train" / variant
    expected = config_to_dict(config)
    if not variant_root.exists():
        return None
    best_incomplete: tuple[Path, Path, int] | None = None
    for run_dir in sorted((path for path in variant_root.iterdir() if path.is_dir()), reverse=True):
        config_path = run_dir / "config.json"
        try:
            if json.loads(config_path.read_text(encoding="utf-8")) != expected:
                continue
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        best = run_dir / "checkpoints" / "best.pt"
        last = run_dir / "checkpoints" / "last.pt"
        if last.exists() and best.exists():
            return run_dir, last, True
        epoch_checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if epoch_checkpoints:
            payload = torch.load(epoch_checkpoints[-1], map_location="cpu", weights_only=False)
            completed_epoch = int(payload.get("epoch", 0))
            if completed_epoch < int(config.train.epochs) and (
                best_incomplete is None or completed_epoch > best_incomplete[2]
            ):
                best_incomplete = (run_dir, epoch_checkpoints[-1], completed_epoch)
    if best_incomplete is None:
        return None
    return best_incomplete[0], best_incomplete[1], False


def _binary_metrics(probability: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(probability) & np.isfinite(target)
    if not np.any(valid):
        return {name: float("nan") for name in ("precision", "recall", "f1", "auroc", "auprc")} | {"count": 0}
    score = np.asarray(probability)[valid].astype(np.float64)
    truth = np.asarray(target)[valid] >= 0.5
    predicted = score >= 0.5
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    order = np.argsort(-score)
    sorted_truth = truth[order].astype(np.float64)
    positives = max(float(sorted_truth.sum()), 1.0)
    negatives = max(float(len(sorted_truth) - sorted_truth.sum()), 1.0)
    tp_curve = np.cumsum(sorted_truth)
    fp_curve = np.cumsum(1.0 - sorted_truth)
    recall_curve = np.concatenate([[0.0], tp_curve / positives, [1.0]])
    precision_curve = np.concatenate([[1.0], tp_curve / np.maximum(tp_curve + fp_curve, 1.0), [sorted_truth.mean()]])
    fpr_curve = np.concatenate([[0.0], fp_curve / negatives, [1.0]])
    tpr_curve = recall_curve
    return {
        "count": int(valid.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(np.trapz(tpr_curve, fpr_curve)),
        "auprc": float(np.trapz(precision_curve, recall_curve)),
    }


def _depth_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & (prediction > 0.0)
    valid &= np.isfinite(target) & (target > 0.0)
    if not np.any(valid):
        return {"count": 0, "absrel": float("nan"), "rmse": float("nan")}
    error = np.asarray(prediction)[valid] - np.asarray(target)[valid]
    return {
        "count": int(valid.sum()),
        "absrel": float(np.mean(np.abs(error) / np.asarray(target)[valid])),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _point_set_metrics(prediction: np.ndarray, target: np.ndarray, threshold_m: float) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    pred = pred[np.isfinite(pred).all(axis=-1)]
    truth = truth[np.isfinite(truth).all(axis=-1)]
    if len(pred) == 0 or len(truth) == 0:
        return {"prediction_count": len(pred), "target_count": len(truth), "precision": 0.0, "recall": 0.0, "chamfer_m": float("nan")}
    pred_distance = cKDTree(truth).query(pred, k=1)[0]
    target_distance = cKDTree(pred).query(truth, k=1)[0]
    return {
        "prediction_count": len(pred),
        "target_count": len(truth),
        "precision": float(np.mean(pred_distance <= threshold_m)),
        "recall": float(np.mean(target_distance <= threshold_m)),
        "chamfer_m": float(0.5 * (np.mean(pred_distance) + np.mean(target_distance))),
    }


def _confidence_calibration(probability: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(probability) & np.isfinite(target)
    if not np.any(valid):
        return {"count": 0, "brier": float("nan"), "ece": float("nan")}
    score = np.asarray(probability, dtype=np.float64)[valid]
    truth = np.asarray(target, dtype=np.float64)[valid]
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (score >= lower) & (score < upper if upper < 1.0 else score <= upper)
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(float(np.mean(score[selected])) - float(np.mean(truth[selected])))
    return {"count": int(valid.sum()), "brier": float(np.mean((score - truth) ** 2)), "ece": float(ece)}


def _tolerant_edge_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    side: int,
    tolerance: int = 1,
) -> dict[str, float]:
    prediction = (probability >= 0.5) & mask
    truth = (target >= 0.5) & mask
    shape = prediction.shape[:-1] + (side, side)
    prediction = prediction.reshape(shape)
    truth = truth.reshape(shape)
    footprint = (1,) * (prediction.ndim - 2) + (2 * tolerance + 1, 2 * tolerance + 1)
    prediction_near = maximum_filter(prediction.astype(np.uint8), size=footprint) > 0
    truth_near = maximum_filter(truth.astype(np.uint8), size=footprint) > 0
    precision = float(np.sum(prediction & truth_near) / max(np.sum(prediction), 1))
    recall = float(np.sum(truth & prediction_near) / max(np.sum(truth), 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-12)),
    }


def _boundary_disagreement(probability: np.ndarray, mask: np.ndarray, side: int) -> float:
    p = probability.reshape(*probability.shape[:-1], side, side)
    m = mask.reshape(*mask.shape[:-1], side, side).astype(bool)
    horizontal_mask = m[:, :, :-1, :, -1] & m[:, :, 1:, :, 0]
    vertical_mask = m[:, :-1, :, -1, :] & m[:, 1:, :, 0, :]
    differences: list[np.ndarray] = []
    horizontal = np.abs(p[:, :, :-1, :, -1] - p[:, :, 1:, :, 0])
    vertical = np.abs(p[:, :-1, :, -1, :] - p[:, 1:, :, 0, :])
    if np.any(horizontal_mask):
        differences.append(horizontal[horizontal_mask])
    if np.any(vertical_mask):
        differences.append(vertical[vertical_mask])
    return float(np.mean(np.concatenate(differences))) if differences else float("nan")


def _component_counts(
    probability: np.ndarray, target: np.ndarray, mask: np.ndarray, side: int
) -> tuple[np.ndarray, np.ndarray]:
    resolution = 4 * (side - 1) + 1
    predicted_counts: list[int] = []
    target_counts: list[int] = []
    for patch_index in range(len(probability)):
        predicted_canvas = np.zeros((resolution, resolution), dtype=np.uint8)
        target_canvas = np.zeros((resolution, resolution), dtype=np.uint8)
        for cell_y in range(4):
            for cell_x in range(4):
                valid = mask[patch_index, cell_y, cell_x].reshape(side, side)
                pred = (probability[patch_index, cell_y, cell_x].reshape(side, side) >= 0.5) & valid
                truth = (target[patch_index, cell_y, cell_x].reshape(side, side) >= 0.5) & valid
                y0, x0 = cell_y * (side - 1), cell_x * (side - 1)
                predicted_canvas[y0 : y0 + side, x0 : x0 + side] |= pred.astype(np.uint8)
                target_canvas[y0 : y0 + side, x0 : x0 + side] |= truth.astype(np.uint8)
        predicted_counts.append(max(cv2.connectedComponents(predicted_canvas, connectivity=8)[0] - 1, 0))
        target_counts.append(max(cv2.connectedComponents(target_canvas, connectivity=8)[0] - 1, 0))
    return np.asarray(predicted_counts, dtype=np.float32), np.asarray(target_counts, dtype=np.float32)


def _fragmentation(probability: np.ndarray, target: np.ndarray, mask: np.ndarray, side: int) -> dict[str, float]:
    """4×4 cell query를 하나의 lattice raster로 합쳐 contour 조각 수를 센다."""

    predicted_array, target_array = _component_counts(probability, target, mask, side)
    return {
        "predicted_components_mean": float(np.mean(predicted_array)) if len(predicted_array) else float("nan"),
        "target_components_mean": float(np.mean(target_array)) if len(target_array) else float("nan"),
        "excess_components_mean": float(np.mean(np.maximum(predicted_array - target_array, 0.0)))
        if len(predicted_array)
        else float("nan"),
    }


def _points_torch(depth: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    return depth.unsqueeze(-1) / rays[..., 2:3].clamp_min(1.0e-6) * rays


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    return float(np.mean(array[finite])) if np.any(finite) else float("nan")


def _sample_geometry_metrics(
    probability: torch.Tensor,
    prediction_depth: torch.Tensor,
    target_edge: torch.Tensor,
    target_depth: torch.Tensor,
    target_valid: torch.Tensor,
    rays: torch.Tensor,
    max_points: int = 256,
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    pred_points = _points_torch(prediction_depth, rays)
    target_points = _points_torch(target_depth.clamp_min(1.0e-6), rays)
    for index in range(probability.shape[0]):
        pred_mask = probability[index] >= 0.5
        gt_mask = (target_edge[index] >= 0.5) & target_valid[index].bool()
        pred = pred_points[index][pred_mask].reshape(-1, 3)
        target = target_points[index][gt_mask].reshape(-1, 3)
        if len(pred) == 0 or len(target) == 0:
            for name in (
                "chamfer_3d",
                "precision_3d",
                "recall_3d",
                "chamfer_bev",
                "precision_bev",
                "recall_bev",
            ):
                values[name].append(float("nan"))
            continue
        pred = pred[:max_points]
        target = target[:max_points]
        distance = torch.cdist(pred.float(), target.float())
        pred_min = distance.min(dim=1).values
        target_min = distance.min(dim=0).values
        values["chamfer_3d"].append(float(0.5 * (pred_min.mean() + target_min.mean())))
        values["precision_3d"].append(float((pred_min <= 0.05).float().mean()))
        values["recall_3d"].append(float((target_min <= 0.05).float().mean()))
        bev_distance = torch.cdist(pred[:, (0, 2)].float(), target[:, (0, 2)].float())
        bev_pred_min = bev_distance.min(dim=1).values
        bev_target_min = bev_distance.min(dim=0).values
        values["chamfer_bev"].append(float(0.5 * (bev_pred_min.mean() + bev_target_min.mean())))
        values["precision_bev"].append(float((bev_pred_min <= 0.04).float().mean()))
        values["recall_bev"].append(float((bev_target_min <= 0.04).float().mean()))
    return values


@torch.inference_mode()
def evaluate_model(
    model: EdgeEstimateModel,
    dataset: EdgePatchDataset,
    config: EdgeEstimateConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[int, dict[str, float]]]:
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
    model.eval()
    cell_probabilities: list[np.ndarray] = []
    cell_targets: list[np.ndarray] = []
    cell_masks: list[np.ndarray] = []
    query_probabilities: list[np.ndarray] = []
    query_targets: list[np.ndarray] = []
    query_masks: list[np.ndarray] = []
    near_predictions: list[np.ndarray] = []
    near_targets: list[np.ndarray] = []
    near_masks: list[np.ndarray] = []
    far_predictions: list[np.ndarray] = []
    far_targets: list[np.ndarray] = []
    far_masks: list[np.ndarray] = []
    confidence_probabilities: list[np.ndarray] = []
    confidence_targets: list[np.ndarray] = []
    type_confusion = np.zeros((3, 3), dtype=np.int64)
    geometry_values: dict[str, list[float]] = defaultdict(list)
    frame_counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    frame_geometry: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    frame_fragmentation: dict[int, list[float]] = defaultdict(list)
    for raw_batch in tqdm(loader, desc=f"evaluate {model.variant}", leave=False, disable=not sys.stderr.isatty()):
        batch = _move_batch(raw_batch, device)
        result = _forward(model, batch)
        cell_probability = torch.sigmoid(result.cell_edge_logit)
        query_probability = torch.sigmoid(result.query_edge_logit)
        cell_probabilities.append(cell_probability.cpu().numpy())
        cell_targets.append(batch["target_cell_edge"].cpu().numpy())
        cell_masks.append(batch["cell_valid"].cpu().numpy())
        query_probabilities.append(query_probability.cpu().numpy())
        query_targets.append(batch["target_query_edge"].cpu().numpy())
        query_masks.append(batch["query_mask"].cpu().numpy())
        near_predictions.append(result.query_depth_near_z.cpu().numpy())
        near_targets.append(batch["target_near_depth_z"].cpu().numpy())
        near_masks.append(batch["target_near_valid"].cpu().numpy())
        far_predictions.append(result.query_depth_far_z.cpu().numpy())
        far_targets.append(batch["target_far_depth_z"].cpu().numpy())
        far_masks.append(batch["target_far_valid"].cpu().numpy())
        confidence_probabilities.append(torch.sigmoid(result.query_confidence_logit).cpu().numpy())
        confidence_targets.append(batch["target_confidence"].cpu().numpy())
        type_prediction = result.cell_type_logits.argmax(dim=-1) + 1
        type_target = batch["target_cell_type"].long()
        type_mask = batch["cell_valid"].bool() & (type_target > 0)
        for gt, pred in zip(type_target[type_mask].cpu().numpy(), type_prediction[type_mask].cpu().numpy()):
            type_confusion[int(gt) - 1, int(pred) - 1] += 1
        sample_geometry = _sample_geometry_metrics(
            query_probability,
            result.query_depth_near_z,
            batch["target_query_edge"],
            batch["target_near_depth_z"],
            batch["target_near_valid"],
            batch["query_ray_dir"],
        )
        for name, values in sample_geometry.items():
            geometry_values[name].extend(values)
        predicted = query_probability >= 0.5
        truth = batch["target_query_edge"] >= 0.5
        valid = batch["query_mask"].bool()
        batch_probability = query_probability.cpu().numpy()
        batch_target = batch["target_query_edge"].cpu().numpy()
        batch_mask = valid.cpu().numpy()
        predicted_components, target_components = _component_counts(
            batch_probability, batch_target, batch_mask, int(config.data.query_grid_size)
        )
        for row, frame_index_value in enumerate(batch["frame_index"].cpu().numpy().reshape(-1)):
            frame_index = int(frame_index_value)
            frame_counts[frame_index] += np.array(
                [
                    int(torch.sum(predicted[row] & truth[row] & valid[row]).cpu()),
                    int(torch.sum(predicted[row] & ~truth[row] & valid[row]).cpu()),
                    int(torch.sum(~predicted[row] & truth[row] & valid[row]).cpu()),
                ]
            )
            for name, values in sample_geometry.items():
                if row < len(values) and np.isfinite(values[row]):
                    frame_geometry[frame_index][name].append(values[row])
            frame_fragmentation[frame_index].append(
                float(max(predicted_components[row] - target_components[row], 0.0))
            )
    cell_probability_array = np.concatenate(cell_probabilities)
    cell_target_array = np.concatenate(cell_targets)
    cell_mask_array = np.concatenate(cell_masks)
    query_probability_array = np.concatenate(query_probabilities)
    query_target_array = np.concatenate(query_targets)
    query_mask_array = np.concatenate(query_masks)
    cell = _binary_metrics(cell_probability_array, cell_target_array, cell_mask_array)
    query = _binary_metrics(query_probability_array, query_target_array, query_mask_array)
    near = _depth_metrics(np.concatenate(near_predictions), np.concatenate(near_targets), np.concatenate(near_masks))
    far = _depth_metrics(np.concatenate(far_predictions), np.concatenate(far_targets), np.concatenate(far_masks))
    metrics: dict[str, float | int] = {f"cell_{key}": value for key, value in cell.items()}
    metrics.update({f"query_{key}": value for key, value in query.items()})
    metrics.update({f"near_depth_{key}": value for key, value in near.items()})
    metrics.update({f"far_depth_{key}": value for key, value in far.items()})
    side = int(config.data.query_grid_size)
    tolerant = _tolerant_edge_metrics(
        query_probability_array, query_target_array, query_mask_array, side=side, tolerance=1
    )
    metrics.update({f"query_tolerance_1_{key}": value for key, value in tolerant.items()})
    calibration = _confidence_calibration(
        np.concatenate(confidence_probabilities), np.concatenate(confidence_targets), query_mask_array
    )
    metrics.update({f"confidence_{key}": value for key, value in calibration.items()})
    confidence_binary = _binary_metrics(
        np.concatenate(confidence_probabilities), np.concatenate(confidence_targets), query_mask_array
    )
    metrics.update({f"confidence_{key}": value for key, value in confidence_binary.items()})
    metrics["shared_boundary_probability_disagreement"] = _boundary_disagreement(
        query_probability_array, query_mask_array, side
    )
    fragmentation = _fragmentation(query_probability_array, query_target_array, query_mask_array, side)
    metrics.update({f"contour_{key}": value for key, value in fragmentation.items()})
    metrics["edge_3d_chamfer_m"] = (
        _finite_mean(geometry_values["chamfer_3d"])
    )
    metrics["edge_3d_precision_5cm"] = (
        _finite_mean(geometry_values["precision_3d"])
    )
    metrics["edge_3d_recall_5cm"] = (
        _finite_mean(geometry_values["recall_3d"])
    )
    metrics["bev_xz_chamfer_m"] = (
        _finite_mean(geometry_values["chamfer_bev"])
    )
    bev_precision = _finite_mean(geometry_values["precision_bev"])
    bev_recall = _finite_mean(geometry_values["recall_bev"])
    metrics["bev_xz_precision_4cm"] = bev_precision
    metrics["bev_xz_recall_4cm"] = bev_recall
    metrics["bev_xz_f1_4cm"] = (
        float(2.0 * bev_precision * bev_recall / max(bev_precision + bev_recall, 1.0e-12))
        if np.isfinite(bev_precision) and np.isfinite(bev_recall)
        else float("nan")
    )
    for class_index, name in enumerate(("crease", "occlusion", "junction")):
        tp = type_confusion[class_index, class_index]
        fp = type_confusion[:, class_index].sum() - tp
        fn = type_confusion[class_index].sum() - tp
        metrics[f"type_{name}_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    per_frame: dict[int, dict[str, float]] = {}
    for frame_index, (tp, fp, fn) in frame_counts.items():
        per_frame[frame_index] = {
            "query_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "edge_3d_chamfer_m": _finite_mean(frame_geometry[frame_index]["chamfer_3d"]),
            "contour_fragmentation": _finite_mean(frame_fragmentation[frame_index]),
        }
    return metrics, per_frame


def train_variant(config: EdgeEstimateConfig, variant: Variant) -> Path:
    """한 variant를 학습하고 best/last checkpoint를 저장한다."""

    ensure_edge_output_roots(config)
    torch.manual_seed(int(config.train.seed))
    compatible_run = _compatible_training_run(config, variant)
    if compatible_run is not None and compatible_run[2]:
        return compatible_run[0] / "checkpoints" / "best.pt"
    device = _device()
    train_dataset = EdgePatchDataset(config, "train", variant)
    test_dataset = EdgePatchDataset(config, "test", variant)
    loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = EdgeEstimateModel(
        config.model,
        variant,
        log_depth_mean=train_dataset.log_depth_mean,
        log_depth_std=train_dataset.log_depth_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.train.amp and device.type == "cuda", init_scale=config.train.amp_initial_scale
    )
    loss_fn = EdgeEstimateLoss(config.loss)
    if compatible_run is None:
        run_dir = config.output_root / "train" / variant / time.strftime("%Y_%m_%d_%H_%M_%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        save_edge_config(config, run_dir / "config.json")
        history: list[dict] = []
        start_epoch = 0
    else:
        run_dir, resume_checkpoint, _ = compatible_run
        assert resume_checkpoint is not None
        payload = load_checkpoint(resume_checkpoint, model, optimizer, map_location=device)
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        start_epoch = int(payload.get("epoch", 0))
        print(f"{variant} 학습 재개: {resume_checkpoint} (다음 epoch={start_epoch + 1})")
    best_f1 = -1.0
    best_chamfer = float("inf")
    for item in history:
        f1 = float(item.get("val_query_f1", float("nan")))
        chamfer = float(item.get("val_edge_3d_chamfer_m", float("inf")))
        if np.isfinite(f1) and (f1 > best_f1 or (f1 == best_f1 and chamfer < best_chamfer)):
            best_f1, best_chamfer = f1, chamfer
    for epoch in range(start_epoch, int(config.train.epochs)):
        model.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        skipped_nonfinite_batches = 0
        consecutive_nonfinite_batches = 0
        progress = tqdm(loader, desc=f"{variant} epoch {epoch + 1}/{config.train.epochs}", disable=not sys.stderr.isatty())
        for raw_batch in progress:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.train.amp and device.type == "cuda"):
                result = _forward(model, batch)
                loss = loss_fn(result, batch)
            if not bool(torch.isfinite(loss.total)):
                raise FloatingPointError(f"{variant} loss에서 NaN/Inf가 발견되었습니다.")
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            if not gradients_finite:
                if not scaler.is_enabled():
                    raise FloatingPointError(f"{variant} gradient에서 NaN/Inf가 발견되었습니다.")
                # AMP overflow는 optimizer step을 건너뛰고 GradScaler가 scale을 낮추게 한다.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_batches += 1
                consecutive_nonfinite_batches += 1
                if consecutive_nonfinite_batches >= 20:
                    raise FloatingPointError(f"{variant} AMP gradient overflow가 20 batch 연속 발생했습니다.")
                progress.set_postfix(loss=sums["total"] / max(batches, 1), skipped=skipped_nonfinite_batches)
                continue
            consecutive_nonfinite_batches = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            for name in loss.__dataclass_fields__:
                sums[name] += float(getattr(loss, name).detach().cpu())
            batches += 1
            progress.set_postfix(loss=sums["total"] / batches)
        train_metrics = {f"train_{key}": value / max(batches, 1) for key, value in sums.items()}
        train_metrics["train_skipped_nonfinite_batches"] = skipped_nonfinite_batches
        validation, _ = evaluate_model(model, test_dataset, config, device)
        epoch_metrics = {"epoch": epoch + 1, **train_metrics, **{f"val_{key}": value for key, value in validation.items()}}
        history.append(epoch_metrics)
        save_checkpoint(
            run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt",
            model,
            optimizer,
            epoch + 1,
            epoch_metrics,
            scaler,
        )
        f1 = float(validation.get("query_f1", float("nan")))
        chamfer = float(validation.get("edge_3d_chamfer_m", float("inf")))
        if np.isfinite(f1) and (f1 > best_f1 or (f1 == best_f1 and chamfer < best_chamfer)):
            best_f1, best_chamfer = f1, chamfer
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch + 1, epoch_metrics, scaler)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    last = run_dir / "checkpoints" / "last.pt"
    save_checkpoint(last, model, optimizer, int(config.train.epochs), history[-1] if history else {}, scaler)
    if not (run_dir / "checkpoints" / "best.pt").exists():
        save_checkpoint(
            run_dir / "checkpoints" / "best.pt",
            model,
            optimizer,
            int(config.train.epochs),
            history[-1] if history else {},
            scaler,
        )
    return run_dir / "checkpoints" / "best.pt"


def _model_from_checkpoint(config: EdgeEstimateConfig, variant: Variant, checkpoint: Path, device: torch.device) -> EdgeEstimateModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EdgeEstimateModel(
        config.model,
        variant,
        log_depth_mean=float(payload.get("log_depth_mean", config.model.log_depth_mean)),
        log_depth_std=float(payload.get("log_depth_std", config.model.log_depth_std)),
    ).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    return model.eval()


def _paired_bootstrap(local: dict[int, dict[str, float]], context: dict[int, dict[str, float]], seed: int) -> dict:
    common = sorted(set(local) & set(context))
    if not common:
        return {"count": 0, "hypothesis_supported": False}
    rng = np.random.default_rng(seed)

    def estimate(metric: str, *, lower_is_better: bool) -> dict:
        values: list[float] = []
        for key in common:
            local_value = local[key].get(metric, float("nan"))
            context_value = context[key].get(metric, float("nan"))
            if np.isfinite(local_value) and np.isfinite(context_value):
                delta = local_value - context_value if lower_is_better else context_value - local_value
                values.append(float(delta))
        if not values:
            return {"count": 0, "mean_improvement": float("nan"), "ci95": [float("nan"), float("nan")]}
        differences = np.asarray(values, dtype=np.float64)
        samples = np.empty(2000, dtype=np.float64)
        for index in range(len(samples)):
            samples[index] = np.mean(rng.choice(differences, size=len(differences), replace=True))
        return {
            "count": len(differences),
            "mean_improvement": float(np.mean(differences)),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }

    f1 = estimate("query_f1", lower_is_better=False)
    chamfer = estimate("edge_3d_chamfer_m", lower_is_better=True)
    fragmentation = estimate("contour_fragmentation", lower_is_better=True)
    f1_supported = f1["count"] > 0 and f1["ci95"][0] > 0.0
    geometry_supported = (
        (chamfer["count"] > 0 and chamfer["ci95"][0] > 0.0)
        or (fragmentation["count"] > 0 and fragmentation["ci95"][0] > 0.0)
    )
    return {
        "paired_frame_count": len(common),
        "query_f1": f1,
        "edge_3d_chamfer_m": chamfer,
        "contour_fragmentation": fragmentation,
        "hypothesis_supported": bool(f1_supported and geometry_supported),
    }


def evaluate_checkpoints(config: EdgeEstimateConfig, checkpoints: dict[Variant, Path]) -> Path:
    """NYU test cache에서 모델별 edge/depth/3D/BEV 지표와 paired bootstrap을 저장한다."""
    ensure_edge_output_roots(config)
    device = _device()
    run_dir = config.output_root / "eval" / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    per_frame: dict[str, dict[int, dict[str, float]]] = {}
    for variant, checkpoint in checkpoints.items():
        dataset = EdgePatchDataset(config, "test", variant)
        model = _model_from_checkpoint(config, variant, checkpoint, device)
        metrics, frame_metrics = evaluate_model(model, dataset, config, device)
        all_metrics[variant] = metrics
        per_frame[variant] = frame_metrics
    if "rgb_local" in per_frame and "rgb_context" in per_frame:
        all_metrics["context_hypothesis_bootstrap"] = _paired_bootstrap(
            per_frame["rgb_local"], per_frame["rgb_context"], config.train.seed
        )
    (run_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "bev_xz_metric_semantics": "NYU에는 world pose가 없으므로 camera XZ 평면에서 계산한 진단 지표",
                "core_hypothesis": (
                    "rgb_context가 rgb_local보다 F1을 높이고 3D Chamfer 또는 fragmentation을 "
                    "낮추는지 paired bootstrap으로 평가"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    save_edge_config(config, run_dir / "config.json")
    return run_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_base_bev_run(base_bev_run: Path, config: EdgeEstimateConfig) -> None:
    """기존 BEV를 수정하지 않고 복사·융합하기 전에 좌표 범위를 확인한다."""

    config_path = base_bev_run / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"base BEV config를 찾지 못했습니다: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    bev = payload.get("bev") or payload.get("config", {}).get("bev")
    if not isinstance(bev, dict):
        raise ValueError("base BEV config에 bev 설정이 없습니다.")
    expected = config.base.bev
    actual_center = tuple(float(value) for value in bev.get("center_xy", ()))
    compatible = len(actual_center) == 2 and np.allclose(actual_center, expected.center_xy, atol=1.0e-8)
    compatible &= math.isclose(float(bev.get("size_m", float("nan"))), expected.size_m, abs_tol=1.0e-8)
    compatible &= math.isclose(
        float(bev.get("meters_per_pixel", float("nan"))), expected.meters_per_pixel, abs_tol=1.0e-8
    )
    if not compatible:
        raise ValueError(
            "base BEV의 center_xy/size_m/meters_per_pixel이 edge experiment BEV와 다릅니다."
        )
    base_path = base_bev_run / "edge_confident" / "bev_rgb.png"
    if not base_path.exists():
        base_path = base_bev_run / "bev_rgb.png"
    if not base_path.exists():
        raise FileNotFoundError(f"base BEV RGB를 찾지 못했습니다: {base_bev_run}")
    with Image.open(base_path) as image:
        if image.size != (expected.resolution, expected.resolution):
            raise ValueError(
                f"base BEV resolution {image.size} != expected {(expected.resolution, expected.resolution)}"
            )


def validate_environment(config: EdgeEstimateConfig, input_rgb: Path | None = None, evaluation_depth: Path | None = None) -> dict:
    """파일, CUDA, fisheye round-trip, 평가 depth shape를 변경 없이 검사한다."""
    input_rgb = input_rgb or config.base.paths.input_rgb
    evaluation_depth = evaluation_depth or config.base.paths.external_depth_z
    rays = build_camera_rays(config.base.camera)
    result = {
        "device": str(_device()),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "input_rgb_exists": input_rgb.exists(),
        "input_rgb_sha256": _sha256(input_rgb) if input_rgb.exists() else None,
        "evaluation_depth_exists": evaluation_depth.exists(),
        "nyu_mat_exists": config.base.paths.nyu_mat.exists(),
        "nyu_train_split_exists": config.base.paths.nyu_split_train.exists(),
        "nyu_test_split_exists": config.base.paths.nyu_split_test.exists(),
        "da_v2_root_exists": config.base.paths.depth_anything_root.exists(),
        "da_v2_checkpoint_exists": config.base.paths.depth_anything_vitl_ckpt.exists(),
        "fisheye_valid_pixels": int(rays.valid.sum()),
        "fisheye_roundtrip_max_error_px": rays.max_roundtrip_error_px,
        "cache_root": str(config.cache_root),
    }
    if evaluation_depth.exists():
        depth = np.load(evaluation_depth, mmap_mode="r")
        result["evaluation_depth_shape"] = list(depth.shape)
        result["evaluation_depth_shape_matches"] = tuple(depth.shape) == (
            config.base.camera.height,
            config.base.camera.width,
        )
    return result


def _support_batch(
    rgb: np.ndarray,
    rays: np.ndarray,
    valid: np.ndarray,
    origins: list[tuple[int, int]],
    da_relative: np.ndarray | None,
    edge_prior_2d: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_rgb, support_rays, support_valid, da_values, da_valid, support_edge = [], [], [], [], [], []
    for y, x in origins:
        support_rgb.append(rgb[y : y + 5, x : x + 5].astype(np.float32) / 255.0)
        support_rays.append(rays[y : y + 5, x : x + 5])
        support_valid.append(valid[y : y + 5, x : x + 5])
        if edge_prior_2d is None:
            support_edge.append(np.zeros((5, 5), dtype=np.float32))
        else:
            support_edge.append(edge_prior_2d[y : y + 5, x : x + 5].astype(np.float32))
        if da_relative is None:
            da_values.append(np.zeros((5, 5), dtype=np.float32))
            da_valid.append(np.zeros((5, 5), dtype=bool))
        else:
            values = da_relative[y : y + 5, x : x + 5]
            da_values.append(np.nan_to_num(values).astype(np.float32))
            da_valid.append(np.isfinite(values) & valid[y : y + 5, x : x + 5])
    return tuple(np.stack(items) for items in (support_rgb, support_rays, support_valid, da_values, da_valid, support_edge))


@torch.inference_mode()
def _coarse_scan(
    model: EdgeEstimateModel,
    rgb: np.ndarray,
    rays: np.ndarray,
    valid: np.ndarray,
    da_relative: np.ndarray | None,
    edge_prior_2d: np.ndarray | None,
    config: EdgeEstimateConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = valid.shape
    ys = list(range(0, h - 4, 4))
    xs = list(range(0, w - 4, 4))
    if ys[-1] != h - 5:
        ys.append(h - 5)
    if xs[-1] != w - 5:
        xs.append(w - 5)
    origins = [(y, x) for y in ys for x in xs]
    probability_sum = np.zeros((h - 1, w - 1), dtype=np.float32)
    type_sum = np.zeros((h - 1, w - 1, 3), dtype=np.float32)
    count = np.zeros((h - 1, w - 1), dtype=np.float32)
    for start in tqdm(
        range(0, len(origins), config.inference.coarse_batch_size),
        desc=f"coarse {model.variant}",
        disable=not sys.stderr.isatty(),
    ):
        batch_origins = origins[start : start + config.inference.coarse_batch_size]
        arrays = _support_batch(rgb, rays, valid, batch_origins, da_relative, edge_prior_2d)
        tensors = [torch.from_numpy(value).to(device) for value in arrays]
        _, edge_logit, type_logits = model.encode_cells(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5])
        edge = torch.sigmoid(edge_logit).cpu().numpy()
        types = torch.softmax(type_logits, dim=-1).cpu().numpy()
        for index, (y, x) in enumerate(batch_origins):
            cell_valid = (
                arrays[2][index, :-1, :-1]
                & arrays[2][index, :-1, 1:]
                & arrays[2][index, 1:, 1:]
                & arrays[2][index, 1:, :-1]
            )
            probability_sum[y : y + 4, x : x + 4] += edge[index] * cell_valid
            type_sum[y : y + 4, x : x + 4] += types[index] * cell_valid[..., None]
            count[y : y + 4, x : x + 4] += cell_valid
    probability = np.divide(probability_sum, count, out=np.zeros_like(probability_sum), where=count > 0)
    type_probability = np.divide(type_sum, count[..., None], out=np.zeros_like(type_sum), where=count[..., None] > 0)
    probability[count == 0] = np.nan
    return probability, type_probability


def _candidate_cells(
    probability: np.ndarray,
    rays: np.ndarray,
    valid: np.ndarray,
    config: EdgeEstimateConfig,
    edge_prior_2d: np.ndarray | None = None,
) -> np.ndarray:
    score_map = np.nan_to_num(probability)
    if edge_prior_2d is not None and config.edge_prior.enabled:
        score_map = score_map + float(config.edge_prior.candidate_weight) * cell_edge_prior(edge_prior_2d)
    mask = np.isfinite(probability) & (score_map >= config.inference.coarse_threshold)
    if config.inference.candidate_dilation_cells > 0:
        size = 2 * config.inference.candidate_dilation_cells + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    mask &= cell_valid
    y, x = np.nonzero(mask)
    if len(x) > config.inference.max_candidate_cells:
        angular_gap, _ = cell_angular_gap(rays, valid)
        score = np.nan_to_num(score_map[y, x]) * np.nan_to_num(angular_gap[y, x])
        keep = np.argsort(-score, kind="stable")[: config.inference.max_candidate_cells]
        y, x = y[keep], x[keep]
    return np.stack([y, x], axis=-1).astype(np.int32)


def _query_rays_for_cells(
    rays: np.ndarray,
    cells_yx: np.ndarray,
    relative: np.ndarray,
    config: EdgeEstimateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    query_rays = np.empty((len(cells_yx), len(relative), 3), dtype=np.float32)
    source_uv = np.empty((len(cells_yx), len(relative), 2), dtype=np.float32)
    for index, (y, x) in enumerate(cells_yx):
        corners = np.stack([rays[y, x], rays[y, x + 1], rays[y + 1, x + 1], rays[y + 1, x]])
        query_rays[index] = spherical_bilerp(corners[None], relative[None])[0]
        projected, _ = project_camera_rays(query_rays[index], config.base.camera)
        source_uv[index] = projected.astype(np.float32)
    return query_rays, source_uv


def _subpixel_edge_nms(probability: np.ndarray) -> np.ndarray:
    """8×8 query 확률 ridge를 gradient 수직 방향으로 얇게 만드는 non-maximum suppression.

    입력은 `(B,Q)`이며 `Q`는 정사각 query grid다. Sobel gradient 방향을 0/45/90/135도로
    양자화하고 양쪽 이웃보다 작은 점만 제거한다. 평평한 plateau는 contour 단절을 피하기
    위해 유지하며, 이후 confidence-ordered deduplication이 공유 cell 변의 중복을 제거한다.
    """

    values = np.asarray(probability, dtype=np.float32)
    side = int(round(math.sqrt(values.shape[-1])))
    if side * side != values.shape[-1]:
        raise ValueError("subpixel NMS의 query 수는 정사각 grid여야 합니다.")
    maps = values.reshape(-1, side, side)
    keep = np.ones_like(maps, dtype=bool)
    for index, edge_map in enumerate(maps):
        grad_x = cv2.Sobel(edge_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edge_map, cv2.CV_32F, 0, 1, ksize=3)
        angle = (np.rad2deg(np.arctan2(grad_y, grad_x)) + 180.0) % 180.0
        padded = np.pad(edge_map, 1, mode="constant", constant_values=-np.inf)
        center = padded[1:-1, 1:-1]
        comparisons = (
            ((angle < 22.5) | (angle >= 157.5), padded[1:-1, :-2], padded[1:-1, 2:]),
            ((angle >= 22.5) & (angle < 67.5), padded[:-2, 2:], padded[2:, :-2]),
            ((angle >= 67.5) & (angle < 112.5), padded[:-2, 1:-1], padded[2:, 1:-1]),
            ((angle >= 112.5) & (angle < 157.5), padded[:-2, :-2], padded[2:, 2:]),
        )
        current = np.zeros_like(edge_map, dtype=bool)
        for direction, first, second in comparisons:
            current |= direction & (center >= first) & (center >= second)
        keep[index] = current
    return keep.reshape(values.shape)


@torch.inference_mode()
def _refine_candidates(
    model: EdgeEstimateModel,
    candidates: np.ndarray,
    coarse_type: np.ndarray,
    rgb: np.ndarray,
    rays: np.ndarray,
    valid: np.ndarray,
    da_relative: np.ndarray | None,
    edge_prior_2d: np.ndarray | None,
    prior_depth_z: np.ndarray | None,
    prior_depth_valid: np.ndarray | None,
    config: EdgeEstimateConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    relative = _query_grid(config.data.query_grid_size)
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    h, w = valid.shape
    for start in tqdm(
        range(0, len(candidates), config.inference.query_batch_size),
        desc=f"refine {model.variant}",
        disable=not sys.stderr.isatty(),
    ):
        cells = candidates[start : start + config.inference.query_batch_size]
        origins: list[tuple[int, int]] = []
        local_indices: list[tuple[int, int]] = []
        for y, x in cells:
            origin_y = int(np.clip(y - 1, 0, h - 5))
            origin_x = int(np.clip(x - 1, 0, w - 5))
            origins.append((origin_y, origin_x))
            local_indices.append((int(y - origin_y), int(x - origin_x)))
        arrays = _support_batch(rgb, rays, valid, origins, da_relative, edge_prior_2d)
        tensors = [torch.from_numpy(value).to(device) for value in arrays]
        cell_features, _, _ = model.encode_cells(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5])
        selected_features = torch.stack(
            [cell_features[index, :, local_y, local_x] for index, (local_y, local_x) in enumerate(local_indices)]
        )
        query_rays, source_uv = _query_rays_for_cells(rays, cells, relative, config)
        if prior_depth_z is None or prior_depth_valid is None:
            query_prior_depth = np.zeros(source_uv.shape[:-1], dtype=np.float32)
            query_prior_valid = np.zeros(source_uv.shape[:-1], dtype=bool)
        else:
            query_prior_depth, prior_weight = _sample_bilinear_masked(prior_depth_z, prior_depth_valid, source_uv)
            query_prior_valid = (prior_weight > 1.0e-6) & np.isfinite(query_prior_depth) & (query_prior_depth > 0.0)
            query_prior_depth = np.nan_to_num(query_prior_depth, nan=0.0).astype(np.float32)
        query_ray_t = torch.from_numpy(query_rays).to(device)
        relative_t = torch.from_numpy(np.broadcast_to(relative, (len(cells), *relative.shape)).copy()).to(device)
        prior_depth_t = torch.from_numpy(query_prior_depth).to(device)
        prior_valid_t = torch.from_numpy(query_prior_valid).to(device)
        edge_logit, near, far, confidence_logit, delta_log_depth = model.decode_selected_queries(
            selected_features, query_ray_t, relative_t, prior_depth_t, prior_valid_t
        )
        edge_probability = torch.sigmoid(edge_logit).cpu().numpy()
        nms_keep = _subpixel_edge_nms(edge_probability)
        pieces["source_uv"].append(source_uv.reshape(-1, 2))
        pieces["relative_uv"].append(np.broadcast_to(relative, (len(cells), *relative.shape)).reshape(-1, 2).copy())
        pieces["ray_dir"].append(query_rays.reshape(-1, 3))
        pieces["edge_probability"].append(edge_probability.reshape(-1))
        pieces["nms_keep"].append(nms_keep.reshape(-1))
        pieces["confidence"].append(torch.sigmoid(confidence_logit).cpu().numpy().reshape(-1))
        pieces["delta_log_depth"].append(delta_log_depth.cpu().numpy().reshape(-1))
        pieces["prior_depth_z"].append(query_prior_depth.reshape(-1))
        pieces["prior_depth_valid"].append(query_prior_valid.reshape(-1))
        pieces["depth_near_z"].append(near.cpu().numpy().reshape(-1))
        pieces["depth_far_z"].append(far.cpu().numpy().reshape(-1))
        type_value = np.argmax(coarse_type[cells[:, 0], cells[:, 1]], axis=-1) + 1
        pieces["edge_type"].append(np.repeat(type_value, len(relative)).astype(np.uint8))
        pieces["parent_cell"].append(np.repeat(cells[:, None, ::-1], len(relative), axis=1).reshape(-1, 2))
    if not pieces:
        return {
            "source_uv": np.empty((0, 2), np.float32),
            "relative_uv": np.empty((0, 2), np.float32),
            "ray_dir": np.empty((0, 3), np.float32),
            "edge_probability": np.empty(0, np.float32),
            "nms_keep": np.empty(0, bool),
            "confidence": np.empty(0, np.float32),
            "delta_log_depth": np.empty(0, np.float32),
            "prior_depth_z": np.empty(0, np.float32),
            "prior_depth_valid": np.empty(0, bool),
            "depth_near_z": np.empty(0, np.float32),
            "depth_far_z": np.empty(0, np.float32),
            "edge_type": np.empty(0, np.uint8),
            "parent_cell": np.empty((0, 2), np.int32),
        }
    result = {key: np.concatenate(value) for key, value in pieces.items()}
    priority = result["nms_keep"].astype(np.float32) * 2.0
    priority += result["edge_probability"] * result["confidence"]
    order = np.argsort(-priority, kind="stable")
    dedupe_key = np.concatenate(
        [
            np.round(result["source_uv"][order], config.inference.dedupe_uv_decimals),
            np.round(result["ray_dir"][order], config.inference.dedupe_uv_decimals + 3),
        ],
        axis=-1,
    )
    _, unique_index = np.unique(dedupe_key, axis=0, return_index=True)
    keep = order[np.sort(unique_index)]
    return {key: value[keep] for key, value in result.items()}


def _raster_queries(
    shape: tuple[int, int],
    source_uv: np.ndarray,
    value: np.ndarray,
    confidence: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    output = np.full(shape, np.nan, dtype=np.float32)
    best = np.full(shape, -np.inf, dtype=np.float32)
    xy = np.rint(source_uv - 0.5).astype(np.int64)
    inside = selected & (xy[:, 0] >= 0) & (xy[:, 0] < shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < shape[0])
    for index in np.flatnonzero(inside):
        x, y = xy[index]
        if confidence[index] > best[y, x]:
            best[y, x] = confidence[index]
            output[y, x] = value[index]
    return output


def _ordered_components(source_uv: np.ndarray, selected: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    points = source_uv[selected]
    if len(points) == 0:
        return np.empty(0, np.int32), np.array([0], np.int64)
    neighbors = cKDTree(points).query_ball_point(points, r=radius)
    labels = np.full(len(points), -1, dtype=np.int32)
    component = 0
    for start in range(len(points)):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    stack.append(neighbor)
        component += 1
    ordered_indices: list[np.ndarray] = []
    offsets = [0]
    selected_indices = np.flatnonzero(selected)
    for component_id in range(component):
        local = np.flatnonzero(labels == component_id)
        centered = points[local] - points[local].mean(axis=0, keepdims=True)
        if len(local) > 1:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            local = local[np.argsort(centered @ vh[0])]
        ordered_indices.append(selected_indices[local])
        offsets.append(offsets[-1] + len(local))
    return np.concatenate(ordered_indices).astype(np.int32), np.asarray(offsets, dtype=np.int64)


def _bev_raster(
    world_points: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    config: EdgeEstimateConfig,
    order: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    *,
    dilate_output: bool = True,
) -> np.ndarray:
    bev = config.base.bev
    output = np.zeros((bev.resolution, bev.resolution), dtype=np.float32)
    half = bev.size_m * 0.5
    min_x, min_y = bev.center_xy[0] - half, bev.center_xy[1] - half
    inside = np.asarray(valid, dtype=bool) & np.isfinite(world_points).all(axis=-1)
    inside &= (world_points[:, 0] >= min_x) & (world_points[:, 0] < min_x + bev.size_m)
    inside &= (world_points[:, 1] >= min_y) & (world_points[:, 1] < min_y + bev.size_m)
    columns = np.full(len(world_points), -1, dtype=np.int64)
    rows = np.full(len(world_points), -1, dtype=np.int64)
    columns[inside] = np.floor((world_points[inside, 0] - min_x) / bev.meters_per_pixel).astype(np.int64)
    rows[inside] = bev.resolution - 1 - np.floor(
        (world_points[inside, 1] - min_y) / bev.meters_per_pixel
    ).astype(np.int64)
    if order is not None and offsets is not None:
        segments: list[tuple[float, int, int]] = []
        max_segment_m = max(4.0 * bev.meters_per_pixel, 0.25)
        for component_index in range(len(offsets) - 1):
            component = order[offsets[component_index] : offsets[component_index + 1]]
            for first, second in zip(component[:-1], component[1:]):
                if inside[first] and inside[second]:
                    distance = float(np.linalg.norm(world_points[first, :2] - world_points[second, :2]))
                    if distance <= max_segment_m:
                        segments.append((float(min(confidence[first], confidence[second])), int(first), int(second)))
        # 낮은 confidence 선분부터 그려 교차 시 높은 confidence가 최종값으로 남게 한다.
        for value, first, second in sorted(segments):
            cv2.line(
                output,
                (int(columns[first]), int(rows[first])),
                (int(columns[second]), int(rows[second])),
                color=value,
                thickness=1,
                lineType=cv2.LINE_8,
            )
    indices = np.flatnonzero(inside)
    if len(indices):
        np.maximum.at(
            output.reshape(-1),
            rows[indices] * bev.resolution + columns[indices],
            confidence[indices],
        )
    # 한 pixel 선분이 downsampling에서 사라지지 않도록 최종 선 폭만 1 cell 확장한다.
    if not dilate_output:
        return output
    return cv2.dilate(output, np.ones((3, 3), np.uint8))


def _remove_small_bev_components(mask: np.ndarray, min_pixels: int = 6) -> np.ndarray:
    """BEV edge 후처리에서 고립된 작은 점 성분을 제거한다."""

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            keep |= labels == label
    return keep


def _bev_edge_map_layers(raw_line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse BEV edge point/line을 지도 레이어에 가깝게 정리한다.

    ``raw_line``은 3D edge query point와 가까운 contour 선분을 한 픽셀 폭으로 찍은
    confidence map이다. 여기에 작은 gap closing과 두께 부여를 적용해 두 산출물을 만든다.

    - polyline: 작은 틈을 메운 선형 confidence layer
    - occupancy: polyline을 약간 두껍게 만든 edge occupancy layer
    """

    raw = np.nan_to_num(raw_line.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    seed = raw > 0.0
    if not np.any(seed):
        return np.zeros_like(raw), np.zeros_like(raw)
    close_kernel = np.ones((5, 5), np.uint8)
    occupancy_kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
    closed = _remove_small_bev_components(closed)
    confidence = cv2.dilate(raw, close_kernel)
    polyline = np.where(closed, confidence, 0.0).astype(np.float32)
    occupancy_mask = cv2.dilate(closed.astype(np.uint8), occupancy_kernel) > 0
    occupancy = np.where(occupancy_mask, cv2.dilate(polyline, occupancy_kernel), 0.0).astype(np.float32)
    return np.clip(polyline, 0.0, 1.0), np.clip(occupancy, 0.0, 1.0)


def _blend_bev_layer(base: np.ndarray, layer: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """기존 BEV RGB 위에 confidence layer를 alpha blend한다."""

    output = base.copy()
    mask = layer > 0.0
    if np.any(mask):
        alpha = (0.25 + 0.55 * np.clip(layer[mask], 0.0, 1.0))[..., None]
        tint = np.asarray(color, dtype=np.float32)
        output[mask] = ((1.0 - alpha) * output[mask].astype(np.float32) + alpha * tint).clip(0, 255).astype(np.uint8)
    return output


def _overlay(rgb: np.ndarray, source_uv: np.ndarray, selected: np.ndarray, edge_type: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    colors = {1: (40, 220, 255), 2: (255, 80, 50), 3: (220, 80, 255)}
    xy = np.rint(source_uv - 0.5).astype(np.int64)
    for index in np.flatnonzero(selected):
        x, y = xy[index]
        if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
            output[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = colors.get(int(edge_type[index]), (255, 255, 0))
    return output


def _save_3d_preview(
    path: Path,
    camera_points: np.ndarray,
    world_points: np.ndarray,
    edge_type: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
) -> None:
    """완성된 near/crease edge의 camera XZ와 world XY 분포를 한 이미지로 저장한다."""

    indices = np.flatnonzero(valid & np.isfinite(camera_points).all(axis=-1) & np.isfinite(world_points).all(axis=-1))
    if len(indices) > 50_000:
        indices = indices[np.linspace(0, len(indices) - 1, 50_000, dtype=np.int64)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    if len(indices):
        palette = np.array([[0.1, 0.8, 1.0], [1.0, 0.25, 0.1], [0.9, 0.2, 1.0]], dtype=np.float32)
        colors = palette[np.clip(edge_type[indices].astype(np.int64) - 1, 0, 2)]
        alpha = np.clip(confidence[indices], 0.15, 1.0)
        colors = np.concatenate([colors, alpha[:, None]], axis=-1)
        axes[0].scatter(camera_points[indices, 0], camera_points[indices, 2], s=1.0, c=colors, linewidths=0)
        axes[1].scatter(world_points[indices, 0], world_points[indices, 1], s=1.0, c=colors, linewidths=0)
    axes[0].set(title="Camera-frame edge (X-Z)", xlabel="X [m]", ylabel="Z [m]")
    axes[1].set(title="World-frame edge (X-Y, BEV)", xlabel="World X [m]", ylabel="World Y [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_variant_result(
    run_dir: Path,
    variant: str,
    rgb: np.ndarray,
    coarse: np.ndarray,
    queries: dict[str, np.ndarray],
    selected: np.ndarray,
    rays: np.ndarray,
    edge_prior_2d: np.ndarray | None,
    config: EdgeEstimateConfig,
    evaluation_depth: np.ndarray | None,
    base_bev_run: Path | None,
) -> dict:
    target = run_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    save_rgb(target / "source_rgb.png", rgb)
    if edge_prior_2d is not None:
        np.save(target / "edge_2d_prior.npy", edge_prior_2d.astype(np.float32))
        save_heatmap(target / "edge_2d_prior.png", edge_prior_2d, value_min=0.0, value_max=1.0)
    save_heatmap(target / "coarse_edge_probability.png", coarse, value_min=0.0, value_max=1.0)
    unknown = ~selected
    queries_to_save = {**queries, "completed": selected.copy(), "unknown": unknown}
    np.savez_compressed(target / "edge_queries.npz", **queries_to_save)
    probability_map = _raster_queries(rgb.shape[:2], queries["source_uv"], queries["edge_probability"], queries["confidence"], selected)
    confidence_map = _raster_queries(rgb.shape[:2], queries["source_uv"], queries["confidence"], queries["confidence"], selected)
    near_map = _raster_queries(rgb.shape[:2], queries["source_uv"], queries["depth_near_z"], queries["confidence"], selected)
    far_selected = selected & (queries["edge_type"] == EDGE_OCCLUSION)
    far_map = _raster_queries(rgb.shape[:2], queries["source_uv"], queries["depth_far_z"], queries["confidence"], far_selected)
    type_map = _raster_queries(rgb.shape[:2], queries["source_uv"], queries["edge_type"].astype(np.float32), queries["confidence"], selected)
    np.save(target / "edge_probability.npy", probability_map)
    np.save(target / "edge_confidence.npy", confidence_map)
    np.save(target / "edge_depth_near_z.npy", near_map)
    np.save(target / "edge_depth_far_z.npy", far_map)
    save_heatmap(target / "edge_probability.png", probability_map, value_min=0.0, value_max=1.0)
    save_heatmap(target / "edge_confidence.png", confidence_map, value_min=0.0, value_max=1.0)
    save_depth(target / "edge_depth_near_z.png", near_map)
    save_depth(target / "edge_depth_far_z.png", far_map)
    save_heatmap(target / "edge_type.png", type_map, cmap_name="tab10", value_min=0.0, value_max=3.0)
    save_rgb(target / "edge_overlay.png", _overlay(rgb, queries["source_uv"], selected, queries["edge_type"]))

    near_camera, _, near_valid = points_from_z_depth(
        queries["depth_near_z"], queries["ray_dir"], z_eps=config.base.camera.geometry_z_eps
    )
    near_valid &= selected
    near_world = camera_to_world_points(near_camera, config.base.camera).astype(np.float32)
    far_camera, _, far_valid = points_from_z_depth(
        queries["depth_far_z"], queries["ray_dir"], z_eps=config.base.camera.geometry_z_eps
    )
    far_valid &= far_selected
    far_world = camera_to_world_points(far_camera, config.base.camera).astype(np.float32)
    order, offsets = _ordered_components(queries["source_uv"], selected, config.inference.contour_radius_px)
    np.savez_compressed(
        target / "edge_polylines_camera.npz",
        points=near_camera[order],
        source_uv=queries["source_uv"][order],
        confidence=queries["confidence"][order],
        edge_type=queries["edge_type"][order],
        offsets=offsets,
    )
    np.savez_compressed(
        target / "edge_polylines_world.npz",
        points=near_world[order],
        source_uv=queries["source_uv"][order],
        confidence=queries["confidence"][order],
        edge_type=queries["edge_type"][order],
        offsets=offsets,
    )
    _save_3d_preview(
        target / "edge_3d_preview.png",
        near_camera,
        near_world,
        queries["edge_type"],
        queries["confidence"],
        near_valid,
    )
    far_order, far_offsets = _ordered_components(
        queries["source_uv"], far_valid, config.inference.contour_radius_px
    )
    near_bev_raw = _bev_raster(
        near_world, queries["confidence"], near_valid, config, order, offsets, dilate_output=False
    )
    near_bev = cv2.dilate(near_bev_raw, np.ones((3, 3), np.uint8))
    near_bev_polyline, near_bev_occupancy = _bev_edge_map_layers(near_bev_raw)
    far_bev = _bev_raster(far_world, queries["confidence"], far_valid, config, far_order, far_offsets)
    edge_root = target / "edge_only"
    edge_root.mkdir(parents=True, exist_ok=True)
    np.save(edge_root / "bev_edge_probability.npy", near_bev)
    np.save(edge_root / "bev_edge_polyline.npy", near_bev_polyline)
    np.save(edge_root / "bev_edge_occupancy.npy", near_bev_occupancy)
    save_heatmap(edge_root / "bev_edge_probability.png", np.where(near_bev > 0, near_bev, np.nan), value_min=0.0, value_max=1.0)
    save_heatmap(edge_root / "bev_edge_polyline.png", np.where(near_bev_polyline > 0, near_bev_polyline, np.nan), value_min=0.0, value_max=1.0)
    save_heatmap(edge_root / "bev_edge_occupancy.png", np.where(near_bev_occupancy > 0, near_bev_occupancy, np.nan), value_min=0.0, value_max=1.0)
    save_heatmap(edge_root / "bev_edge_near.png", np.where(near_bev > 0, near_bev, np.nan), cmap_name="turbo", value_min=0.0, value_max=1.0)
    save_heatmap(edge_root / "bev_edge_far.png", np.where(far_bev > 0, far_bev, np.nan), cmap_name="cool", value_min=0.0, value_max=1.0)

    fused_exists = False
    if base_bev_run is not None:
        base_path = base_bev_run / "edge_confident" / "bev_rgb.png"
        if not base_path.exists():
            base_path = base_bev_run / "bev_rgb.png"
        if base_path.exists():
            base = np.asarray(Image.open(base_path).convert("RGB"))
            if base.shape[:2] != near_bev.shape:
                raise ValueError(f"Base BEV shape {base.shape[:2]} != edge BEV shape {near_bev.shape}")
            fused = base.copy()
            mask = near_bev > 0.0
            fused[mask] = (0.35 * fused[mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
            fused_root = target / "fused"
            fused_root.mkdir(parents=True, exist_ok=True)
            save_rgb(fused_root / "bev_rgb_with_edges.png", fused)
            save_rgb(
                fused_root / "bev_rgb_with_edge_polyline.png",
                _blend_bev_layer(base, near_bev_polyline, (255, 40, 40)),
            )
            save_rgb(
                fused_root / "bev_rgb_with_edge_occupancy.png",
                _blend_bev_layer(base, near_bev_occupancy, (255, 120, 40)),
            )
            overlay_rgb = np.zeros_like(base)
            overlay_rgb[mask] = np.array([255, 40, 40], dtype=np.uint8)
            save_rgb(fused_root / "bev_edge_overlay.png", overlay_rgb)
            fused_exists = True

    metrics = {
        "candidate_query_count": int(len(selected)),
        "completed_edge_count": int(selected.sum()),
        "unknown_count": int(unknown.sum()),
        "crease_count": int(np.sum(selected & (queries["edge_type"] == 1))),
        "occlusion_count": int(np.sum(selected & (queries["edge_type"] == 2))),
        "junction_count": int(np.sum(selected & (queries["edge_type"] == 3))),
        "bev_edge_cells": int(np.sum(near_bev > 0.0)),
        "bev_edge_polyline_cells": int(np.sum(near_bev_polyline > 0.0)),
        "bev_edge_occupancy_cells": int(np.sum(near_bev_occupancy > 0.0)),
        "fused_bev_created": fused_exists,
    }
    if evaluation_depth is not None:
        gt_valid = np.isfinite(evaluation_depth) & (evaluation_depth > 0.0) & (rays[..., 2] > config.base.camera.geometry_z_eps)
        labels = build_pseudo_edges(evaluation_depth, gt_valid, rays, config.data)
        gt_edge = _sample_bilinear(labels.edge_soft, queries["source_uv"])
        gt_near, gt_near_weight = _sample_bilinear_masked(
            labels.near_depth_z, np.isfinite(labels.near_depth_z), queries["source_uv"]
        )
        gt_ignore = _sample_nearest(labels.ignore.astype(np.uint8), queries["source_uv"]).astype(bool)
        eval_mask = ~gt_ignore & np.isfinite(gt_edge)
        gt_query_valid = eval_mask & (gt_edge >= 0.5) & (gt_near_weight > 1.0e-6) & np.isfinite(gt_near)
        metrics.update({f"gt_edge_{key}": value for key, value in _binary_metrics(queries["edge_probability"], gt_edge, eval_mask).items()})
        metrics.update(
            {
                f"gt_near_depth_{key}": value
                for key, value in _depth_metrics(
                    queries["depth_near_z"], gt_near, gt_query_valid & selected
                ).items()
            }
        )
        gt_query_camera, _, gt_query_point_valid = points_from_z_depth(
            gt_near, queries["ray_dir"], z_eps=config.base.camera.geometry_z_eps
        )
        gt_query_point_valid &= gt_query_valid
        point_metrics = _point_set_metrics(
            near_camera[near_valid], gt_query_camera[gt_query_point_valid], threshold_m=0.05
        )
        metrics.update({f"gt_edge_3d_{key}": value for key, value in point_metrics.items()})

        gt_project_depth, gt_project_weight = _sample_bilinear_masked(
            evaluation_depth, gt_valid, queries["source_uv"]
        )
        gt_project_camera, _, gt_project_point_valid = points_from_z_depth(
            gt_project_depth, queries["ray_dir"], z_eps=config.base.camera.geometry_z_eps
        )
        gt_project_valid = selected & (gt_project_weight > 1.0e-6) & gt_project_point_valid
        gt_project_world = camera_to_world_points(gt_project_camera, config.base.camera).astype(np.float32)
        gt_project_bev_raw = _bev_raster(
            gt_project_world,
            queries["confidence"],
            gt_project_valid,
            config,
            order,
            offsets,
            dilate_output=False,
        )
        gt_project_bev = cv2.dilate(gt_project_bev_raw, np.ones((3, 3), np.uint8))
        gt_project_polyline, gt_project_occupancy = _bev_edge_map_layers(gt_project_bev_raw)
        np.save(edge_root / "bev_edge_projected_with_gt_depth.npy", gt_project_bev)
        np.save(edge_root / "bev_edge_projected_with_gt_depth_polyline.npy", gt_project_polyline)
        np.save(edge_root / "bev_edge_projected_with_gt_depth_occupancy.npy", gt_project_occupancy)
        save_heatmap(
            edge_root / "bev_edge_projected_with_gt_depth.png",
            np.where(gt_project_bev > 0, gt_project_bev, np.nan),
            value_min=0.0,
            value_max=1.0,
        )
        save_heatmap(
            edge_root / "bev_edge_projected_with_gt_depth_polyline.png",
            np.where(gt_project_polyline > 0, gt_project_polyline, np.nan),
            value_min=0.0,
            value_max=1.0,
        )
        save_heatmap(
            edge_root / "bev_edge_projected_with_gt_depth_occupancy.png",
            np.where(gt_project_occupancy > 0, gt_project_occupancy, np.nan),
            value_min=0.0,
            value_max=1.0,
        )
        metrics["gt_depth_projected_edge_cells"] = int(np.sum(gt_project_bev > 0.0))
        metrics["gt_depth_projected_edge_polyline_cells"] = int(np.sum(gt_project_polyline > 0.0))
        metrics["gt_depth_projected_edge_occupancy_cells"] = int(np.sum(gt_project_occupancy > 0.0))

        gt_overlay = _overlay(rgb, queries["source_uv"], gt_edge >= 0.5, _sample_nearest(labels.edge_type, queries["source_uv"]))
        gt_root = target / "gt"
        gt_root.mkdir(parents=True, exist_ok=True)
        save_rgb(gt_root / "edge_gt_overlay.png", gt_overlay)
        yy, xx = np.nonzero(labels.edge)
        gt_rays = rays[yy, xx]
        gt_depth = labels.near_depth_z[yy, xx]
        gt_camera, _, gt_point_valid = points_from_z_depth(gt_depth, gt_rays, z_eps=config.base.camera.geometry_z_eps)
        gt_world = camera_to_world_points(gt_camera, config.base.camera)
        gt_bev = _bev_raster(gt_world, np.ones(len(gt_world), np.float32), gt_point_valid, config)
        save_heatmap(gt_root / "bev_edge_gt.png", np.where(gt_bev > 0, gt_bev, np.nan), value_min=0.0, value_max=1.0)
        pred_bev_mask = near_bev > 0.0
        gt_bev_mask = gt_bev > 0.0
        tp = int(np.sum(pred_bev_mask & gt_bev_mask))
        fp = int(np.sum(pred_bev_mask & ~gt_bev_mask))
        fn = int(np.sum(~pred_bev_mask & gt_bev_mask))
        bev_precision = tp / max(tp + fp, 1)
        bev_recall = tp / max(tp + fn, 1)
        metrics["gt_bev_edge_precision"] = float(bev_precision)
        metrics["gt_bev_edge_recall"] = float(bev_recall)
        metrics["gt_bev_edge_f1"] = float(
            2.0 * bev_precision * bev_recall / max(bev_precision + bev_recall, 1.0e-12)
        )
        pred_bev_yx = np.argwhere(pred_bev_mask).astype(np.float32) * config.base.bev.meters_per_pixel
        gt_bev_yx = np.argwhere(gt_bev_mask).astype(np.float32) * config.base.bev.meters_per_pixel
        metrics["gt_bev_edge_chamfer_m"] = _point_set_metrics(
            pred_bev_yx, gt_bev_yx, threshold_m=config.base.bev.meters_per_pixel
        )["chamfer_m"]
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def _relative_da(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(depth.shape, np.nan, dtype=np.float32)
    good = valid & np.isfinite(depth) & (depth > 0.0)
    if np.any(good):
        values = np.log(depth[good])
        result[good] = values - float(np.median(values))
    return result


def run_inference(
    config: EdgeEstimateConfig,
    checkpoints: dict[Variant, Path],
    *,
    input_rgb: Path | None = None,
    evaluation_depth_path: Path | None = None,
    prior_depth_path: Path | None = None,
    base_bev_run: Path | None = None,
) -> Path:
    """한 RGB에서 coarse cell scan 후 high-confidence subpixel 3D edge만 BEV에 반영한다.

    `evaluation_depth_path`는 RGB hash guard를 통과한 경우 결과 평가에만 사용되며,
    모델 입력이나 scale 정렬에는 전달되지 않는다. `base_bev_run`도 원본을 수정하지
    않고 좌표 설정을 검증한 뒤 실험 폴더에 융합 복사본만 만든다.
    """
    ensure_edge_output_roots(config)
    input_rgb = input_rgb or config.base.paths.input_rgb
    rgb = np.asarray(Image.open(input_rgb).convert("RGB"))
    expected_shape = (config.base.camera.height, config.base.camera.width)
    if rgb.shape[:2] != expected_shape:
        raise ValueError(f"입력 RGB shape {rgb.shape[:2]} != camera shape {expected_shape}")
    rays_result = build_camera_rays(config.base.camera)
    rays, valid = rays_result.rays_cv, rays_result.valid
    if base_bev_run is not None:
        _validate_base_bev_run(base_bev_run, config)
    evaluation_depth = None
    gt_used = False
    evaluation_hash_matches: bool | None = None
    if evaluation_depth_path is not None:
        if not evaluation_depth_path.exists():
            raise FileNotFoundError(f"evaluation depth를 찾지 못했습니다: {evaluation_depth_path}")
        evaluation_hash_matches = _sha256(input_rgb).lower() == config.base.eval.input_rgb_sha256.lower()
        if evaluation_hash_matches or not config.inference.hash_guard_evaluation_depth:
            evaluation_depth = np.load(evaluation_depth_path).astype(np.float32)
            if evaluation_depth.shape != expected_shape:
                raise ValueError("evaluation depth shape가 RGB/camera와 다릅니다.")
            gt_used = True
    prior_depth = None
    prior_depth_valid = None
    if prior_depth_path is not None:
        if not prior_depth_path.exists():
            raise FileNotFoundError(f"prior depth를 찾지 못했습니다: {prior_depth_path}")
        prior_depth = np.load(prior_depth_path).astype(np.float32)
        if prior_depth.shape != expected_shape:
            raise ValueError("prior depth shape가 RGB/camera와 다릅니다.")
        prior_depth_valid = np.isfinite(prior_depth) & (prior_depth > 0.0) & valid
    da_relative = None
    if "rgb_da_context" in checkpoints:
        da_runner = DepthAnythingMetricWrapper(config.base.paths, config.base.backbone)
        da_depth = da_runner.predict(rgb)
        da_relative = _relative_da(da_depth, valid)
        del da_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    edge_prior_2d = (
        estimate_2d_edge_prior(rgb, valid, config.edge_prior)
        if config.edge_prior.enabled
        else None
    )
    run_dir = config.output_root / "inference" / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_edge_config(config, run_dir / "config.json")
    device = _device()
    metrics: dict[str, dict] = {}
    for variant, checkpoint in checkpoints.items():
        model = _model_from_checkpoint(config, variant, checkpoint, device)
        prior = da_relative if variant == "rgb_da_context" else None
        coarse, coarse_type = _coarse_scan(model, rgb, rays, valid, prior, edge_prior_2d, config, device)
        candidates = _candidate_cells(coarse, rays, valid, config, edge_prior_2d)
        queries = _refine_candidates(
            model,
            candidates,
            coarse_type,
            rgb,
            rays,
            valid,
            prior,
            edge_prior_2d,
            prior_depth,
            prior_depth_valid,
            config,
            device,
        )
        selected = (
            (queries["edge_probability"] >= config.inference.query_edge_threshold)
            & (queries["confidence"] >= config.inference.confidence_threshold)
            & queries["nms_keep"]
            & np.isfinite(queries["depth_near_z"])
            & (queries["depth_near_z"] > 0.0)
            & (queries["ray_dir"][:, 2] > config.base.camera.geometry_z_eps)
            & np.isfinite(queries["source_uv"]).all(axis=-1)
            & (queries["source_uv"][:, 0] >= 0.5)
            & (queries["source_uv"][:, 0] <= config.base.camera.width - 0.5)
            & (queries["source_uv"][:, 1] >= 0.5)
            & (queries["source_uv"][:, 1] <= config.base.camera.height - 0.5)
        )
        metrics[variant] = _save_variant_result(
            run_dir,
            variant,
            rgb,
            coarse,
            queries,
            selected,
            rays,
            edge_prior_2d,
            config,
            evaluation_depth,
            base_bev_run,
        )
        metrics[variant]["coarse_candidate_cells"] = int(len(candidates))
        # 수백만 query 배열을 다음 variant의 모델을 올리기 전에 즉시 해제한다.
        # 그렇지 않으면 세 번째 variant에서 Python/NumPy 메모리 피크가 크게 증가한다.
        del model, coarse, coarse_type, candidates, queries, selected
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = {
        "pipeline": "isolated 5x5-context subpixel 3D edge completion experiment",
        "input_rgb": str(input_rgb.resolve()),
        "input_rgb_sha256": _sha256(input_rgb),
        "evaluation_depth": str(evaluation_depth_path.resolve()) if evaluation_depth_path is not None else None,
        "prior_depth": str(prior_depth_path.resolve()) if prior_depth_path is not None else None,
        "prior_depth_used_as_model_input": prior_depth_path is not None,
        "evaluation_rgb_hash_matches_reference": evaluation_hash_matches,
        "evaluation_gt_used": gt_used,
        "evaluation_gt_never_used_as_input_or_scale_alignment": prior_depth_path != evaluation_depth_path,
        "completed_semantics": "subpixel query rays are learned completion, not camera observations",
        "limitations": [
            "NYU의 pinhole FOV와 raw depth sensor noise가 pseudo edge 품질을 제한함",
            "NYU indoor scene과 Isaac warehouse fisheye 사이 domain gap이 존재함",
            "단일 RGB completion은 confidence가 있는 구조 prior이며 미관측 geometry의 실측 복원이 아님",
        ],
        "base_bev_run": str(base_bev_run.resolve()) if base_bev_run is not None else None,
        "variants": list(checkpoints),
        "checkpoints": {key: str(value.resolve()) for key, value in checkpoints.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_edge_report(run_dir, metadata, metrics, list(checkpoints))
    return run_dir


def latest_checkpoints(config: EdgeEstimateConfig) -> dict[Variant, Path]:
    result: dict[Variant, Path] = {}
    for variant in VARIANTS:
        compatible_run = _compatible_training_run(config, variant)
        if compatible_run is not None and compatible_run[2]:
            result[variant] = compatible_run[0] / "checkpoints" / "best.pt"
    return result
