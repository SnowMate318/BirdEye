from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wide_fov_supervision_v2.config import PipelineConfig
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths
from wide_fov_supervision_v2.modules.refiner import RayAwareQueryRefiner
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint
from wide_fov_supervision_v2.train.query_cache import query_sidecar_root
from wide_fov_supervision_v2.train.trainer import _predict_with_stencils, _query_from_sidecar, _query_tensors, _to_bchw


def depth_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """dense 또는 query z-depth의 AbsRel/RMSE를 계산하는 공개 helper."""

    valid = mask & np.isfinite(pred) & np.isfinite(target) & (pred > 0.0) & (target > 0.0)
    if not np.any(valid):
        return {"valid": 0, "absrel": float("nan"), "rmse": float("nan")}
    prediction = pred[valid].astype(np.float64)
    reference = target[valid].astype(np.float64)
    return {
        "valid": int(valid.sum()),
        "absrel": float(np.mean(np.abs(prediction - reference) / np.clip(reference, 1.0e-6, None))),
        "rmse": float(np.sqrt(np.mean((prediction - reference) ** 2))),
    }


def _new_accumulator() -> dict[str, float]:
    return {
        "depth_valid": 0.0,
        "d0_absrel_sum": 0.0,
        "dstar_absrel_sum": 0.0,
        "d0_sq_error_sum": 0.0,
        "dstar_sq_error_sum": 0.0,
        "normal_valid": 0.0,
        "d0_normal_angle_sum": 0.0,
        "dstar_normal_angle_sum": 0.0,
    }


def _accumulate_depth(
    accumulator: dict[str, float],
    depth0: torch.Tensor,
    depth_star: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = mask & torch.isfinite(depth0) & torch.isfinite(depth_star) & torch.isfinite(target)
    valid &= (depth0 > 0.0) & (depth_star > 0.0) & (target > 0.0)
    if not torch.any(valid):
        return
    d0 = depth0[valid].double()
    dstar = depth_star[valid].double()
    gt = target[valid].double()
    accumulator["depth_valid"] += float(valid.sum().item())
    accumulator["d0_absrel_sum"] += float((torch.abs(d0 - gt) / gt.clamp_min(1.0e-6)).sum().item())
    accumulator["dstar_absrel_sum"] += float((torch.abs(dstar - gt) / gt.clamp_min(1.0e-6)).sum().item())
    accumulator["d0_sq_error_sum"] += float(((d0 - gt) ** 2).sum().item())
    accumulator["dstar_sq_error_sum"] += float(((dstar - gt) ** 2).sum().item())


def _accumulate_normal(
    accumulator: dict[str, float],
    normal0: torch.Tensor,
    normal_star: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = mask & torch.isfinite(normal0).all(dim=-1) & torch.isfinite(normal_star).all(dim=-1)
    valid &= torch.isfinite(target).all(dim=-1)
    if not torch.any(valid):
        return
    n0 = torch.nn.functional.normalize(normal0[valid].double(), dim=-1)
    nstar = torch.nn.functional.normalize(normal_star[valid].double(), dim=-1)
    gt = torch.nn.functional.normalize(target[valid].double(), dim=-1)
    angle0 = torch.rad2deg(torch.acos((n0 * gt).sum(dim=-1).clamp(-1.0, 1.0)))
    angle_star = torch.rad2deg(torch.acos((nstar * gt).sum(dim=-1).clamp(-1.0, 1.0)))
    accumulator["normal_valid"] += float(valid.sum().item())
    accumulator["d0_normal_angle_sum"] += float(angle0.sum().item())
    accumulator["dstar_normal_angle_sum"] += float(angle_star.sum().item())


def _finish_accumulator(accumulator: dict[str, float]) -> dict[str, float | int]:
    depth_count = int(accumulator["depth_valid"])
    normal_count = int(accumulator["normal_valid"])
    result: dict[str, float | int] = {"depth_valid": depth_count, "normal_valid": normal_count}
    if depth_count:
        result.update(
            {
                "d0_absrel": accumulator["d0_absrel_sum"] / depth_count,
                "dstar_absrel": accumulator["dstar_absrel_sum"] / depth_count,
                "d0_rmse": float(np.sqrt(accumulator["d0_sq_error_sum"] / depth_count)),
                "dstar_rmse": float(np.sqrt(accumulator["dstar_sq_error_sum"] / depth_count)),
                "absrel_improvement": (accumulator["d0_absrel_sum"] - accumulator["dstar_absrel_sum"]) / depth_count,
            }
        )
    if normal_count:
        result.update(
            {
                "d0_normal_angular_error_deg": accumulator["d0_normal_angle_sum"] / normal_count,
                "dstar_normal_angular_error_deg": accumulator["dstar_normal_angle_sum"] / normal_count,
                "normal_angular_improvement_deg": (
                    accumulator["d0_normal_angle_sum"] - accumulator["dstar_normal_angle_sum"]
                )
                / normal_count,
            }
        )
    return result


def evaluate_cached_predictions(config: PipelineConfig) -> Path:
    """고정 uniform query와 guided sparse query에서 D0/D*를 같은 GT와 비교한다."""

    indices = read_nyu_split(config.paths.nyu_split_test)
    if config.train.max_eval_items is not None:
        indices = indices[: int(config.train.max_eval_items)]
    teacher_root = config.paths.outputs / "cache" / "nyu" / "test"
    sidecar_root = query_sidecar_root(config) / "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RayAwareQueryRefiner(config.refiner).to(device).eval()
    checkpoint_loaded = False
    if config.paths.checkpoint is not None and Path(config.paths.checkpoint).exists():
        load_checkpoint(Path(config.paths.checkpoint), model, map_location=device)
        checkpoint_loaded = True

    accumulators = {"fixed_uniform": _new_accumulator(), "guided_sparse": _new_accumulator(), "all": _new_accumulator()}
    evaluated_items = 0
    with torch.inference_mode():
        for index in indices:
            teacher_path = teacher_root / f"{int(index):06d}.npz"
            sidecar_path = sidecar_root / f"{int(index):06d}.npz"
            if not teacher_path.exists() or not sidecar_path.exists():
                continue
            with np.load(teacher_path) as data:
                sample = {name: data[name].copy() for name in data.files}
            with np.load(sidecar_path) as data:
                payload = {name: data[name].copy() for name in data.files}
            queries = _query_from_sidecar(payload)
            if len(queries) == 0:
                continue
            source_all = _to_bchw(sample, device)
            source = source_all[:5]
            center, normal_star, stencil_pred_valid, stencil = _predict_with_stencils(
                model,
                config,
                source,
                queries,
                payload["stencil_ray_dir"],
                payload["stencil_source_observed"],
            )
            q_rays, _, _, _, q_observed = _query_tensors(queries, device)
            stencil_rays = torch.from_numpy(payload["stencil_ray_dir"].astype(np.float32)).unsqueeze(0).to(device)
            target_depth = torch.from_numpy(payload["query_depth_gt_z"].astype(np.float32)).unsqueeze(0).to(device)
            target_stencil = torch.from_numpy(payload["stencil_depth_gt_z"].astype(np.float32)).unsqueeze(0).to(device)
            target_valid = torch.from_numpy(payload["query_gt_valid"].astype(bool)).unsqueeze(0).to(device)
            target_stencil_valid = torch.from_numpy(payload["stencil_gt_valid"].astype(bool)).unsqueeze(0).to(device)
            target_normal, target_normal_valid = normals_from_stencil_depths(
                target_depth,
                target_stencil,
                q_rays,
                stencil_rays,
                z_eps=config.camera.geometry_z_eps,
            )
            d0_stencil = stencil["depth0_query_z"].reshape(1, -1, 4)
            normal0_depth, normal0_valid = normals_from_stencil_depths(
                center["depth0_query_z"],
                d0_stencil,
                q_rays,
                stencil_rays,
                z_eps=config.camera.geometry_z_eps,
            )
            base_mask = (
                (q_observed > 0.5)
                & (center["source_valid_query"] > 0.5)
                & target_valid
            )
            normal_mask = (
                base_mask
                & target_stencil_valid.all(dim=-1)
                & target_normal_valid
                & normal0_valid
                & stencil_pred_valid
            )
            group_masks = {
                "fixed_uniform": torch.from_numpy(payload["uniform"].astype(bool)).unsqueeze(0).to(device),
                "guided_sparse": torch.from_numpy(payload["guided"].astype(bool)).unsqueeze(0).to(device),
                "all": torch.ones_like(base_mask),
            }
            for name, group_mask in group_masks.items():
                _accumulate_depth(
                    accumulators[name],
                    center["depth0_query_z"],
                    center["depth_final_z"],
                    target_depth,
                    base_mask & group_mask,
                )
                _accumulate_normal(
                    accumulators[name],
                    normal0_depth,
                    normal_star,
                    target_normal,
                    normal_mask & group_mask,
                )
            evaluated_items += 1

    summary = {
        "items": evaluated_items,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint": str(config.paths.checkpoint) if config.paths.checkpoint is not None else None,
        "query_groups": {name: _finish_accumulator(values) for name, values in accumulators.items()},
    }
    out_dir = config.paths.outputs / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
