from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch

from wide_fov_supervision_v2.backbone.depth_source import ExternalDepthPrediction, load_external_z_depth
from wide_fov_supervision_v2.backbone.runner import BackbonePrediction, BackboneRunner
from wide_fov_supervision_v2.config import PipelineConfig, config_to_dict, ensure_output_roots, make_default_config, save_config
from wide_fov_supervision_v2.generate_html.dashboard import generate_dashboard
from wide_fov_supervision_v2.modules.adaptive_ray import (
    AdaptiveRayResult,
    RayQuerySet,
    generate_front_hemisphere_queries,
    generate_guided_observed_queries,
    generate_source_queries,
    merge_query_sets,
)
from wide_fov_supervision_v2.modules.bev_mapping import build_bev_outputs, build_bev_valid
from wide_fov_supervision_v2.modules.camera_geometry import (
    build_fisheye_rays,
    camera_to_world_points,
    cell_angular_gap,
    cv_rays_to_world,
    points_from_z_depth,
    project_fisheye_rays,
)
from wide_fov_supervision_v2.modules.dense_coverage import build_dense_coverage_bev
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths, query_stencil_rays, sample_at_uv
from wide_fov_supervision_v2.modules.refiner import RayAwareQueryRefiner
from wide_fov_supervision_v2.modules.visualization import save_coverage, save_depth, save_heatmap, save_mask, save_normal, save_rgb
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint


def load_rgb(path: Path) -> np.ndarray:
    """RGB PNG/JPG를 `uint8 (H,W,3)`로 읽는다."""

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    """파일 SHA256 hash를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(config: PipelineConfig, mode: str) -> Path:
    """timestamp run directory를 만든다."""

    root = config.paths.outputs / ("inference" if mode == "infer" else mode)
    run_dir = root / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_prediction(run_dir: Path, pred: BackbonePrediction) -> None:
    branch_dir = run_dir / pred.branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    np.save(branch_dir / "depth0_z.npy", pred.depth0_z.astype(np.float32))
    np.save(branch_dir / "normal0.npy", pred.normal0.astype(np.float32))
    np.save(branch_dir / "valid.npy", pred.valid.astype(bool))
    save_depth(branch_dir / "depth0_z.png", pred.depth0_z)
    save_normal(branch_dir / "normal0.png", pred.normal0)


def _prediction_to_tensors(pred: BackbonePrediction, rgb: np.ndarray, source_rays: np.ndarray, lens_valid: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    source_depth0 = torch.from_numpy(np.nan_to_num(pred.depth0_z.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)).unsqueeze(0).unsqueeze(0).to(device)
    source_normal0 = torch.from_numpy(np.nan_to_num(pred.normal0.astype(np.float32))).permute(2, 0, 1).unsqueeze(0).to(device)
    source_rays_t = torch.from_numpy(source_rays.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    valid = lens_valid & pred.valid & np.isfinite(pred.depth0_z) & (pred.depth0_z > 0.0) & np.isfinite(pred.normal0).all(axis=-1)
    source_valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    return source_rgb, source_depth0, source_normal0, source_rays_t, source_valid


def _query_tensors(query: RayQuerySet, sl: slice, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = query.subset(np.arange(len(query))[sl])
    rays = torch.from_numpy(q.ray_dir.astype(np.float32)).unsqueeze(0).to(device)
    uv = torch.from_numpy(q.source_uv.astype(np.float32)).unsqueeze(0).to(device)
    rel = torch.from_numpy(np.nan_to_num(q.relative_uv.astype(np.float32))).unsqueeze(0).to(device)
    sampling = torch.from_numpy(np.nan_to_num(q.sampling_features.astype(np.float32))).unsqueeze(0).to(device)
    obs = torch.from_numpy(q.observed.astype(np.float32)).unsqueeze(0).to(device)
    return rays, uv, rel, sampling, obs


def run_refiner_on_queries(
    config: PipelineConfig,
    pred: BackbonePrediction,
    rgb: np.ndarray,
    source_rays: np.ndarray,
    lens_valid: np.ndarray,
    queries: RayQuerySet,
) -> dict[str, np.ndarray]:
    """query set 전체를 chunk 단위로 refiner에 통과시킨다."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RayAwareQueryRefiner(config.refiner).to(device).eval() if config.toggles.enable_refiner else None
    if model is not None and config.paths.checkpoint is not None and Path(config.paths.checkpoint).exists():
        load_checkpoint(Path(config.paths.checkpoint), model, map_location=device)
    source = _prediction_to_tensors(pred, rgb, source_rays, lens_valid, device)
    depth_final = np.full((len(queries),), np.nan, dtype=np.float32)
    depth0_query = np.full((len(queries),), np.nan, dtype=np.float32)
    delta = np.zeros((len(queries),), dtype=np.float32)
    normal_final = np.full((len(queries), 3), np.nan, dtype=np.float32)
    source_valid_query = np.zeros((len(queries),), dtype=bool)
    chunk = int(config.ray.query_chunk_size)
    with torch.inference_mode():
        source_features = model.encode_source(*source) if model is not None else None
        for start in range(0, len(queries), chunk):
            stop = min(start + chunk, len(queries))
            q_rays, q_uv, q_rel, q_sampling, q_obs = _query_tensors(queries, slice(start, stop), device)
            if model is not None:
                center = model.decode_queries(
                    source[1], source[2], source[3], source[4], source_features,
                    q_rays, q_uv, q_rel, q_sampling, q_obs,
                )
            else:
                d0_query = sample_at_uv(source[1], q_uv).squeeze(-1)
                valid_query = sample_at_uv(source[4], q_uv).squeeze(-1)
                normal0_query = torch.nn.functional.normalize(sample_at_uv(source[2], q_uv), dim=-1)
                source_ray_query = torch.nn.functional.normalize(sample_at_uv(source[3], q_uv), dim=-1)
                finite = torch.isfinite(d0_query) & (d0_query > 0.0) & (valid_query > 0.5)
                center = {
                    "depth_final_z": torch.where(finite, d0_query, torch.full_like(d0_query, float("nan"))),
                    "delta_log_depth": torch.zeros_like(d0_query),
                    "depth0_query_z": d0_query,
                    "normal0_query": normal0_query,
                    "source_ray_query": source_ray_query,
                    "source_valid_query": valid_query,
                }
            stencil_rays = query_stencil_rays(q_rays, config.ray.stencil_step_rad)
            flat_stencil = stencil_rays.reshape(1, -1, 3)
            uv_np, uv_valid_np = project_fisheye_rays(flat_stencil.cpu().numpy()[0], config.camera)
            s_uv = torch.from_numpy(uv_np.astype(np.float32)).unsqueeze(0).to(device)
            s_valid = torch.from_numpy(uv_valid_np.astype(np.float32)).unsqueeze(0).to(device)
            s_rel = q_rel.repeat_interleave(4, dim=1)
            s_sampling = q_sampling.repeat_interleave(4, dim=1)
            s_obs = q_obs.repeat_interleave(4, dim=1) * s_valid
            if model is not None:
                stencil = model.decode_queries(
                    source[1], source[2], source[3], source[4], source_features,
                    flat_stencil, s_uv, s_rel, s_sampling, s_obs,
                )
                stencil_depth = stencil["depth_final_z"].reshape(1, -1, 4)
                stencil_source_valid = stencil["source_valid_query"].reshape(1, -1, 4) > 0.5
            else:
                stencil_depth = sample_at_uv(source[1], s_uv).squeeze(-1).reshape(1, -1, 4)
                stencil_source_valid = sample_at_uv(source[4], s_uv).squeeze(-1).reshape(1, -1, 4) > 0.5
            n_star, n_valid = normals_from_stencil_depths(center["depth_final_z"], stencil_depth, q_rays, stencil_rays, z_eps=config.camera.geometry_z_eps)
            all_depth = torch.cat([center["depth_final_z"].unsqueeze(-1), stencil_depth], dim=-1)
            finite_depth = torch.isfinite(all_depth).all(dim=-1) & (all_depth > 0.0).all(dim=-1)
            safe_log = torch.log(torch.nan_to_num(all_depth, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-4))
            continuous = (
                safe_log.amax(dim=-1) - safe_log.amin(dim=-1)
                <= float(config.loss.depth_discontinuity_log_threshold)
            )
            n_valid &= stencil_source_valid.all(dim=-1) & finite_depth & continuous
            depth_final[start:stop] = center["depth_final_z"][0].detach().cpu().numpy().astype(np.float32)
            depth0_query[start:stop] = center["depth0_query_z"][0].detach().cpu().numpy().astype(np.float32)
            delta[start:stop] = center["delta_log_depth"][0].detach().cpu().numpy().astype(np.float32)
            nf = n_star[0].detach().cpu().numpy().astype(np.float32)
            nf[~n_valid[0].detach().cpu().numpy()] = np.nan
            normal_final[start:stop] = nf
            source_valid_query[start:stop] = (center["source_valid_query"][0].detach().cpu().numpy() > 0.5)
    depth_final[queries.unknown] = np.nan
    depth0_query[queries.unknown] = np.nan
    normal_final[queries.unknown] = np.nan
    return {
        "depth_final_z": depth_final,
        "depth0_query_z": depth0_query,
        "delta_log_depth": delta,
        "normal_final": normal_final,
        "source_valid_query": source_valid_query,
    }


def _sample_query_rgb(rgb: np.ndarray, queries: RayQuerySet) -> np.ndarray:
    uv = queries.source_uv.astype(np.float32)
    colors = np.zeros((len(queries), 3), dtype=np.uint8)
    # OpenCV remap은 dst width/height가 SHRT_MAX보다 작아야 하므로 query를
    # 1xQ strip으로 한 번에 보내지 않고 안전한 chunk로 나눠 sampling한다.
    chunk = 30_000
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        map_x = (uv[start:stop, 0] - 0.5).reshape(1, -1)
        map_y = (uv[start:stop, 1] - 0.5).reshape(1, -1)
        colors[start:stop] = cv2.remap(
            rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[0]
    colors[queries.unknown] = 0
    return colors.astype(np.uint8)


def _sample_query_map(value: np.ndarray, queries: RayQuerySet) -> np.ndarray:
    """dense scalar map을 query의 source_uv에서 bilinear sampling한다."""

    uv = queries.source_uv.astype(np.float32)
    sampled = np.full((len(queries),), np.nan, dtype=np.float32)
    chunk = 30_000
    source = np.asarray(value, dtype=np.float32)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        map_x = (uv[start:stop, 0] - 0.5).reshape(1, -1)
        map_y = (uv[start:stop, 1] - 0.5).reshape(1, -1)
        sampled[start:stop] = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )[0]
    sampled[queries.unknown] = np.nan
    return sampled


def _query_depth_metrics(prefix: str, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """동일 query 위치에서 z-depth AbsRel/RMSE를 계산한다."""

    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0.0)
        & (target > 0.0)
    )
    if not np.any(valid):
        return {f"{prefix}_valid": 0}
    pred = prediction[valid].astype(np.float64)
    gt = target[valid].astype(np.float64)
    return {
        f"{prefix}_valid": int(valid.sum()),
        f"{prefix}_absrel": float(np.mean(np.abs(pred - gt) / np.clip(gt, 1.0e-6, None))),
        f"{prefix}_rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
    }


def _query_points(config: PipelineConfig, queries: RayQuerySet, depth_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_points, radial, valid = points_from_z_depth(depth_z, queries.ray_dir, z_eps=config.camera.geometry_z_eps)
    valid &= queries.observed & ~queries.unknown
    world_points = camera_to_world_points(camera_points, config.camera).astype(np.float32)
    world_points[~valid] = np.nan
    camera_points[~valid] = np.nan
    return camera_points, world_points, radial, valid


def _normals_to_world(normals_cv: np.ndarray, config: PipelineConfig) -> np.ndarray:
    out = cv_rays_to_world(normals_cv, np.asarray(config.camera.world_from_camera, dtype=np.float64))
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1.0e-6, None)
    out[~np.isfinite(norm[..., 0])] = np.nan
    return out.astype(np.float32)


def _load_matching_isaac_gt(config: PipelineConfig, rgb_hash: str) -> dict[str, np.ndarray] | None:
    if not config.toggles.enable_gt_evaluation:
        return None
    if config.eval.use_isaac_gt_only_on_hash_match and rgb_hash.lower() != config.eval.input_rgb_sha256.lower():
        return None
    ref = config.paths.isaac_reference_run
    depth_path = ref / "depth_z.npy"
    if not depth_path.exists():
        return None
    return {"depth_z": np.load(depth_path).astype(np.float32)}


def _error_maps(pred_depth: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    mask = valid & np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0.0) & (gt_depth > 0.0)
    err = np.full_like(pred_depth, np.nan, dtype=np.float32)
    err[mask] = np.abs(pred_depth[mask] - gt_depth[mask]) / np.clip(gt_depth[mask], 1.0e-6, None)
    metrics = {"gt_valid": int(mask.sum())}
    if np.any(mask):
        metrics.update(
            {
                "gt_absrel": float(np.mean(err[mask])),
                "gt_rmse": float(np.sqrt(np.mean((pred_depth[mask] - gt_depth[mask]) ** 2))),
            }
        )
    return err, metrics


def _finite_stats(prefix: str, values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    """유효한 spacing 값의 percentile을 JSON용 scalar로 요약한다."""

    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    samples = np.asarray(values, dtype=np.float64)[valid]
    if len(samples) == 0:
        return {f"{prefix}_count": 0}
    quantiles = np.percentile(samples, [50.0, 90.0, 95.0, 99.0])
    return {
        f"{prefix}_count": int(len(samples)),
        f"{prefix}_p50": float(quantiles[0]),
        f"{prefix}_p90": float(quantiles[1]),
        f"{prefix}_p95": float(quantiles[2]),
        f"{prefix}_p99": float(quantiles[3]),
        f"{prefix}_max": float(np.max(samples)),
    }


def _save_adaptive_diagnostics(run_dir: Path, result: AdaptiveRayResult) -> None:
    """guided sampler의 원본 수치 map과 비교 가능한 PNG를 함께 저장한다."""

    arrays = {
        "ray_gap_before": result.angular_gap_before,
        "ray_gap_after": result.angular_gap_planned_after,
        "surface_gap_before_m": result.surface_gap_before_m,
        "surface_gap_planned_after_m": result.surface_gap_planned_after_m,
        "bev_gap_before_cells": result.bev_gap_before_cells,
        "bev_gap_planned_after_cells": result.bev_gap_planned_after_cells,
        "sampling_priority": result.sampling_priority,
        "planned_added_ray_density": result.added_density,
    }
    for name, array in arrays.items():
        np.save(run_dir / f"{name}.npy", np.asarray(array))

    # before/after 쌍은 같은 color scale을 사용해야 색만 보고 실제 감소량을 비교할 수 있다.
    for before_name, after_name in (
        ("ray_gap_before", "ray_gap_after"),
        ("surface_gap_before_m", "surface_gap_planned_after_m"),
        ("bev_gap_before_cells", "bev_gap_planned_after_cells"),
    ):
        before = arrays[before_name]
        finite = before[np.isfinite(before)]
        high = float(np.percentile(finite, 99.0)) if len(finite) else 1.0
        save_heatmap(run_dir / f"{before_name}.png", before, value_min=0.0, value_max=max(high, 1.0e-8))
        save_heatmap(run_dir / f"{after_name}.png", arrays[after_name], value_min=0.0, value_max=max(high, 1.0e-8))

    save_heatmap(run_dir / "sampling_priority.png", result.sampling_priority)
    save_heatmap(run_dir / "planned_added_ray_density.png", result.added_density)
    save_mask(run_dir / "sampling_eligible.png", result.eligible_mask)
    np.save(run_dir / "sampling_eligible.npy", result.eligible_mask.astype(bool))


def _density_region_metrics(density: np.ndarray) -> dict[str, float]:
    """영상 중심과 외곽의 added-query density를 같은 cell 면적으로 비교한다."""

    height, width = density.shape
    yy, xx = np.mgrid[:height, :width]
    nx = (xx - (width - 1) * 0.5) / max(width * 0.5, 1.0)
    ny = (yy - (height - 1) * 0.5) / max(height * 0.5, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    center = radius < 0.25
    outer = (radius >= 0.75) & (radius < 1.0)
    return {
        "added_density_center_mean": float(np.mean(density[center])) if np.any(center) else 0.0,
        "added_density_outer_mean": float(np.mean(density[outer])) if np.any(outer) else 0.0,
    }


def validate_environment(config: PipelineConfig | None = None) -> dict:
    """경로, split, projection round-trip을 빠르게 검증한다."""

    config = config or make_default_config()
    rays = build_fisheye_rays(config.camera)
    gap, _ = cell_angular_gap(rays.rays_cv, rays.valid)
    checks = {
        "depth_source": config.backbone.depth_source,
        "input_rgb_exists": config.paths.input_rgb.exists(),
        "nyu_mat_exists": config.paths.nyu_mat.exists(),
        "depth_anything_root_exists": config.paths.depth_anything_root.exists(),
        "depth_anything_vitl_ckpt_exists": config.paths.depth_anything_vitl_ckpt.exists(),
        "dsine_root_exists": config.paths.dsine_root.exists(),
        "dsine_ckpt_exists": config.paths.dsine_ckpt.exists(),
        "fisheye_valid_pixels": int(rays.valid.sum()),
        "fisheye_roundtrip_max_error_px": rays.max_roundtrip_error_px,
        "gap_before_max_rad": float(np.nanmax(gap)),
    }
    if config.backbone.depth_source == "external_npy":
        try:
            external = load_external_z_depth(
                config.paths.external_depth_z,
                (config.camera.height, config.camera.width),
            )
            checks.update(
                {
                    "external_depth_exists": True,
                    "external_depth_shape_valid": True,
                    "external_depth_valid_pixels": int(external.valid.sum()),
                    "external_depth_sha256": external.metadata["external_depth_sha256"],
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            checks.update(
                {
                    "external_depth_exists": config.paths.external_depth_z.exists(),
                    "external_depth_shape_valid": False,
                    "external_depth_error": str(exc),
                }
            )
    return checks


def run_inference(config: PipelineConfig | None = None) -> Path:
    """현재 ``rgb.png``에 3D·BEV guided ray 보완 파이프라인을 실행한다.

    backbone D0가 있어야 실제 surface/BEV 간격을 계산할 수 있으므로, 과거의
    angular-only 구현과 달리 backbone을 query sampler보다 먼저 실행한다.
    """

    config = config or make_default_config()
    ensure_output_roots(config)
    run_dir = create_run_dir(config, "infer")
    rgb = load_rgb(config.paths.input_rgb)
    rgb_hash = sha256_file(config.paths.input_rgb)
    depth_source = config.backbone.depth_source
    if depth_source not in {"da_v2", "external_npy"}:
        raise ValueError(f"Unsupported depth_source={depth_source!r}; expected 'da_v2' or 'external_npy'.")
    external_depth: ExternalDepthPrediction | None = None
    if depth_source == "external_npy":
        external_depth = load_external_z_depth(config.paths.external_depth_z, rgb.shape[:2])
    save_rgb(run_dir / "source_rgb.png", rgb)
    shutil.copy2(config.paths.input_rgb, run_dir / "input_rgb_original.png")
    save_config(config, run_dir / "config.json")

    rays = build_fisheye_rays(config.camera)
    np.save(run_dir / "source_rays.npy", rays.rays_cv.astype(np.float32))

    # D0/N0가 먼저 만들어져야 추가 ray를 실제 3D·BEV 희소 위치에 배치할 수 있다.
    predictions: list[BackbonePrediction] = []
    runner = BackboneRunner(config.paths, config.backbone, config.camera)
    depth_override_z = None if external_depth is None else external_depth.depth_z
    depth_metadata = None if external_depth is None else external_depth.metadata
    if config.toggles.enable_direct_backbone:
        direct = runner.run_direct(
            rgb,
            depth_override_z=depth_override_z,
            depth_metadata=depth_metadata,
        )
        _save_prediction(run_dir, direct)
        predictions.append(direct)
    if config.toggles.enable_tangent_backbone:
        tangent = runner.run_tangent(
            rgb,
            rays.rays_cv,
            depth_override_z=depth_override_z,
            depth_metadata=depth_metadata,
        )
        _save_prediction(run_dir, tangent)
        predictions.append(tangent)
    if not predictions:
        raise RuntimeError("At least one backbone branch must be enabled.")
    primary = next((prediction for prediction in predictions if prediction.branch == "tangent"), predictions[0])
    guidance_valid = (
        rays.valid
        & primary.valid
        & np.isfinite(primary.depth0_z)
        & (primary.depth0_z > 0.0)
        & np.isfinite(primary.normal0).all(axis=-1)
    )

    adaptive = generate_guided_observed_queries(
        rays.rays_cv,
        guidance_valid,
        primary.depth0_z,
        config.camera,
        config.bev,
        config.ray,
        mode="surface_bev",
    )
    _save_adaptive_diagnostics(run_dir, adaptive)
    source_queries = generate_source_queries(rays.rays_cv, guidance_valid)
    observed_parts = [source_queries]
    if config.toggles.enable_adaptive_ray_generation:
        observed_parts.append(adaptive.queries)
    queries = merge_query_sets(observed_parts, config.ray)
    applied_density = np.zeros_like(adaptive.added_density, dtype=np.float32)
    applied_parent = queries.parent_cell[queries.added]
    if len(applied_parent):
        np.add.at(applied_density, (applied_parent[:, 0], applied_parent[:, 1]), 1.0)
    np.save(run_dir / "adaptive_added_ray_density.npy", applied_density)
    save_heatmap(run_dir / "adaptive_added_ray_density.png", applied_density)
    np.save(run_dir / "added_ray_density.npy", applied_density)
    save_heatmap(run_dir / "added_ray_density.png", applied_density)

    # 180도 후보는 coverage 전용으로 분리한다. Refiner/point/BEV query와 합치지 않는다.
    coverage = np.zeros((config.camera.height, config.camera.width), dtype=np.uint8)
    hemisphere_count = 0
    hemisphere_unknown_count = 0
    if config.toggles.enable_front_hemisphere_queries:
        hemisphere_queries, coverage = generate_front_hemisphere_queries(
            config.camera,
            rays.valid,
            None,
            config.ray,
        )
        hemisphere_count = len(hemisphere_queries)
        hemisphere_unknown_count = int(hemisphere_queries.unknown.sum())
        hemisphere_queries.save_npz(
            run_dir / "front_hemisphere_queries.npz",
            depth0_z=np.full(hemisphere_count, np.nan, dtype=np.float32),
            depth_final_z=np.full(hemisphere_count, np.nan, dtype=np.float32),
            normal_final=np.full((hemisphere_count, 3), np.nan, dtype=np.float32),
        )
    save_coverage(run_dir / "front_hemisphere_coverage.png", coverage)

    # frozen foundation model은 Refiner와 동시에 GPU에 둘 필요가 없다.
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # query 위치는 primary D0로 한 번만 정하고 모든 branch에 똑같이 적용한다.
    refiner_outputs: dict[str, dict[str, np.ndarray]] = {}
    for prediction in predictions:
        branch_out = run_refiner_on_queries(config, prediction, rgb, rays.rays_cv, rays.valid, queries)
        refiner_outputs[prediction.branch] = branch_out
        branch_dir = run_dir / prediction.branch
        np.save(branch_dir / "query_depth_final_z.npy", branch_out["depth_final_z"])
        np.save(branch_dir / "query_normal_final.npy", branch_out["normal_final"])
        np.save(branch_dir / "query_delta_log_depth.npy", branch_out["delta_log_depth"])
    refiner_out = refiner_outputs[primary.branch]

    camera_points, world_points, radial, point_valid = _query_points(config, queries, refiner_out["depth_final_z"])
    colors = _sample_query_rgb(rgb, queries)
    normals_world = _normals_to_world(refiner_out["normal_final"], config)
    np.save(run_dir / "query_points_camera.npy", camera_points.astype(np.float32))
    np.save(run_dir / "query_points_world.npy", world_points.astype(np.float32))
    np.save(run_dir / "query_points_rgb.npy", colors.astype(np.uint8))
    queries.save_npz(
        run_dir / "ray_queries.npz",
        depth0_z=refiner_out["depth0_query_z"].astype(np.float32),
        depth_final_z=refiner_out["depth_final_z"].astype(np.float32),
        normal_final=refiner_out["normal_final"].astype(np.float32),
        radial_depth=radial.astype(np.float32),
        point_valid=point_valid.astype(bool),
        source_valid_query=refiner_out["source_valid_query"].astype(bool),
    )

    selected_cells = applied_density > 0.0
    metrics: dict = {
        "rgb_sha256": rgb_hash,
        "depth_source": depth_source,
        "primary_branch": primary.branch,
        "query_guidance_branch": primary.branch,
        "sampling_mode": "surface_bev",
        "target_angular_gap_rad": adaptive.target_angular_gap_rad,
        "target_surface_gap_m": config.ray.target_surface_gap_m,
        "target_bev_gap_cells": config.ray.target_bev_gap_cells,
        "source_query_count": int(len(source_queries)),
        "adaptive_generated_query_count": int(len(adaptive.queries)),
        "adaptive_added_query_count": int(queries.added.sum()),
        "refiner_query_count": int(len(queries)),
        "hemisphere_query_count": int(hemisphere_count),
        "hemisphere_unknown_query_count": int(hemisphere_unknown_count),
        "eligible_cell_count": int(adaptive.eligible_mask.sum()),
        "selected_cell_count": int(selected_cells.sum()),
        "candidate_query_count": int(adaptive.candidate_query_count),
        "query_budget": int(adaptive.query_budget),
        "query_budget_truncated": bool(adaptive.budget_truncated),
        "refiner_enabled": bool(config.toggles.enable_refiner),
        "refiner_checkpoint_loaded": bool(
            config.toggles.enable_refiner
            and config.paths.checkpoint is not None
            and Path(config.paths.checkpoint).exists()
        ),
    }
    adaptive_density_metrics = _density_region_metrics(applied_density)
    metrics.update(adaptive_density_metrics)
    metrics.update({f"adaptive_{key}": value for key, value in adaptive_density_metrics.items()})
    metrics.update(_finite_stats("surface_gap_before_selected_m", adaptive.surface_gap_before_m, selected_cells))
    metrics.update(_finite_stats("surface_gap_planned_after_selected_m", adaptive.surface_gap_planned_after_m, selected_cells))
    metrics.update(_finite_stats("bev_gap_before_selected_cells", adaptive.bev_gap_before_cells, selected_cells))
    metrics.update(_finite_stats("bev_gap_planned_after_selected_cells", adaptive.bev_gap_planned_after_cells, selected_cells))

    if external_depth is not None:
        metrics.update(external_depth.metadata)

    gt = _load_matching_isaac_gt(config, rgb_hash)
    gt_is_external_input = False
    if gt is not None and external_depth is not None:
        gt_path = config.paths.isaac_reference_run / "depth_z.npy"
        gt_is_external_input = (
            gt_path.is_file()
            and sha256_file(gt_path) == external_depth.metadata["external_depth_sha256"]
        )
    if gt_is_external_input:
        metrics["gt_evaluation_skipped"] = True
        metrics["gt_evaluation_skipped_reason"] = (
            "external depth input is byte-identical to the Isaac evaluation GT; metrics would leak ground truth"
        )
    elif gt is not None:
        # GT는 hash가 같은 reference에서 평가에만 사용하며 sampler/refiner 입력에는 사용하지 않는다.
        error_map, gt_metrics = _error_maps(primary.depth0_z, gt["depth_z"], guidance_valid)
        save_heatmap(run_dir / "depth_gt_absrel_error.png", error_map)
        np.save(run_dir / "depth_gt_absrel_error.npy", error_map)
        metrics.update(gt_metrics)
        target_query_depth = _sample_query_map(gt["depth_z"], queries)
        query_mask = point_valid & queries.observed
        metrics.update(_query_depth_metrics("query_d0", refiner_out["depth0_query_z"], target_query_depth, query_mask))
        metrics.update(_query_depth_metrics("query_dstar", refiner_out["depth_final_z"], target_query_depth, query_mask))

    if config.toggles.enable_bev:
        base_camera_points, _, base_valid = points_from_z_depth(
            primary.depth0_z,
            rays.rays_cv,
            z_eps=config.camera.geometry_z_eps,
        )
        base_valid &= guidance_valid
        base_world_points = camera_to_world_points(base_camera_points, config.camera).astype(np.float32)
        bev_valid_before = build_bev_valid(base_world_points, base_valid, config.bev)
        # coverage 증가는 added ray만으로 귀속한다. source D*가 D0 point를 이동시킨
        # 효과를 "새 ray가 채운 cell"로 잘못 세지 않는다.
        added_bev_valid = build_bev_valid(world_points, point_valid & queries.added, config.bev)
        bev_valid_after_adaptive = np.maximum(bev_valid_before, added_bev_valid)

        bev = build_bev_outputs(
            colors,
            world_points,
            normals_world,
            point_valid,
            config.bev,
            normal_valid_mask=np.isfinite(normals_world).all(axis=-1),
        )
        final_bev_rgb = bev.bev_rgb
        final_bev_valid = np.maximum(bev_valid_after_adaptive, bev.bev_valid)
        final_observed_top = bev.observed_top_occupancy
        final_top_probability = bev.top_probability_map
        combined_density = applied_density.copy()

        if config.toggles.enable_dense_coverage_bev:
            dense = build_dense_coverage_bev(
                rgb,
                primary.depth0_z,
                rays.rays_cv,
                guidance_valid,
                config.camera,
                config.bev,
                config.ray,
                base_bev_rgb=final_bev_rgb,
                base_bev_valid=final_bev_valid,
                base_top_occupancy=final_observed_top,
                base_top_probability=final_top_probability,
                floor_z=bev.floor_z,
            )
            final_bev_rgb = dense.bev_rgb
            final_bev_valid = dense.bev_valid
            if dense.observed_top_occupancy is not None:
                final_observed_top = dense.observed_top_occupancy
            if dense.top_probability_map is not None:
                final_top_probability = dense.top_probability_map
            final_support_occupancy = dense.observed_support_occupancy
            combined_density = combined_density + dense.added_density
            np.save(run_dir / "dense_added_ray_density.npy", dense.added_density)
            save_heatmap(run_dir / "dense_added_ray_density.png", dense.added_density)
            metrics.update(dense.metrics)
        else:
            metrics["dense_coverage_enabled"] = False
            final_support_occupancy = np.where(final_bev_valid > 0, 255, 0).astype(np.uint8)

        newly_covered = np.where((final_bev_valid > 0) & (bev_valid_before == 0), 255, 0).astype(np.uint8)
        np.save(run_dir / "added_ray_density.npy", combined_density)
        save_heatmap(run_dir / "added_ray_density.png", combined_density)
        Image.fromarray(bev_valid_before).save(run_dir / "bev_valid_before.png")
        Image.fromarray(final_bev_valid).save(run_dir / "bev_valid_after.png")
        Image.fromarray(newly_covered).save(run_dir / "newly_covered_bev_cells.png")
        Image.fromarray(final_bev_rgb).save(run_dir / "bev_rgb.png")
        Image.fromarray(final_bev_valid).save(run_dir / "bev_valid.png")
        Image.fromarray(255 - final_observed_top).save(run_dir / "observed_top_occupancy.png")
        np.save(run_dir / "observed_top_occupancy.npy", final_observed_top)
        Image.fromarray(255 - final_support_occupancy).save(run_dir / "observed_support_occupancy.png")
        np.save(run_dir / "observed_support_occupancy.npy", final_support_occupancy)
        Image.fromarray(final_top_probability).save(run_dir / "top_probability_map.png")
        metrics.update(_density_region_metrics(combined_density))
        metrics["bev_unique_cells_before"] = int(np.count_nonzero(bev_valid_before))
        metrics["bev_unique_cells_after_adaptive"] = int(np.count_nonzero(bev_valid_after_adaptive))
        metrics["bev_unique_cells_after"] = int(np.count_nonzero(final_bev_valid))
        metrics["bev_newly_covered_cells"] = int(np.count_nonzero(newly_covered))
        metrics["bev_valid_cells_with_final_normals"] = int(np.count_nonzero(bev.bev_valid))
        metrics["observed_top_cells"] = int(np.count_nonzero(final_observed_top))
        metrics["observed_support_cells"] = int(np.count_nonzero(final_support_occupancy))
        metrics.update(bev.metadata)

    metadata = {
        "pipeline": "wide_fov_supervision_v2 D0-guided 3D/BEV adaptive ray + ray-aware refiner",
        "depth_source": depth_source,
        "depth_source_semantics": (
            "external source-camera z-depth in metres"
            if external_depth is not None
            else "Depth Anything V2 metric z-depth"
        ),
        "gt_evaluation_is_independent": not gt_is_external_input,
        "ray_generation_semantics": "camera geometry deterministically creates ray directions; the learned refiner predicts only query z-depth",
        "dense_coverage_semantics": "dense source-cell rays are streamed directly into BEV with the selected D0 depth source; they are not stored in ray_queries.npz",
        "observed_support_occupancy_semantics": "black PNG pixels mean any observed BEV support after source, adaptive, and dense rays; this is coverage, not top-facing occupancy",
        "planned_after_semantics": "planned-after maps show the requested subdivision before the query budget; selected cells are identified by added_ray_density",
        "front_hemisphere_semantics": "180-degree candidates are coverage-only and are never passed to the refiner, loss, 3D point cloud, or BEV",
        "occupancy_semantics": "observed_top_occupancy is not a classic free/occupied map; black PNG means observed top-facing non-floor surface",
        "branches": [prediction.branch for prediction in predictions],
        "backbone_metadata": {prediction.branch: prediction.metadata for prediction in predictions},
        "config": config_to_dict(config),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if config.toggles.enable_html:
        generate_dashboard(
            run_dir,
            metadata,
            metrics,
            [
                "source_rgb.png",
                "ray_gap_before.png",
                "ray_gap_after.png",
                "surface_gap_before_m.png",
                "surface_gap_planned_after_m.png",
                "bev_gap_before_cells.png",
                "bev_gap_planned_after_cells.png",
                "sampling_priority.png",
                "sampling_eligible.png",
                "planned_added_ray_density.png",
                "adaptive_added_ray_density.png",
                "dense_added_ray_density.png",
                "added_ray_density.png",
                "front_hemisphere_coverage.png",
                f"{primary.branch}/depth0_z.png",
                f"{primary.branch}/normal0.png",
                "depth_gt_absrel_error.png",
                "bev_valid_before.png",
                "bev_valid_after.png",
                "newly_covered_bev_cells.png",
                "bev_rgb.png",
                "observed_top_occupancy.png",
                "observed_support_occupancy.png",
                "top_probability_map.png",
            ],
        )
    return run_dir
