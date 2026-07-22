from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

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
    spherical_bilerp,
)
from wide_fov_supervision_v2.modules.bev_mapping import build_bev_outputs, build_bev_valid
from wide_fov_supervision_v2.modules.camera_geometry import (
    build_fisheye_rays,
    camera_to_world_points,
    cell_angular_gap,
    cv_rays_to_world,
    points_from_z_depth,
)
from wide_fov_supervision_v2.modules.quad_completion.model import QuadRayCompletionModel
from wide_fov_supervision_v2.modules.query_geometry import dense_normals_from_depth, normals_from_stencil_depths
from wide_fov_supervision_v2.modules.surface_rasterization import FloorSurfaceRasterResult, rasterize_floor_surfaces
from wide_fov_supervision_v2.modules.visualization import save_coverage, save_depth, save_heatmap, save_mask, save_normal, save_rgb
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(config: PipelineConfig, mode: str) -> Path:
    root = config.paths.outputs / ("inference" if mode == "infer" else mode)
    run_dir = root / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_prediction(run_dir: Path, prediction: BackbonePrediction) -> None:
    branch_dir = run_dir / prediction.branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    np.save(branch_dir / "depth0_z.npy", prediction.depth0_z.astype(np.float32))
    np.save(branch_dir / "normal0.npy", prediction.normal0.astype(np.float32))
    np.save(branch_dir / "valid.npy", prediction.valid.astype(bool))
    save_depth(branch_dir / "depth0_z.png", prediction.depth0_z)
    save_normal(branch_dir / "normal0.png", prediction.normal0)


def _select_final_query_depth(
    model_depth_z: np.ndarray,
    base_depth_z: np.ndarray,
    source_cell_continuous: np.ndarray,
    *,
    use_base_for_continuous: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """최종 query z-depth를 선택한다.

    Convex-quad completion 모델은 ``D = base * exp(delta)`` 형태로 residual을
    예측한다. 랙/물체 경계처럼 source 2x2 corner가 불연속인 cell에서는 이 residual이
    필요하지만, 바닥처럼 네 corner가 이미 같은 연속 표면을 보는 cell에서는 residual이
    오히려 metric depth를 밀어 BEV 바닥 coverage를 깨뜨릴 수 있다.

    따라서 ``source_cell_continuous``인 query는 유효한 bilinear ``base_depth_z``를
    최종 depth로 사용하고, 나머지 edge/discontinuous query만 모델 depth를 사용한다.
    반환되는 bool mask는 실제로 base depth가 적용된 query 위치다.
    """

    final_depth = np.asarray(model_depth_z, dtype=np.float32).copy()
    if not use_base_for_continuous:
        return final_depth, np.zeros(final_depth.shape, dtype=bool)
    base_depth = np.asarray(base_depth_z, dtype=np.float32)
    base_mask = (
        np.asarray(source_cell_continuous, dtype=bool)
        & np.isfinite(base_depth)
        & (base_depth > 0.0)
    )
    final_depth[base_mask] = base_depth[base_mask]
    return final_depth, base_mask


def _save_adaptive_diagnostics(run_dir: Path, result: AdaptiveRayResult) -> None:
    arrays = {
        "ray_gap_before": result.angular_gap_before,
        "ray_gap_after": result.angular_gap_planned_after,
        "surface_gap_before_m": result.surface_gap_before_m,
        "surface_gap_planned_after_m": result.surface_gap_planned_after_m,
        "bev_gap_before_cells": result.bev_gap_before_cells,
        "bev_gap_planned_after_cells": result.bev_gap_planned_after_cells,
        "sampling_priority": result.sampling_priority,
        "added_ray_density": result.added_density,
    }
    for name, array in arrays.items():
        np.save(run_dir / f"{name}.npy", np.asarray(array))
    for before_name, after_name in (
        ("ray_gap_before", "ray_gap_after"),
        ("surface_gap_before_m", "surface_gap_planned_after_m"),
        ("bev_gap_before_cells", "bev_gap_planned_after_cells"),
    ):
        finite = arrays[before_name][np.isfinite(arrays[before_name])]
        maximum = float(np.percentile(finite, 99.0)) if len(finite) else 1.0
        save_heatmap(run_dir / f"{before_name}.png", arrays[before_name], value_min=0.0, value_max=maximum)
        save_heatmap(run_dir / f"{after_name}.png", arrays[after_name], value_min=0.0, value_max=maximum)
    save_heatmap(run_dir / "sampling_priority.png", result.sampling_priority)
    save_heatmap(run_dir / "added_ray_density.png", result.added_density)
    save_mask(run_dir / "sampling_eligible.png", result.eligible_mask)
    np.save(run_dir / "sampling_eligible.npy", result.eligible_mask)


def _query_support(
    queries: RayQuerySet,
    rgb: np.ndarray,
    depth_z: np.ndarray,
    rays: np.ndarray,
    source_valid: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """각 query parent cell의 실제 2x2 fisheye support를 고정 순서로 꺼낸다."""

    parent = queries.parent_cell[indices]
    y, x = parent[:, 0], parent[:, 1]
    support_rgb = np.stack([rgb[y, x], rgb[y, x + 1], rgb[y + 1, x + 1], rgb[y + 1, x]], axis=1)
    support_depth = np.stack(
        [depth_z[y, x], depth_z[y, x + 1], depth_z[y + 1, x + 1], depth_z[y + 1, x]], axis=1
    )
    support_rays = np.stack(
        [rays[y, x], rays[y, x + 1], rays[y + 1, x + 1], rays[y + 1, x]], axis=1
    )
    support_valid = np.stack(
        [source_valid[y, x], source_valid[y, x + 1], source_valid[y + 1, x + 1], source_valid[y + 1, x]],
        axis=1,
    )
    return (
        support_rays.astype(np.float32),
        support_rgb.astype(np.float32) / 255.0,
        support_depth.astype(np.float32),
        support_valid.astype(bool),
    )


def _stencil_rays_from_parent(queries: RayQuerySet, source_rays: np.ndarray, indices: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    parent = queries.parent_cell[indices]
    y, x = parent[:, 0], parent[:, 1]
    rel = queries.relative_uv[indices]
    offsets = np.array([[-step, 0.0], [step, 0.0], [0.0, -step], [0.0, step]], dtype=np.float32)
    stencil_rel = np.clip(rel[:, None, :] + offsets[None], 0.0, 1.0)
    r00, r10 = source_rays[y, x], source_rays[y, x + 1]
    r11, r01 = source_rays[y + 1, x + 1], source_rays[y + 1, x]
    stencil_rays = spherical_bilerp(
        r00[:, None],
        r10[:, None],
        r01[:, None],
        r11[:, None],
        stencil_rel[..., 0],
        stencil_rel[..., 1],
    )
    return stencil_rays.astype(np.float32), stencil_rel.astype(np.float32)


def run_completion_on_queries(
    config: PipelineConfig,
    prediction: BackbonePrediction,
    rgb: np.ndarray,
    source_rays: np.ndarray,
    lens_valid: np.ndarray,
    queries: RayQuerySet,
) -> dict[str, np.ndarray | bool]:
    """각 adaptive query를 parent cell의 네 support만 사용해 단일 pass로 복원한다."""

    count = len(queries)
    output = {
        "rgb": np.zeros((count, 3), dtype=np.float32),
        "depth_z": np.full(count, np.nan, dtype=np.float32),
        "model_depth_z": np.full(count, np.nan, dtype=np.float32),
        "base_rgb": np.zeros((count, 3), dtype=np.float32),
        "base_depth_z": np.full(count, np.nan, dtype=np.float32),
        "base_depth_applied": np.zeros(count, dtype=bool),
        "valid_probability": np.zeros(count, dtype=np.float32),
        "confidence_probability": np.zeros(count, dtype=np.float32),
        "delta_log_depth": np.zeros(count, dtype=np.float32),
        "normal": np.full((count, 3), np.nan, dtype=np.float32),
    }
    if count == 0:
        output["checkpoint_loaded"] = False
        return output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuadRayCompletionModel(config.completion).to(device).eval()
    checkpoint_loaded = False
    if config.toggles.enable_completion and config.paths.checkpoint is not None:
        checkpoint_path = Path(config.paths.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Completion checkpoint가 없습니다: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, map_location=device)
        checkpoint_loaded = True
    # DSINE normal 유효성은 completion 입력이나 query 선택에 사용하지 않는다.
    source_valid = lens_valid & np.isfinite(prediction.depth0_z) & (prediction.depth0_z > 0.0)
    batch_size = min(4096, max(1, int(config.ray.query_chunk_size)))
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            indices = np.arange(start, stop)
            support_rays, support_rgb, support_depth, support_valid = _query_support(
                queries, rgb, prediction.depth0_z, source_rays, source_valid, indices
            )
            stencil_rays, stencil_rel = _stencil_rays_from_parent(
                queries, source_rays, indices, config.ray.normal_stencil_relative_step
            )
            to_tensor = lambda value: torch.from_numpy(value).to(device)
            support_rays_t = to_tensor(support_rays)
            support_rgb_t = to_tensor(support_rgb)
            support_depth_t = to_tensor(support_depth)
            support_valid_t = to_tensor(support_valid)
            query_rays_t = to_tensor(queries.ray_dir[indices, None].astype(np.float32))
            query_rel_t = to_tensor(queries.relative_uv[indices, None].astype(np.float32))
            query_mask_t = torch.ones((len(indices), 1), dtype=torch.bool, device=device)
            center = model(
                support_rays_t,
                support_rgb_t,
                support_depth_t,
                support_valid_t,
                query_rays_t,
                query_rel_t,
                query_mask_t,
            )
            stencil = model(
                support_rays_t,
                support_rgb_t,
                support_depth_t,
                support_valid_t,
                to_tensor(stencil_rays),
                to_tensor(stencil_rel),
                torch.ones((len(indices), 4), dtype=torch.bool, device=device),
            )
            continuous_t = torch.from_numpy(queries.source_cell_continuous[indices]).to(device=device)
            center_model_depth = center.depth_z[:, 0]
            center_base_depth = center.base_depth_z[:, 0]
            stencil_model_depth = stencil.depth_z
            stencil_base_depth = stencil.base_depth_z
            if config.completion.use_base_depth_for_continuous_queries:
                center_base_valid = torch.isfinite(center_base_depth) & (center_base_depth > 0.0)
                stencil_base_valid = torch.isfinite(stencil_base_depth).all(dim=-1) & (stencil_base_depth > 0.0).all(dim=-1)
                base_depth_t = continuous_t & center_base_valid & stencil_base_valid
                center_depth = torch.where(base_depth_t, center_base_depth, center_model_depth)
                stencil_depth = torch.where(base_depth_t[:, None], stencil_base_depth, stencil_model_depth)
            else:
                base_depth_t = torch.zeros_like(continuous_t, dtype=torch.bool)
                center_depth = center_model_depth
                stencil_depth = stencil_model_depth
            normal, normal_valid = normals_from_stencil_depths(
                center_depth[:, None],
                stencil_depth[:, None, :],
                query_rays_t,
                to_tensor(stencil_rays[:, None]),
                z_eps=config.camera.geometry_z_eps,
            )
            output["rgb"][indices] = center.rgb[:, 0].cpu().numpy()
            output["depth_z"][indices] = center_depth.cpu().numpy()
            output["model_depth_z"][indices] = center_model_depth.cpu().numpy()
            output["base_rgb"][indices] = center.base_rgb[:, 0].cpu().numpy()
            output["base_depth_z"][indices] = center.base_depth_z[:, 0].cpu().numpy()
            output["base_depth_applied"][indices] = base_depth_t.cpu().numpy()
            output["delta_log_depth"][indices] = center.delta_log_depth[:, 0].cpu().numpy()
            if checkpoint_loaded:
                output["valid_probability"][indices] = torch.sigmoid(center.valid_logit[:, 0]).cpu().numpy()
                output["confidence_probability"][indices] = torch.sigmoid(center.confidence_logit[:, 0]).cpu().numpy()
            else:
                # 학습 전 baseline은 valid 2x2 cell을 허용하되 edge confidence는 보수적으로 둔다.
                output["valid_probability"][indices] = support_valid.all(axis=1).astype(np.float32)
                output["confidence_probability"][indices] = queries.source_cell_continuous[indices].astype(np.float32)
            normal_np = normal[:, 0].cpu().numpy().astype(np.float32)
            normal_np[~normal_valid[:, 0].cpu().numpy()] = np.nan
            output["normal"][indices] = normal_np
    output["checkpoint_loaded"] = checkpoint_loaded
    return output


def _dense_source_geometry(
    config: PipelineConfig,
    prediction: BackbonePrediction,
    rgb: np.ndarray,
    rays: np.ndarray,
    lens_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = lens_valid & np.isfinite(prediction.depth0_z) & (prediction.depth0_z > 0.0)
    camera_points, _, point_valid = points_from_z_depth(
        prediction.depth0_z, rays, z_eps=config.camera.geometry_z_eps
    )
    valid &= point_valid
    world_points = camera_to_world_points(camera_points, config.camera).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        depth_t = torch.from_numpy(np.nan_to_num(prediction.depth0_z).astype(np.float32))[None, None].to(device)
        rays_t = torch.from_numpy(rays.astype(np.float32)).permute(2, 0, 1)[None].to(device)
        valid_t = torch.from_numpy(valid)[None, None].to(device)
        normal_t, normal_valid_t = dense_normals_from_depth(
            depth_t, rays_t, valid_t, z_eps=config.camera.geometry_z_eps
        )
    normal = normal_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    normal_valid = normal_valid_t[0, 0].cpu().numpy()
    return {
        "camera_points": camera_points,
        "world_points": world_points,
        "rgb": rgb,
        "normal": normal,
        "valid": valid,
        "normal_valid": normal_valid,
    }


def _normals_to_world(normals_cv: np.ndarray, config: PipelineConfig) -> np.ndarray:
    world = cv_rays_to_world(normals_cv, np.asarray(config.camera.world_from_camera, dtype=np.float64))
    norm = np.linalg.norm(world, axis=-1, keepdims=True)
    world = world / np.clip(norm, 1.0e-6, None)
    world[~np.isfinite(norm[..., 0])] = np.nan
    return world.astype(np.float32)


def _query_geometry(config: PipelineConfig, queries: RayQuerySet, depth_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points, _, valid = points_from_z_depth(depth_z, queries.ray_dir, z_eps=config.camera.geometry_z_eps)
    valid &= queries.observed & ~queries.unknown
    world_points = camera_to_world_points(camera_points, config.camera).astype(np.float32)
    camera_points[~valid] = np.nan
    world_points[~valid] = np.nan
    return camera_points, world_points, valid


def _save_completion_previews(
    run_dir: Path,
    rgb: np.ndarray,
    queries: RayQuerySet,
    completion: dict[str, np.ndarray | bool],
) -> None:
    h, w = rgb.shape[:2]
    xy = np.rint(queries.source_uv - 0.5).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    sampling = rgb.copy()
    continuous = queries.source_cell_continuous & inside
    edge = ~queries.source_cell_continuous & inside
    sampling[xy[continuous, 1], xy[continuous, 0]] = np.array([30, 220, 80], dtype=np.uint8)
    sampling[xy[edge, 1], xy[edge, 0]] = np.array([240, 70, 60], dtype=np.uint8)
    save_rgb(run_dir / "quad_sampling_preview.png", sampling)

    rgb_preview = np.zeros_like(rgb)
    pred_rgb = np.clip(np.asarray(completion["rgb"]) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_preview[xy[inside, 1], xy[inside, 0]] = pred_rgb[inside]
    save_rgb(run_dir / "completion_rgb_preview.png", rgb_preview)
    depth_preview = np.full((h, w), np.nan, dtype=np.float32)
    confidence_map = np.full((h, w), np.nan, dtype=np.float32)
    depth_preview[xy[inside, 1], xy[inside, 0]] = np.asarray(completion["depth_z"])[inside]
    confidence_map[xy[inside, 1], xy[inside, 0]] = np.asarray(completion["confidence_probability"])[inside]
    save_depth(run_dir / "completion_depth_preview.png", depth_preview)
    save_heatmap(run_dir / "confidence_map.png", confidence_map, value_min=0.0, value_max=1.0)


def _save_bev_branch(
    run_dir: Path,
    name: str,
    config: PipelineConfig,
    source: dict[str, np.ndarray],
    query_camera: np.ndarray,
    query_world: np.ndarray,
    query_rgb: np.ndarray,
    query_normal_cv: np.ndarray,
    query_mask: np.ndarray,
    bev_valid_before: np.ndarray,
    floor_surface: FloorSurfaceRasterResult | None,
) -> dict[str, int]:
    branch_dir = run_dir / name
    branch_dir.mkdir(parents=True, exist_ok=True)
    query_valid = query_mask & np.isfinite(query_world).all(axis=-1)
    np.save(branch_dir / "query_points_camera.npy", query_camera[query_valid].astype(np.float32))
    np.save(branch_dir / "query_points_world.npy", query_world[query_valid].astype(np.float32))
    np.save(branch_dir / "query_points_rgb.npy", query_rgb[query_valid].astype(np.uint8))

    source_valid = source["valid"]
    source_world = source["world_points"][source_valid]
    source_rgb = source["rgb"][source_valid]
    source_normal_cv = source["normal"][source_valid]
    world_points = np.concatenate([source_world, query_world[query_valid]], axis=0)
    colors = np.concatenate([source_rgb, query_rgb[query_valid]], axis=0)
    normals_cv = np.concatenate([source_normal_cv, query_normal_cv[query_valid]], axis=0)
    normals_world = _normals_to_world(normals_cv, config)
    all_valid = np.ones(len(world_points), dtype=bool)
    normal_valid = np.isfinite(normals_world).all(axis=-1)
    bev = build_bev_outputs(colors, world_points, normals_world, all_valid, config.bev, normal_valid_mask=normal_valid)
    final_rgb = bev.bev_rgb
    surface_newly_count = 0
    surface_cell_count = 0
    if floor_surface is not None:
        final_rgb = floor_surface.bev_rgb.copy()
        point_mask = bev.bev_rgb[..., 3] > 0
        final_rgb[point_mask] = bev.bev_rgb[point_mask]
        surface_newly = np.where((floor_surface.bev_valid > 0) & (bev_valid_before == 0), 255, 0).astype(np.uint8)
        Image.fromarray(floor_surface.bev_rgb).save(branch_dir / "floor_surface_rgb.png")
        Image.fromarray(floor_surface.bev_valid).save(branch_dir / "floor_surface_valid.png")
        Image.fromarray(surface_newly).save(branch_dir / "floor_surface_newly_covered_bev_cells.png")
        surface_newly_count = int(np.count_nonzero(surface_newly))
        surface_cell_count = int(np.count_nonzero(floor_surface.bev_valid))
    bev_valid = np.maximum(bev_valid_before, bev.bev_valid)
    if floor_surface is not None:
        bev_valid = np.maximum(bev_valid, floor_surface.bev_valid)
    newly = np.where((bev_valid > 0) & (bev_valid_before == 0), 255, 0).astype(np.uint8)
    Image.fromarray(final_rgb).save(branch_dir / "bev_rgb.png")
    Image.fromarray(bev_valid).save(branch_dir / "bev_valid.png")
    Image.fromarray(newly).save(branch_dir / "newly_covered_bev_cells.png")
    Image.fromarray(255 - bev.observed_top_occupancy).save(branch_dir / "observed_top_occupancy.png")
    np.save(branch_dir / "observed_top_occupancy.npy", bev.observed_top_occupancy)
    Image.fromarray(bev.top_probability_map).save(branch_dir / "top_probability_map.png")
    return {
        f"{name}_query_count": int(query_valid.sum()),
        f"{name}_bev_unique_cells": int(np.count_nonzero(bev_valid)),
        f"{name}_newly_covered_bev_cells": int(np.count_nonzero(newly)),
        f"{name}_floor_surface_fill_cells": surface_cell_count,
        f"{name}_floor_surface_newly_covered_bev_cells": surface_newly_count,
    }


def _finite_stats(prefix: str, values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    samples = values[valid]
    if len(samples) == 0:
        return {f"{prefix}_count": 0}
    p50, p90, p95, p99 = np.percentile(samples, [50, 90, 95, 99])
    return {
        f"{prefix}_count": int(len(samples)),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p90": float(p90),
        f"{prefix}_p95": float(p95),
        f"{prefix}_p99": float(p99),
        f"{prefix}_max": float(np.max(samples)),
    }


def _sample_map_at_query(value: np.ndarray, source_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center query 좌표에서 scalar/vector map을 strict bilinear sampling한다."""

    array = np.asarray(value)
    x = source_uv[:, 0] - 0.5
    y = source_uv[:, 1] - 0.5
    h, w = array.shape[:2]
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= w - 1.0) & (y >= 0.0) & (y <= h - 1.0)
    x0 = np.floor(np.clip(x, 0.0, w - 1.0)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0.0, h - 1.0)).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    v00, v10 = array[y0, x0], array[y0, x1]
    v01, v11 = array[y1, x0], array[y1, x1]
    if array.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
        finite = np.isfinite(v00).all(axis=-1) & np.isfinite(v10).all(axis=-1)
        finite &= np.isfinite(v01).all(axis=-1) & np.isfinite(v11).all(axis=-1)
    else:
        finite = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
        finite &= (v00 > 0.0) & (v10 > 0.0) & (v01 > 0.0) & (v11 > 0.0)
    sampled = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    ).astype(np.float32)
    return sampled, inside & finite


def _depth_error_metrics(prefix: str, prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    mask = valid & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0.0) & (target > 0.0)
    if not np.any(mask):
        return {f"{prefix}_count": 0}
    error = prediction[mask] - target[mask]
    return {
        f"{prefix}_count": int(mask.sum()),
        f"{prefix}_abs_rel": float(np.mean(np.abs(error) / target[mask].clip(min=1.0e-6))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
    }


def _evaluate_matching_isaac_gt(
    config: PipelineConfig,
    run_dir: Path,
    rgb_hash: str,
    external_depth: ExternalDepthPrediction | None,
    primary: BackbonePrediction,
    rgb: np.ndarray,
    rays: np.ndarray,
    queries: RayQuerySet,
    completion: dict[str, np.ndarray | bool],
) -> dict[str, object]:
    """현재 RGB hash와 일치하는 Isaac GT를 모델 입력과 분리해 평가한다."""

    if not config.toggles.enable_gt_evaluation:
        return {"gt_evaluation_skipped": True, "gt_evaluation_skipped_reason": "disabled"}
    if config.eval.use_isaac_gt_only_on_hash_match and rgb_hash.lower() != config.eval.input_rgb_sha256.lower():
        return {"gt_evaluation_skipped": True, "gt_evaluation_skipped_reason": "RGB hash mismatch"}
    gt_path = config.paths.isaac_reference_run / "depth_z.npy"
    if not gt_path.exists():
        return {"gt_evaluation_skipped": True, "gt_evaluation_skipped_reason": "GT depth not found"}
    if external_depth is not None and sha256_file(gt_path) == external_depth.metadata["external_depth_sha256"]:
        return {
            "gt_evaluation_skipped": True,
            "gt_evaluation_skipped_reason": "external depth input is byte-identical to Isaac GT",
        }
    gt_depth = np.load(gt_path).astype(np.float32)
    gt_valid = rays[..., 2] > config.camera.geometry_z_eps
    gt_valid &= np.isfinite(gt_depth) & (gt_depth > 0.0)
    dense_error = np.full_like(gt_depth, np.nan)
    dense_mask = gt_valid & np.isfinite(primary.depth0_z) & (primary.depth0_z > 0.0)
    dense_error[dense_mask] = np.abs(primary.depth0_z[dense_mask] - gt_depth[dense_mask]) / gt_depth[dense_mask]
    np.save(run_dir / "depth_gt_absrel_error.npy", dense_error)
    save_heatmap(run_dir / "depth_gt_absrel_error.png", dense_error)

    query_gt, query_gt_valid = _sample_map_at_query(gt_depth, queries.source_uv)
    query_metrics = _depth_error_metrics(
        "query_bilinear", np.asarray(completion["base_depth_z"]), query_gt, query_gt_valid
    )
    query_metrics.update(
        _depth_error_metrics("query_completion", np.asarray(completion["depth_z"]), query_gt, query_gt_valid)
    )
    query_error = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    xy = np.rint(queries.source_uv - 0.5).astype(np.int64)
    valid_xy = query_gt_valid & (xy[:, 0] >= 0) & (xy[:, 0] < rgb.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < rgb.shape[0])
    relative_error = np.abs(np.asarray(completion["depth_z"]) - query_gt) / query_gt.clip(min=1.0e-6)
    query_error[xy[valid_xy, 1], xy[valid_xy, 0]] = relative_error[valid_xy]
    np.save(run_dir / "completion_depth_gt_absrel_error.npy", query_error)
    save_heatmap(run_dir / "completion_depth_gt_absrel_error.png", query_error)

    gt_prediction = BackbonePrediction(
        branch="gt",
        depth0_z=gt_depth,
        normal0=np.zeros((*gt_depth.shape, 3), dtype=np.float32),
        valid=gt_valid,
        metadata={},
    )
    gt_geometry = _dense_source_geometry(config, gt_prediction, rgb, rays, gt_valid)
    query_gt_normal, normal_sample_valid = _sample_map_at_query(gt_geometry["normal"], queries.source_uv)
    query_gt_normal /= np.linalg.norm(query_gt_normal, axis=-1, keepdims=True).clip(min=1.0e-6)
    pred_normal = np.asarray(completion["normal"])
    normal_valid = normal_sample_valid & np.isfinite(pred_normal).all(axis=-1)
    normal_error = np.full(len(queries), np.nan, dtype=np.float32)
    if np.any(normal_valid):
        cosine = np.sum(pred_normal[normal_valid] * query_gt_normal[normal_valid], axis=-1)
        normal_error[normal_valid] = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    normal_error_map = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    normal_xy = valid_xy & normal_valid
    normal_error_map[xy[normal_xy, 1], xy[normal_xy, 0]] = normal_error[normal_xy]
    np.save(run_dir / "normal_gt_angular_error.npy", normal_error_map)
    save_heatmap(run_dir / "normal_gt_angular_error.png", normal_error_map, value_min=0.0, value_max=90.0)
    query_metrics.update(_depth_error_metrics("source_d0", primary.depth0_z, gt_depth, dense_mask))
    query_metrics["query_normal_gt_count"] = int(normal_valid.sum())
    query_metrics["query_normal_gt_mean_degrees"] = float(np.nanmean(normal_error)) if np.any(normal_valid) else float("nan")
    query_metrics["gt_evaluation_skipped"] = False
    return query_metrics


def validate_environment(config: PipelineConfig | None = None) -> dict:
    config = config or make_default_config()
    rays = build_fisheye_rays(config.camera)
    gap, _ = cell_angular_gap(rays.rays_cv, rays.valid)
    checks = {
        "depth_source": config.backbone.depth_source,
        "input_rgb_exists": config.paths.input_rgb.exists(),
        "nyu_mat_exists": config.paths.nyu_mat.exists(),
        "depth_anything_root_exists": config.paths.depth_anything_root.exists(),
        "depth_anything_checkpoint_exists": config.paths.depth_anything_vitl_ckpt.exists(),
        "dsine_root_exists": config.paths.dsine_root.exists(),
        "dsine_checkpoint_exists": config.paths.dsine_ckpt.exists(),
        "fisheye_valid_pixels": int(rays.valid.sum()),
        "fisheye_roundtrip_max_error_px": rays.max_roundtrip_error_px,
        "gap_before_max_rad": float(np.nanmax(gap)),
    }
    if config.backbone.depth_source == "external_npy":
        try:
            external = load_external_z_depth(config.paths.external_depth_z, (config.camera.height, config.camera.width))
            checks["external_depth_valid_pixels"] = int(external.valid.sum())
            checks["external_depth_shape_valid"] = True
        except (FileNotFoundError, ValueError) as error:
            checks["external_depth_shape_valid"] = False
            checks["external_depth_error"] = str(error)
    return checks


def run_inference(config: PipelineConfig | None = None) -> Path:
    """현재 fisheye RGB에 Convex Quad RGB-D ray completion과 두 BEV branch를 실행한다."""

    config = config or make_default_config()
    ensure_output_roots(config)
    run_dir = create_run_dir(config, "infer")
    rgb = load_rgb(config.paths.input_rgb)
    if rgb.shape[:2] != (config.camera.height, config.camera.width):
        raise ValueError(f"입력 RGB shape {rgb.shape[:2]}가 camera 설정과 다릅니다.")
    rgb_hash = sha256_file(config.paths.input_rgb)
    save_rgb(run_dir / "source_rgb.png", rgb)
    shutil.copy2(config.paths.input_rgb, run_dir / "input_rgb_original.png")
    save_config(config, run_dir / "config.json")
    rays = build_fisheye_rays(config.camera)
    np.save(run_dir / "source_rays.npy", rays.rays_cv.astype(np.float32))

    external_depth: ExternalDepthPrediction | None = None
    if config.backbone.depth_source == "external_npy":
        external_depth = load_external_z_depth(config.paths.external_depth_z, rgb.shape[:2])
    elif config.backbone.depth_source != "da_v2":
        raise ValueError("depth_source는 'da_v2' 또는 'external_npy'여야 합니다.")
    runner = BackboneRunner(config.paths, config.backbone, config.camera)
    depth_override = None if external_depth is None else external_depth.depth_z
    depth_metadata = None if external_depth is None else external_depth.metadata
    predictions: list[BackbonePrediction] = []
    if config.toggles.enable_direct_backbone:
        predictions.append(runner.run_direct(rgb, depth_override_z=depth_override, depth_metadata=depth_metadata))
    if config.toggles.enable_tangent_backbone:
        predictions.append(
            runner.run_tangent(
                rgb, rays.rays_cv, depth_override_z=depth_override, depth_metadata=depth_metadata
            )
        )
    if not predictions:
        raise RuntimeError("direct 또는 tangent backbone을 하나 이상 켜야 합니다.")
    for prediction in predictions:
        _save_prediction(run_dir, prediction)
    primary = next((prediction for prediction in predictions if prediction.branch == "tangent"), predictions[0])
    guidance_valid = rays.valid & np.isfinite(primary.depth0_z) & (primary.depth0_z > 0.0)
    adaptive = generate_guided_observed_queries(
        rays.rays_cv,
        guidance_valid,
        primary.depth0_z,
        config.camera,
        config.bev,
        config.ray,
        mode="surface_bev",
    )
    queries = adaptive.queries if config.toggles.enable_adaptive_ray_generation else adaptive.queries.subset(np.zeros(len(adaptive.queries), dtype=bool))
    _save_adaptive_diagnostics(run_dir, adaptive)
    np.save(run_dir / "source_cell_continuous.npy", queries.source_cell_continuous)

    coverage = np.zeros((config.camera.height, config.camera.width), dtype=np.uint8)
    hemisphere_count = hemisphere_unknown = 0
    if config.toggles.enable_front_hemisphere_queries:
        hemisphere, coverage = generate_front_hemisphere_queries(config.camera, rays.valid, None, config.ray)
        hemisphere_count = len(hemisphere)
        hemisphere_unknown = int(hemisphere.unknown.sum())
        hemisphere.save_npz(
            run_dir / "front_hemisphere_queries.npz",
            depth_z=np.full(len(hemisphere), np.nan, dtype=np.float32),
            rgb=np.full((len(hemisphere), 3), np.nan, dtype=np.float32),
            confidence=np.full(len(hemisphere), np.nan, dtype=np.float32),
        )
    save_coverage(run_dir / "front_hemisphere_coverage.png", coverage)

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    branch_outputs: dict[str, dict[str, np.ndarray | bool]] = {}
    for prediction in predictions:
        completion = run_completion_on_queries(config, prediction, rgb, rays.rays_cv, rays.valid, queries)
        branch_outputs[prediction.branch] = completion
        branch_dir = run_dir / prediction.branch
        np.save(branch_dir / "query_rgb_pred.npy", completion["rgb"])
        np.save(branch_dir / "query_depth_pred_z.npy", completion["depth_z"])
        np.save(branch_dir / "query_depth_model_z.npy", completion["model_depth_z"])
        np.save(branch_dir / "query_valid_probability.npy", completion["valid_probability"])
        np.save(branch_dir / "query_confidence_probability.npy", completion["confidence_probability"])
        np.save(branch_dir / "query_normal_final.npy", completion["normal"])
    completion = branch_outputs[primary.branch]
    for filename, key in (
        ("query_rgb_pred.npy", "rgb"),
        ("query_depth_pred_z.npy", "depth_z"),
        ("query_depth_model_z.npy", "model_depth_z"),
        ("query_valid_probability.npy", "valid_probability"),
        ("query_confidence_probability.npy", "confidence_probability"),
    ):
        np.save(run_dir / filename, completion[key])
    _save_completion_previews(run_dir, rgb, queries, completion)

    query_camera, query_world, query_depth_valid = _query_geometry(config, queries, completion["depth_z"])
    query_rgb = np.clip(np.asarray(completion["rgb"]) * 255.0, 0.0, 255.0).astype(np.uint8)
    valid_probability = np.asarray(completion["valid_probability"])
    confidence_probability = np.asarray(completion["confidence_probability"])
    probability_mask = (
        query_depth_valid
        & (valid_probability >= config.completion.valid_probability_threshold)
        & (confidence_probability >= config.completion.confidence_probability_threshold)
    )
    continuous_mask = probability_mask & queries.source_cell_continuous
    edge_mask = probability_mask
    if np.any(continuous_mask & ~edge_mask):
        raise AssertionError("continuous_only는 edge_confident의 subset이어야 합니다.")

    queries.save_npz(
        run_dir / "quad_completion_queries.npz",
        rgb_pred=np.asarray(completion["rgb"]),
        depth_base_z=np.asarray(completion["base_depth_z"]),
        depth_model_z=np.asarray(completion["model_depth_z"]),
        depth_pred_z=np.asarray(completion["depth_z"]),
        base_depth_applied=np.asarray(completion["base_depth_applied"]),
        valid_probability=valid_probability,
        confidence_probability=confidence_probability,
        normal_final=np.asarray(completion["normal"]),
        continuous_only=continuous_mask,
        edge_confident=edge_mask,
    )

    source = _dense_source_geometry(config, primary, rgb, rays.rays_cv, rays.valid)
    bev_valid_before = build_bev_valid(source["world_points"], source["valid"], config.bev)
    Image.fromarray(bev_valid_before).save(run_dir / "bev_valid_before.png")
    floor_surface = None
    if config.toggles.enable_floor_surface_rasterization:
        floor_surface = rasterize_floor_surfaces(source["rgb"], source["world_points"], source["valid"], config.bev)
        np.save(run_dir / "floor_surface_source_cell_mask.npy", floor_surface.source_cell_mask)
        save_mask(run_dir / "floor_surface_source_cell_mask.png", floor_surface.source_cell_mask)
    metrics: dict[str, object] = {
        "rgb_sha256": rgb_hash,
        "depth_source": config.backbone.depth_source,
        "primary_branch": primary.branch,
        "adaptive_query_count": len(queries),
        "continuous_source_query_count": int(queries.source_cell_continuous.sum()),
        "edge_source_query_count": int((~queries.source_cell_continuous).sum()),
        "continuous_base_depth_enabled": bool(config.completion.use_base_depth_for_continuous_queries),
        "continuous_base_depth_applied_count": int(np.asarray(completion["base_depth_applied"]).sum()),
        "valid_probability_pass_count": int((valid_probability >= config.completion.valid_probability_threshold).sum()),
        "confidence_probability_pass_count": int((confidence_probability >= config.completion.confidence_probability_threshold).sum()),
        "checkpoint_loaded": bool(completion["checkpoint_loaded"]),
        "checkpoint": str(config.paths.checkpoint) if config.paths.checkpoint is not None else None,
        "candidate_query_count": adaptive.candidate_query_count,
        "query_budget": adaptive.query_budget,
        "query_budget_truncated": adaptive.budget_truncated,
        "hemisphere_query_count": hemisphere_count,
        "hemisphere_unknown_query_count": hemisphere_unknown,
        "bev_unique_cells_before": int(np.count_nonzero(bev_valid_before)),
        "floor_surface_rasterization_enabled": bool(config.toggles.enable_floor_surface_rasterization),
        "floor_surface_source_cell_count": int(floor_surface.source_cell_mask.sum()) if floor_surface is not None else 0,
        "floor_surface_floor_z": float(floor_surface.floor_z) if floor_surface is not None else float("nan"),
    }
    metrics.update(_finite_stats("surface_gap_before_m", adaptive.surface_gap_before_m))
    metrics.update(_finite_stats("surface_gap_after_m", adaptive.surface_gap_planned_after_m))
    metrics.update(_finite_stats("bev_gap_before_cells", adaptive.bev_gap_before_cells))
    metrics.update(_finite_stats("bev_gap_after_cells", adaptive.bev_gap_planned_after_cells))
    if external_depth is not None:
        metrics.update(external_depth.metadata)
    metrics.update(
        _evaluate_matching_isaac_gt(
            config,
            run_dir,
            rgb_hash,
            external_depth,
            primary,
            rgb,
            rays.rays_cv,
            queries,
            completion,
        )
    )

    if config.toggles.enable_bev:
        metrics.update(
            _save_bev_branch(
                run_dir,
                "continuous_only",
                config,
                source,
                query_camera,
                query_world,
                query_rgb,
                np.asarray(completion["normal"]),
                continuous_mask,
                bev_valid_before,
                floor_surface,
            )
        )
        metrics.update(
            _save_bev_branch(
                run_dir,
                "edge_confident",
                config,
                source,
                query_camera,
                query_world,
                query_rgb,
                np.asarray(completion["normal"]),
                edge_mask,
                bev_valid_before,
                floor_surface,
            )
        )
        # 기본 최종 결과는 edge_confident branch다.
        for filename in ("bev_rgb.png", "bev_valid.png", "newly_covered_bev_cells.png", "observed_top_occupancy.png", "top_probability_map.png"):
            shutil.copy2(run_dir / "edge_confident" / filename, run_dir / filename)

    metadata = {
        "pipeline": "convex-quad four-support RGB-D ray completion",
        "ray_generation_semantics": "ray direction/count are deterministic camera geometry; the model restores RGB-D attributes only",
        "quad_assumption": "a wide convex quadrilateral sampled from public RGB-D approximates a sparse adjacent 2x2 fisheye ray cell; it does not recover a real camera pose",
        "scale_semantics": "output depth follows the valid median scale of the four support depths; DA-V2 global scale is not newly identified",
        "depth_policy": "continuous source cells use corner-bilinear base depth for final geometry; discontinuous edge cells use completion model depth",
        "confidence_semantics": "valid means observable RGB-D; confidence means locally continuous enough to insert into 3D/BEV",
        "branch_semantics": {
            "continuous_only": "source cell is depth-continuous and valid/confidence pass thresholds",
            "edge_confident": "all adaptive cells are considered and valid/confidence pass thresholds",
        },
        "front_hemisphere_semantics": "unknown 180-degree rays are coverage-only and never enter completion, point cloud, or BEV",
        "floor_surface_rasterization_semantics": "continuous source cells near the estimated floor height are filled as BEV polygons behind point splats; these floor fills do not enter top-facing non-floor occupancy",
        "occupancy_semantics": "black observed_top_occupancy pixels mean observed top-facing non-floor surface; this is not a classic free/occupied grid",
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
                "quad_sampling_preview.png",
                "completion_rgb_preview.png",
                "completion_depth_preview.png",
                "confidence_map.png",
                "surface_gap_before_m.png",
                "surface_gap_planned_after_m.png",
                "bev_gap_before_cells.png",
                "bev_gap_planned_after_cells.png",
                "sampling_priority.png",
                "sampling_eligible.png",
                "added_ray_density.png",
                "floor_surface_source_cell_mask.png",
                "front_hemisphere_coverage.png",
                f"{primary.branch}/depth0_z.png",
                f"{primary.branch}/normal0.png",
                "depth_gt_absrel_error.png",
                "completion_depth_gt_absrel_error.png",
                "normal_gt_angular_error.png",
                "bev_valid_before.png",
                "continuous_only/bev_rgb.png",
                "continuous_only/bev_valid.png",
                "continuous_only/newly_covered_bev_cells.png",
                "continuous_only/floor_surface_rgb.png",
                "continuous_only/floor_surface_valid.png",
                "continuous_only/floor_surface_newly_covered_bev_cells.png",
                "edge_confident/bev_rgb.png",
                "edge_confident/bev_valid.png",
                "edge_confident/newly_covered_bev_cells.png",
                "edge_confident/floor_surface_rgb.png",
                "edge_confident/floor_surface_valid.png",
                "edge_confident/floor_surface_newly_covered_bev_cells.png",
                "edge_confident/observed_top_occupancy.png",
                "edge_confident/top_probability_map.png",
            ],
        )
    return run_dir
