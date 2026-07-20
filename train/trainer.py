from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wide_fov_supervision_v2.config import PipelineConfig, save_config
from wide_fov_supervision_v2.datasets.nyu.dataset import TeacherCacheDataset
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split
from wide_fov_supervision_v2.loss.partial_loss import PartialLossMasks, PartialSupervisionLoss
from wide_fov_supervision_v2.modules.adaptive_ray import RayQuerySet
from wide_fov_supervision_v2.modules.camera_geometry import project_fisheye_rays
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths
from wide_fov_supervision_v2.modules.refiner import RayAwareQueryRefiner
from wide_fov_supervision_v2.train.checkpoints import save_checkpoint
from wide_fov_supervision_v2.train.query_cache import query_sidecar_root


def _collate_list(batch: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    return batch


def _to_bchw(sample: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.from_numpy(sample["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    depth0 = torch.from_numpy(np.nan_to_num(sample["depth0_z"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)).unsqueeze(0).unsqueeze(0).to(device)
    normal0 = torch.from_numpy(np.nan_to_num(sample["normal0"].astype(np.float32))).permute(2, 0, 1).unsqueeze(0).to(device)
    rays = torch.from_numpy(sample["source_rays"].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    # 모델 입력 유효성은 GT 제공 여부와 분리한다. query 위치 선택과 inference에서는
    # RGB 대응 및 teacher finite 여부만 알 수 있기 때문이다.
    if "source_observed" in sample and "teacher_valid" in sample:
        source_valid = sample["source_observed"].astype(bool) & sample["teacher_valid"].astype(bool)
    else:
        source_valid = sample["source_valid"].astype(bool)
    valid = torch.from_numpy(source_valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt = torch.from_numpy(sample["depth_gt_z"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    return rgb, depth0, normal0, rays, valid, gt


def _query_tensors(query, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rays = torch.from_numpy(query.ray_dir.astype(np.float32)).unsqueeze(0).to(device)
    uv = torch.from_numpy(query.source_uv.astype(np.float32)).unsqueeze(0).to(device)
    rel = torch.from_numpy(np.nan_to_num(query.relative_uv.astype(np.float32))).unsqueeze(0).to(device)
    sampling = torch.from_numpy(np.nan_to_num(query.sampling_features.astype(np.float32))).unsqueeze(0).to(device)
    observed = torch.from_numpy(query.observed.astype(np.float32)).unsqueeze(0).to(device)
    return rays, uv, rel, sampling, observed


def _query_from_sidecar(payload: dict[str, np.ndarray]) -> RayQuerySet:
    """query sidecar의 공통 필드를 ``RayQuerySet``으로 복원한다."""

    return RayQuerySet(
        ray_dir=payload["ray_dir"].astype(np.float32),
        source_uv=payload["source_uv"].astype(np.float32),
        parent_cell=payload["parent_cell"].astype(np.int32),
        relative_uv=payload["relative_uv"].astype(np.float32),
        angular_gap_before=payload["angular_gap_before"].astype(np.float32),
        surface_gap_before_m=payload["surface_gap_before_m"].astype(np.float32),
        bev_gap_before_cells=payload["bev_gap_before_cells"].astype(np.float32),
        sampling_score=payload["sampling_score"].astype(np.float32),
        subdivision_u=payload["subdivision_u"].astype(np.int16),
        subdivision_v=payload["subdivision_v"].astype(np.int16),
        sampling_features=payload["sampling_features"].astype(np.float32),
        observed=payload["observed"].astype(bool),
        added=payload["added"].astype(bool),
        unknown=payload["unknown"].astype(bool),
    )


def _load_query_sidecar(config: PipelineConfig, split: str, index: int) -> tuple[RayQuerySet, dict[str, np.ndarray]]:
    path = query_sidecar_root(config) / split / f"{int(index):06d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"NYU query sidecar not found: {path}. Run `python run.py --mode cache` first."
        )
    with np.load(path) as data:
        payload = {name: data[name].copy() for name in data.files}
    return _query_from_sidecar(payload), payload


def _predict_with_stencils(
    model: RayAwareQueryRefiner,
    config: PipelineConfig,
    source,
    query: RayQuerySet,
    stencil_rays_np: np.ndarray,
    stencil_observed_np: np.ndarray,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    rgb, depth0, normal0, rays_source, valid_source = source
    device = rgb.device
    q_rays, q_uv, q_rel, q_sampling, q_obs = _query_tensors(query, device)
    source_features = model.encode_source(rgb, depth0, normal0, rays_source, valid_source)
    center = model.decode_queries(
        depth0, normal0, rays_source, valid_source, source_features,
        q_rays, q_uv, q_rel, q_sampling, q_obs,
    )
    stencil_rays = torch.from_numpy(stencil_rays_np.astype(np.float32)).unsqueeze(0).to(device)
    flat_stencil = stencil_rays.reshape(1, -1, 3)
    uv_np, uv_valid_np = project_fisheye_rays(flat_stencil.detach().cpu().numpy()[0], config.camera)
    s_uv = torch.from_numpy(uv_np.astype(np.float32)).unsqueeze(0).to(device)
    stencil_observed = stencil_observed_np.reshape(-1).astype(bool)
    s_valid = torch.from_numpy((uv_valid_np & stencil_observed).astype(np.float32)).unsqueeze(0).to(device)
    s_rel = q_rel.repeat_interleave(4, dim=1)
    s_sampling = q_sampling.repeat_interleave(4, dim=1)
    s_obs = (q_obs.repeat_interleave(4, dim=1) * s_valid).to(device)
    stencil = model.decode_queries(
        depth0, normal0, rays_source, valid_source, source_features,
        flat_stencil, s_uv, s_rel, s_sampling, s_obs,
    )
    stencil_depth = stencil["depth_final_z"].reshape(1, -1, 4)
    normal_star, stencil_valid = normals_from_stencil_depths(center["depth_final_z"], stencil_depth, q_rays, stencil_rays, z_eps=config.camera.geometry_z_eps)
    return center, normal_star, stencil_valid, stencil


def train_refiner(config: PipelineConfig) -> Path:
    """NYU teacher cache로 RayAwareQueryRefiner를 학습한다."""

    cache_root = config.paths.outputs / "cache" / "nyu"
    train_indices = read_nyu_split(config.paths.nyu_split_train)
    dataset = TeacherCacheDataset(cache_root, "train", train_indices, max_items=config.train.max_train_items)
    loader = DataLoader(dataset, batch_size=int(config.train.batch_size), shuffle=True, num_workers=0, collate_fn=_collate_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config.train.seed))
    model = RayAwareQueryRefiner(config.refiner).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
    loss_fn = PartialSupervisionLoss(config.loss, config.toggles)
    run_dir = config.paths.outputs / "train" / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.json")

    model.train()
    for epoch in range(int(config.train.epochs)):
        stats = {"loss": 0.0, "count": 0}
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}/{config.train.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            batch_has_loss = False
            for sample in batch:
                source = _to_bchw(sample, device)
                rgb, depth0, normal0, rays_source, valid_source, _ = source
                queries, query_payload = _load_query_sidecar(config, "train", int(sample["index"]))
                if len(queries) == 0:
                    continue
                target_depth = torch.from_numpy(query_payload["query_depth_gt_z"].astype(np.float32)).unsqueeze(0).to(device)
                query_gt_valid = torch.from_numpy(query_payload["query_gt_valid"].astype(bool)).unsqueeze(0).to(device)
                query_observed_gt = torch.from_numpy(query_payload["query_source_observed"].astype(bool)).unsqueeze(0).to(device)
                stencil_gt_depth = torch.from_numpy(query_payload["stencil_depth_gt_z"].astype(np.float32)).unsqueeze(0).to(device)
                stencil_gt_valid = torch.from_numpy(query_payload["stencil_gt_valid"].astype(bool)).unsqueeze(0).to(device)
                stencil_observed_gt = torch.from_numpy(query_payload["stencil_source_observed"].astype(bool)).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=config.train.amp and device.type == "cuda"):
                    center, normal_star, stencil_valid, _ = _predict_with_stencils(
                        model,
                        config,
                        (rgb, depth0, normal0, rays_source, valid_source),
                        queries,
                        query_payload["stencil_ray_dir"],
                        query_payload["stencil_source_observed"],
                    )
                    q_rays, q_uv, _, _, q_obs = _query_tensors(queries, device)
                    target_normal = center["normal0_query"]
                    source_valid_q = center["source_valid_query"] > 0.5
                    observed = q_obs > 0.5
                    depth_mask = (
                        observed
                        & source_valid_q
                        & query_observed_gt
                        & query_gt_valid
                        & torch.isfinite(target_depth)
                        & (target_depth > 0.0)
                    )
                    all_gt_depth = torch.cat([target_depth.unsqueeze(-1), stencil_gt_depth], dim=-1)
                    all_gt_valid = query_gt_valid & stencil_gt_valid.all(dim=-1) & stencil_observed_gt.all(dim=-1)
                    safe_log_depth = torch.log(
                        torch.nan_to_num(all_gt_depth, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-4)
                    )
                    depth_continuous = all_gt_valid & (
                        safe_log_depth.amax(dim=-1) - safe_log_depth.amin(dim=-1)
                        <= float(config.loss.depth_discontinuity_log_threshold)
                    )
                    normal_mask = depth_mask & stencil_valid & depth_continuous
                    radial_mask = depth_mask & depth_continuous
                    delta_mask = observed & source_valid_q
                    result = loss_fn(
                        pred_depth_z=center["depth_final_z"],
                        target_depth_z=target_depth,
                        pred_normal=normal_star,
                        target_normal=target_normal,
                        query_rays=q_rays,
                        source_rays_query=center["source_ray_query"],
                        depth0_query_z=center["depth0_query_z"],
                        delta_log_depth=center["delta_log_depth"],
                        masks=PartialLossMasks(
                            depth=depth_mask,
                            normal=normal_mask,
                            radial=radial_mask,
                            delta=delta_mask,
                        ),
                    )
                # sample별로 즉시 backward해 1024² source encoder graph 두 개가
                # batch_size=2에서 동시에 GPU에 남지 않도록 한다.
                scaler.scale(result.total / max(1, len(batch))).backward()
                batch_has_loss = True
                stats["loss"] += float(result.total.detach().cpu())
                stats["count"] += 1
            if not batch_has_loss:
                continue
            # AMP는 non-finite gradient가 있을 때 optimizer step을 조용히 skip할 수 있다.
            # 이 파이프라인에서는 mask 오류를 즉시 드러내기 위해 명시적으로 검사한다.
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            if not gradients_finite:
                raise FloatingPointError(
                    "Refiner gradient에 NaN/Inf가 발견되었습니다. "
                    "depth/normal/view mask와 cached query의 유효성을 확인하세요."
                )
            if config.train.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=stats["loss"] / max(1, stats["count"]))
        save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt", model, optimizer, epoch + 1, {"train_loss": stats["loss"] / max(1, stats["count"])})
    save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, int(config.train.epochs), {})
    return run_dir / "checkpoints" / "last.pt"
