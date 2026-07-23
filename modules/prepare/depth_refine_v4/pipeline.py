from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wide_fov_supervision_v2.backbone.depth_anything import DepthAnythingMetricWrapper
from wide_fov_supervision_v2.modules.bev_mapping import build_bev_rgb
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays, camera_to_world_points, points_from_z_depth
from wide_fov_supervision_v2.modules.visualization import save_depth, save_heatmap, save_rgb

from .config import DepthRefineV4Config, save_v4_config
from .dataset import DepthRefineV4Dataset, build_v4_cache
from .edge_condition import condition_preview, depth_edge_condition, load_v2_edge_condition
from .losses import DepthRefineV4Loss
from .model import EdgeConditionedDepthRefiner
from .report import generate_report


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_environment(config: DepthRefineV4Config, input_rgb: Path | None = None) -> dict:
    checks = {
        "nyu_mat": config.base.paths.nyu_mat.exists(),
        "da_root": config.base.paths.depth_anything_root.exists(),
        "da_vitl_ckpt": config.base.paths.depth_anything_vitl_ckpt.exists(),
        "input_rgb": (input_rgb or config.base.paths.input_rgb).exists(),
        "v2_context_checkpoint": latest_v2_context_checkpoint(config) is not None,
        "device": str(_device()),
        "output_root": str(config.output_root),
    }
    return checks


def save_checkpoint(path: Path, model: EdgeConditionedDepthRefiner, optimizer: torch.optim.Optimizer | None, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": model.checkpoint_schema,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, model: EdgeConditionedDepthRefiner, *, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != model.checkpoint_schema:
        raise RuntimeError(f"V4 checkpoint schema mismatch: expected={model.checkpoint_schema}, actual={payload.get('schema')}")
    model.load_state_dict(payload["model"], strict=True)
    return payload


def train_refiner(config: DepthRefineV4Config) -> Path:
    build_v4_cache(config, "train")
    build_v4_cache(config, "test")
    torch.manual_seed(int(config.train.seed))
    device = _device()
    dataset = DepthRefineV4Dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(config.train.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = EdgeConditionedDepthRefiner(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.train.learning_rate), weight_decay=float(config.train.weight_decay))
    criterion = DepthRefineV4Loss(config.loss)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.train.amp and device.type == "cuda"))
    run_dir = config.output_root / "train" / time.strftime("%Y_%m_%d_%H_%M_%S")
    save_v4_config(config, run_dir / "config.json")
    best_loss = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    for epoch in range(1, int(config.train.epochs) + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(loader, desc=f"v4 epoch {epoch}/{config.train.epochs}")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            edge_condition = _augment_condition(
                batch["edge_condition"],
                dropout_probability=float(config.train.condition_dropout_probability),
                jitter_probability=float(config.train.condition_jitter_probability),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(config.train.amp and device.type == "cuda")):
                result = model(
                    batch["rgb"],
                    batch["depth0_z"],
                    batch["rays"],
                    batch["valid"],
                    edge_condition,
                    batch.get("da_features"),
                )
                loss_batch = {**batch, "edge_condition": edge_condition}
                loss = criterion(result, loss_batch)
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.total.detach().cpu())
            losses.append(value)
            progress.set_postfix(loss=f"{value:.4f}")
        metrics = {"train_loss": float(np.mean(losses)) if losses else float("nan")}
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, metrics)
        save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, metrics)
        if metrics["train_loss"] < best_loss:
            best_loss = metrics["train_loss"]
            save_checkpoint(best_path, model, optimizer, epoch, metrics)
    return best_path


@torch.inference_mode()
def evaluate_refiner(config: DepthRefineV4Config, checkpoint: Path | None = None) -> Path:
    build_v4_cache(config, "test")
    device = _device()
    model = EdgeConditionedDepthRefiner(config.model).to(device).eval()
    ckpt = checkpoint or latest_checkpoint(config)
    if ckpt is not None:
        load_checkpoint(ckpt, model, map_location=device)
    loader = DataLoader(DepthRefineV4Dataset(config, "test"), batch_size=int(config.train.batch_size), shuffle=False, num_workers=0)
    metrics = []
    for batch in tqdm(loader, desc="v4 evaluate"):
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model(
            batch["rgb"],
            batch["depth0_z"],
            batch["rays"],
            batch["valid"],
            batch["edge_condition"],
            batch.get("da_features"),
        )
        zero_result = model(
            batch["rgb"],
            batch["depth0_z"],
            batch["rays"],
            batch["valid"],
            torch.zeros_like(batch["edge_condition"]),
            batch.get("da_features"),
        )
        metrics.append(
            {
                "d0": _depth_metrics(batch["depth0_z"], batch["depth_gt_z"], batch["valid"]),
                "zero_condition": _depth_metrics(zero_result.depth_final_z, batch["depth_gt_z"], batch["valid"]),
                "final": _depth_metrics(result.depth_final_z, batch["depth_gt_z"], batch["valid"]),
            }
        )
    out = config.output_root / "eval" / time.strftime("%Y_%m_%d_%H_%M_%S")
    out.mkdir(parents=True, exist_ok=True)
    merged = {
        f"{name}_{key}": float(np.nanmean([item[name][key] for item in metrics]))
        for name in ("d0", "zero_condition", "final")
        for key in ("absrel", "rmse")
    }
    (out / "metrics.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    save_v4_config(config, out / "config.json")
    return out


@torch.inference_mode()
def run_inference(
    config: DepthRefineV4Config,
    *,
    input_rgb: Path | None = None,
    checkpoint: Path | None = None,
    evaluation_depth: Path | None = None,
    depth0_path: Path | None = None,
    edge_run: Path | None = None,
) -> Path:
    input_path = input_rgb or config.base.paths.input_rgb
    rgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    rays_data = build_fisheye_rays(config.base.camera)
    rays = rays_data.rays_cv
    source_valid = rays_data.valid & (rays[..., 2] > config.base.camera.geometry_z_eps)
    if rgb.shape[:2] != source_valid.shape:
        rgb = cv2.resize(rgb, (config.base.camera.width, config.base.camera.height), interpolation=cv2.INTER_AREA)
    da_features = None
    if depth0_path is not None:
        depth0 = np.asarray(np.load(depth0_path), dtype=np.float32)
        _unused_depth, da_features = DepthAnythingMetricWrapper(config.base.paths, config.base.backbone).predict_with_features(rgb)
    else:
        depth0, da_features = DepthAnythingMetricWrapper(config.base.paths, config.base.backbone).predict_with_features(rgb)
    if depth0.shape != source_valid.shape:
        depth0 = cv2.resize(depth0.astype(np.float32), (config.base.camera.width, config.base.camera.height), interpolation=cv2.INTER_LINEAR)
    run_dir = config.output_root / "inference" / time.strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(run_dir / "source_rgb.png", rgb)
    np.save(run_dir / "depth0_z.npy", depth0.astype(np.float32))
    save_depth(run_dir / "depth0_z.png", depth0)

    variant_dir = _ensure_v2_edge_run(config, run_dir, input_path, edge_run, run_dir / "depth0_z.npy")
    if variant_dir is None:
        condition, _ = depth_edge_condition(depth0, source_valid, threshold=float(config.data.depth_edge_log_threshold), band_radius=int(config.data.edge_band_radius_px))
        edge_source = "depth0_gradient_fallback"
    else:
        condition = load_v2_edge_condition(variant_dir, source_valid.shape, depth0)
        edge_source = str(variant_dir)
    np.save(run_dir / "edge_condition.npy", condition)
    save_rgb(run_dir / "edge_condition.png", condition_preview(condition))

    model = EdgeConditionedDepthRefiner(config.model).to(_device()).eval()
    ckpt = checkpoint or latest_checkpoint(config)
    if ckpt is not None and ckpt.exists():
        load_checkpoint(ckpt, model, map_location=_device())
        checkpoint_used = str(ckpt)
    else:
        checkpoint_used = None
    batch = {
        "rgb": torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]).to(_device()),
        "rays": torch.from_numpy(rays.astype(np.float32).transpose(2, 0, 1)[None]).to(_device()),
        "valid": torch.from_numpy(source_valid[None, None]).to(_device()),
        "depth0_z": torch.from_numpy(depth0.astype(np.float32)[None, None]).to(_device()),
        "edge_condition": torch.from_numpy(condition.astype(np.float32)[None]).to(_device()),
        "da_features": torch.from_numpy(da_features.astype(np.float32)[None]).to(_device()),
    }
    result = model(
        batch["rgb"],
        batch["depth0_z"],
        batch["rays"],
        batch["valid"],
        batch["edge_condition"],
        batch["da_features"],
    )
    zero_result = model(
        batch["rgb"],
        batch["depth0_z"],
        batch["rays"],
        batch["valid"],
        torch.zeros_like(batch["edge_condition"]),
        batch["da_features"],
    )
    depth_final = result.depth_final_z[0, 0].detach().cpu().numpy().astype(np.float32)
    depth_zero_condition = zero_result.depth_final_z[0, 0].detach().cpu().numpy().astype(np.float32)
    depth_diffusion = edge_aware_diffusion(depth0, condition, source_valid)
    delta = result.delta_log_depth[0, 0].detach().cpu().numpy().astype(np.float32)
    gate = result.gate[0, 0].detach().cpu().numpy().astype(np.float32)
    np.save(run_dir / "depth_final_z.npy", depth_final)
    np.save(run_dir / "depth_zero_condition_z.npy", depth_zero_condition)
    np.save(run_dir / "depth_edge_diffusion_z.npy", depth_diffusion)
    np.save(run_dir / "delta_log_depth.npy", delta)
    np.save(run_dir / "refinement_gate.npy", gate)
    save_depth(run_dir / "depth_final_z.png", depth_final)
    save_depth(run_dir / "depth_zero_condition_z.png", depth_zero_condition)
    save_depth(run_dir / "depth_edge_diffusion_z.png", depth_diffusion)
    save_heatmap(run_dir / "delta_log_depth.png", delta, value_min=-float(config.model.max_delta_log_depth), value_max=float(config.model.max_delta_log_depth))
    save_heatmap(run_dir / "refinement_gate.png", gate, value_min=0.0, value_max=1.0)
    _save_bev(config, run_dir, rgb, rays, source_valid, depth0, "d0")
    _save_bev(config, run_dir, rgb, rays, source_valid, depth_diffusion, "edge_diffusion")
    _save_bev(config, run_dir, rgb, rays, source_valid, depth_zero_condition, "zero_condition")
    _save_bev(config, run_dir, rgb, rays, source_valid, depth_final, "final")
    metrics: dict[str, float | int | str | None] = {}
    if evaluation_depth is not None and evaluation_depth.exists():
        gt = np.asarray(np.load(evaluation_depth), dtype=np.float32)
        if gt.shape != source_valid.shape:
            gt = cv2.resize(gt, (source_valid.shape[1], source_valid.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = source_valid & np.isfinite(gt) & (gt > 0.0)
        metrics.update({f"d0_{k}": v for k, v in _depth_metrics_np(depth0, gt, valid).items()})
        metrics.update({f"edge_diffusion_{k}": v for k, v in _depth_metrics_np(depth_diffusion, gt, valid).items()})
        metrics.update({f"zero_condition_{k}": v for k, v in _depth_metrics_np(depth_zero_condition, gt, valid).items()})
        metrics.update({f"final_{k}": v for k, v in _depth_metrics_np(depth_final, gt, valid).items()})
        save_heatmap(run_dir / "depth_error_d0.png", np.where(valid, np.abs(depth0 - gt), np.nan), cmap_name="magma")
        save_heatmap(run_dir / "depth_error_final.png", np.where(valid, np.abs(depth_final - gt), np.nan), cmap_name="magma")
    metadata = {
        "input_rgb": str(input_path),
        "checkpoint": checkpoint_used,
        "edge_condition_source": edge_source,
        "evaluation_depth_used_as_input": False,
        "scale_policy": "D* = D0 * exp(zero_mean_delta_log_depth)",
        "comparison_outputs": ["d0", "edge_diffusion", "zero_condition", "final"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save_v4_config(config, run_dir / "config.json")
    generate_report(run_dir, metadata, metrics)
    return run_dir


def _augment_condition(
    edge_condition: torch.Tensor,
    *,
    dropout_probability: float,
    jitter_probability: float,
) -> torch.Tensor:
    result = edge_condition
    if dropout_probability > 0.0:
        keep = (torch.rand((result.shape[0], 1, 1, 1), device=result.device) >= dropout_probability).to(result.dtype)
        result = result * keep
    if jitter_probability > 0.0 and torch.rand((), device=result.device) < jitter_probability:
        shift_y = int(torch.randint(-1, 2, (), device=result.device).item())
        shift_x = int(torch.randint(-1, 2, (), device=result.device).item())
        result = torch.roll(result, shifts=(shift_y, shift_x), dims=(-2, -1))
    return result


def edge_aware_diffusion(depth0: np.ndarray, condition: np.ndarray, valid: np.ndarray, iterations: int = 24) -> np.ndarray:
    """Nonlearned comparison baseline: smooth D0 while using V2 edge as a barrier."""

    depth = np.asarray(depth0, dtype=np.float32).copy()
    mask = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    edge = np.clip(np.nan_to_num(condition[0], nan=0.0), 0.0, 1.0)
    barrier = np.exp(-6.0 * edge).astype(np.float32)
    log_depth = np.log(np.where(mask, depth, 1.0)).astype(np.float32)
    for _ in range(int(iterations)):
        accum = log_depth.copy()
        weight = np.ones_like(log_depth, dtype=np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.roll(log_depth, shift=(dy, dx), axis=(0, 1))
            shifted_mask = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            shifted_barrier = np.roll(barrier, shift=(dy, dx), axis=(0, 1))
            w = np.minimum(barrier, shifted_barrier) * shifted_mask.astype(np.float32)
            accum += shifted * w
            weight += w
        log_depth = np.where(mask, accum / np.maximum(weight, 1.0e-6), log_depth)
    output = np.exp(log_depth).astype(np.float32)
    output[~mask] = np.nan
    return output


def latest_checkpoint(config: DepthRefineV4Config) -> Path | None:
    root = config.output_root / "train"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None


def latest_v2_context_checkpoint(config: DepthRefineV4Config) -> Path | None:
    root = config.base.paths.outputs / "edge_estimate" / "v2" / "train" / "rgb_context"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None


def _ensure_v2_edge_run(config: DepthRefineV4Config, run_dir: Path, input_rgb: Path, edge_run: Path | None, depth0_path: Path) -> Path | None:
    if edge_run is not None:
        variant = edge_run / config.inference.edge_variant
        return variant if variant.exists() else edge_run
    if not config.inference.run_v2_if_missing:
        return None
    ckpt = latest_v2_context_checkpoint(config)
    if ckpt is None:
        return None
    from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.config import make_edge_config
    from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.pipeline import run_inference as run_v2_inference

    edge_config = make_edge_config()
    v2_dir = run_v2_inference(
        edge_config,
        {"rgb_context": ckpt},
        input_rgb=input_rgb,
        evaluation_depth_path=None,
        prior_depth_path=depth0_path,
        base_bev_run=None,
    )
    (run_dir / "v2_edge_run.txt").write_text(str(v2_dir), encoding="utf-8")
    return v2_dir / "rgb_context"


def _depth_metrics(result, target, valid) -> dict[str, float]:
    pred = result.detach()
    tgt = target.detach()
    mask = valid.bool() & torch.isfinite(pred) & torch.isfinite(tgt) & (pred > 0.0) & (tgt > 0.0)
    if not torch.any(mask):
        return {"absrel": float("nan"), "rmse": float("nan")}
    error = pred[mask] - tgt[mask]
    return {"absrel": float(torch.mean(torch.abs(error) / tgt[mask]).cpu()), "rmse": float(torch.sqrt(torch.mean(error * error)).cpu())}


def _depth_metrics_np(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0.0) & (target > 0.0)
    if not np.any(mask):
        return {"count": 0, "absrel": float("nan"), "rmse": float("nan")}
    error = prediction[mask] - target[mask]
    return {"count": int(mask.sum()), "absrel": float(np.mean(np.abs(error) / target[mask])), "rmse": float(np.sqrt(np.mean(error**2)))}


def _save_bev(config: DepthRefineV4Config, run_dir: Path, rgb: np.ndarray, rays: np.ndarray, valid: np.ndarray, depth: np.ndarray, name: str) -> None:
    points, _, point_valid = points_from_z_depth(depth, rays, z_eps=float(config.base.camera.geometry_z_eps))
    world = camera_to_world_points(points, config.base.camera).astype(np.float32)
    bev, bev_valid, _ = build_bev_rgb(rgb.reshape(-1, 3), world.reshape(-1, 3), (valid & point_valid).reshape(-1), config.base.bev)
    Image.fromarray(bev[..., :3]).save(run_dir / f"bev_{name}.png")
    Image.fromarray(bev_valid).save(run_dir / f"bev_{name}_valid.png")
