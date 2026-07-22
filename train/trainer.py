from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wide_fov_supervision_v2.config import PipelineConfig, save_config
from wide_fov_supervision_v2.datasets.nyu.quad_dataset import NYUQuadCompletionDataset
from wide_fov_supervision_v2.loss.completion_loss import QuadCompletionLoss
from wide_fov_supervision_v2.modules.quad_completion.model import QuadRayCompletionModel
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths
from wide_fov_supervision_v2.train.checkpoints import load_checkpoint, save_checkpoint


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def predict_batch_with_stencils(
    model: QuadRayCompletionModel,
    batch: dict[str, torch.Tensor],
    *,
    z_eps: float,
    compute_normals: bool = True,
) -> tuple:
    """중심 query와 relative-UV stencil을 한 모델로 예측해 N*과 GT normal을 만든다."""

    result = model(
        batch["support_ray_dir"],
        batch["support_rgb"],
        batch["support_depth_z"],
        batch["support_valid"],
        batch["query_ray_dir"],
        batch["query_relative_uv"],
        batch["query_mask"],
    )
    if not compute_normals:
        return result, None, None, None
    b, q = batch["query_mask"].shape
    stencil_result = model(
        batch["support_ray_dir"],
        batch["support_rgb"],
        batch["support_depth_z"],
        batch["support_valid"],
        batch["stencil_ray_dir"].reshape(b, q * 4, 3),
        batch["stencil_relative_uv"].reshape(b, q * 4, 2),
        batch["stencil_valid"].reshape(b, q * 4),
    )
    stencil_depth = stencil_result.depth_z.reshape(b, q, 4)
    pred_normal, pred_normal_valid = normals_from_stencil_depths(
        result.depth_z,
        stencil_depth,
        batch["query_ray_dir"],
        batch["stencil_ray_dir"],
        z_eps=z_eps,
    )
    target_normal, target_normal_valid = normals_from_stencil_depths(
        batch["target_depth_z"],
        batch["stencil_depth_z"],
        batch["query_ray_dir"],
        batch["stencil_ray_dir"],
        z_eps=z_eps,
    )
    normal_mask = (
        pred_normal_valid
        & target_normal_valid
        & batch["target_valid"].bool()
        & batch["stencil_valid"].bool().all(dim=-1)
    )
    return result, pred_normal, target_normal, normal_mask


def train_completion_model(config: PipelineConfig) -> Path:
    """NYU convex-quad dataset으로 단일-pass RGB-D completion 모델을 학습한다."""

    torch.manual_seed(int(config.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NYUQuadCompletionDataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(config.train.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = QuadRayCompletionModel(config.completion).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if config.paths.checkpoint is not None:
        payload = load_checkpoint(config.paths.checkpoint, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        history_path = config.paths.checkpoint.parent.parent / "history.json"
        if history_path.exists():
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            history = [entry for entry in loaded if int(entry.get("epoch", 0)) <= start_epoch]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.train.amp and device.type == "cuda",
        init_scale=float(config.train.amp_initial_scale),
    )
    loss_fn = QuadCompletionLoss(config.loss, config.toggles)
    run_dir = (
        config.paths.checkpoint.parent.parent
        if config.paths.checkpoint is not None
        else config.paths.outputs / "train" / time.strftime("%Y_%m_%d_%H_%M_%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.json")

    for epoch in range(start_epoch, int(config.train.epochs)):
        model.train()
        sums = {name: 0.0 for name in ("total", "depth", "rgb", "valid", "confidence", "normal", "residual", "cycle")}
        batch_count = 0
        progress = tqdm(loader, desc=f"quad completion epoch {epoch + 1}/{config.train.epochs}")
        for raw_batch in progress:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.train.amp and device.type == "cuda"):
                result, pred_normal, target_normal, normal_mask = predict_batch_with_stencils(
                    model,
                    batch,
                    z_eps=config.camera.geometry_z_eps,
                    compute_normals=bool(config.toggles.enable_normal_loss),
                )
                losses = loss_fn(
                    result=result,
                    support_rgb=batch["support_rgb"],
                    support_depth_z=batch["support_depth_z"],
                    support_valid=batch["support_valid"],
                    query_relative_uv=batch["query_relative_uv"],
                    source_continuous=batch["source_continuous"],
                    target_rgb=batch["target_rgb"],
                    target_depth_z=batch["target_depth_z"],
                    target_valid=batch["target_valid"],
                    target_confidence=batch["target_confidence"],
                    query_mask=batch["query_mask"],
                    corner_query_mask=batch["corner_query_mask"],
                    pred_normal=pred_normal,
                    target_normal=target_normal,
                    normal_mask=normal_mask,
                )
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise FloatingPointError("Quad completion gradient에 NaN/Inf가 발견되었습니다.")
            if config.train.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            for name in sums:
                sums[name] += float(getattr(losses, name).detach().cpu())
            batch_count += 1
            progress.set_postfix(loss=sums["total"] / batch_count)

        epoch_metrics = {name: value / max(1, batch_count) for name, value in sums.items()}
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)
        save_checkpoint(
            run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt",
            model,
            optimizer,
            epoch + 1,
            epoch_metrics,
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_path = run_dir / "checkpoints" / "last.pt"
    save_checkpoint(final_path, model, optimizer, int(config.train.epochs), history[-1] if history else {})
    return final_path
