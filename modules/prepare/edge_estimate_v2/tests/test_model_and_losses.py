from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from wide_fov_supervision_v2.config import BevConfig, FisheyeCameraConfig
from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.config import (
    EdgeEstimateConfig,
    EdgeLossConfig,
    EdgeModelConfig,
    save_edge_config,
)
from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.edge_prior import estimate_2d_edge_prior
from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.losses import EdgeEstimateLoss
from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.model import EdgeEstimateModel
from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.pipeline import (
    latest_checkpoints,
    load_checkpoint,
    run_inference,
    save_checkpoint,
)


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(3)
    support_rgb = torch.rand(batch, 5, 5, 3)
    support_rays = torch.nn.functional.normalize(
        torch.rand(batch, 5, 5, 3) + torch.tensor([0.0, 0.0, 2.0]), dim=-1
    )
    support_valid = torch.ones(batch, 5, 5, dtype=torch.bool)
    query_rays = torch.nn.functional.normalize(
        torch.rand(batch, 4, 4, 64, 3) + torch.tensor([0.0, 0.0, 2.0]), dim=-1
    )
    query_uv = torch.rand(batch, 4, 4, 64, 2)
    return support_rgb, support_rays, support_valid, query_rays, query_uv


def test_three_variants_have_public_output_shapes() -> None:
    inputs = _inputs()
    for variant in ("rgb_local", "rgb_context", "rgb_da_context"):
        model = EdgeEstimateModel(
            EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=3),
            variant,
        )
        result = model(*inputs)
        assert result.cell_edge_logit.shape == (2, 4, 4)
        assert result.cell_type_logits.shape == (2, 4, 4, 3)
        assert result.query_edge_logit.shape == (2, 4, 4, 64)
        assert result.query_depth_near_z.shape == (2, 4, 4, 64)
        assert torch.all(result.query_depth_far_z >= result.query_depth_near_z)


def test_2d_edge_prior_responds_to_rgb_boundary() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, 16:] = 255
    edge = estimate_2d_edge_prior(rgb)

    assert edge.shape == (32, 32)
    assert np.isfinite(edge).all()
    assert float(edge[:, 15:17].mean()) > float(edge[:, :8].mean()) + 0.25


def test_2d_edge_prior_is_used_by_model_encoder() -> None:
    inputs = _inputs(batch=1)
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_context",
    ).eval()
    no_edge = torch.zeros(1, 5, 5)
    strong_edge = torch.ones(1, 5, 5)

    first = model(*inputs, support_edge_2d=no_edge)
    second = model(*inputs, support_edge_2d=strong_edge)

    assert not torch.allclose(first.cell_edge_logit, second.cell_edge_logit)


def test_prior_depth_scale_is_preserved() -> None:
    inputs = _inputs(batch=1)
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_context",
    ).eval()
    prior = torch.full((1, 4, 4, 64), 1.5)
    prior_valid = torch.ones_like(prior, dtype=torch.bool)

    first = model(*inputs, query_prior_depth_z=prior, query_prior_valid=prior_valid)
    second = model(*inputs, query_prior_depth_z=2.0 * prior, query_prior_valid=prior_valid)

    torch.testing.assert_close(second.query_delta_log_depth, first.query_delta_log_depth)
    torch.testing.assert_close(second.query_depth_near_z, 2.0 * first.query_depth_near_z)


def test_rgb_only_model_does_not_read_da_tensor() -> None:
    inputs = _inputs(batch=1)
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_context",
    ).eval()
    zeros = torch.zeros(1, 5, 5)
    random = torch.randn(1, 5, 5) * 100.0
    first = model(*inputs, zeros, torch.zeros_like(zeros, dtype=torch.bool))
    second = model(*inputs, random, torch.ones_like(zeros, dtype=torch.bool))
    torch.testing.assert_close(first.query_edge_logit, second.query_edge_logit)
    torch.testing.assert_close(first.query_depth_near_z, second.query_depth_near_z)


def test_loss_is_nan_safe_and_backpropagates() -> None:
    inputs = _inputs(batch=1)
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_context",
    )
    result = model(*inputs)
    query_mask = torch.ones(1, 4, 4, 64, dtype=torch.bool)
    query_mask[..., -1] = False
    target_near = torch.full((1, 4, 4, 64), 2.0)
    target_near[..., -1] = float("nan")
    batch = {
        "cell_valid": torch.ones(1, 4, 4, dtype=torch.bool),
        "query_mask": query_mask,
        "target_cell_edge": torch.randint(0, 2, (1, 4, 4)).float(),
        "target_query_edge": torch.randint(0, 2, (1, 4, 4, 64)).float(),
        "target_cell_type": torch.randint(0, 4, (1, 4, 4)),
        "target_near_depth_z": target_near,
        "target_far_depth_z": torch.full((1, 4, 4, 64), 4.0),
        "target_near_valid": query_mask,
        "target_far_valid": torch.zeros_like(query_mask),
        "target_confidence": query_mask.float(),
        "query_ray_dir": inputs[3],
    }
    loss = EdgeEstimateLoss(EdgeLossConfig())(result, batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_empty_masks_make_depth_type_and_consistency_graph_zeros() -> None:
    inputs = _inputs(batch=1)
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_context",
    )
    result = model(*inputs)
    empty_query = torch.zeros(1, 4, 4, 64, dtype=torch.bool)
    batch = {
        "cell_valid": torch.zeros(1, 4, 4, dtype=torch.bool),
        "query_mask": empty_query,
        "target_cell_edge": torch.full((1, 4, 4), float("nan")),
        "target_query_edge": torch.full((1, 4, 4, 64), float("nan")),
        "target_cell_type": torch.zeros(1, 4, 4, dtype=torch.long),
        "target_near_depth_z": torch.full((1, 4, 4, 64), float("nan")),
        "target_far_depth_z": torch.full((1, 4, 4, 64), float("nan")),
        "target_near_valid": empty_query,
        "target_far_valid": empty_query,
        "target_confidence": torch.full((1, 4, 4, 64), float("nan")),
        "query_ray_dir": inputs[3],
    }
    loss = EdgeEstimateLoss(EdgeLossConfig())(result, batch)
    for value in (
        loss.coarse,
        loss.query_focal,
        loss.query_dice,
        loss.edge_type,
        loss.near_depth,
        loss.far_depth,
        loss.confidence,
        loss.boundary_consistency,
        loss.tangent,
    ):
        torch.testing.assert_close(value, torch.zeros_like(value))
    loss.total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_checkpoint_rejects_variant_mix(tmp_path: Path) -> None:
    config = EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1)
    local = EdgeEstimateModel(config, "rgb_local")
    checkpoint = tmp_path / "local.pt"
    save_checkpoint(checkpoint, local, None, 1, {})
    context = EdgeEstimateModel(config, "rgb_context")
    with pytest.raises(RuntimeError, match="schema"):
        load_checkpoint(checkpoint, context)


def test_latest_checkpoint_prefers_completed_run_over_newer_incomplete_run(tmp_path: Path) -> None:
    config = EdgeEstimateConfig()
    config.base.paths.outputs = tmp_path / "outputs"
    model = EdgeEstimateModel(
        EdgeModelConfig(point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1),
        "rgb_local",
    )
    variant_root = config.output_root / "train" / "rgb_local"
    complete = variant_root / "2026_01_01_00_00_00"
    incomplete = variant_root / "2026_01_02_00_00_00"
    for run_dir in (complete, incomplete):
        save_edge_config(config, run_dir / "config.json")
        save_checkpoint(run_dir / "checkpoints" / "best.pt", model, None, 1, {})
    save_checkpoint(complete / "checkpoints" / "last.pt", model, None, config.train.epochs, {})

    selected = latest_checkpoints(config)

    assert selected["rgb_local"] == complete / "checkpoints" / "best.pt"


def test_small_image_inference_writes_query_bev_and_report(tmp_path: Path) -> None:
    config = EdgeEstimateConfig()
    config.base.paths.outputs = tmp_path / "outputs"
    config.base.camera = FisheyeCameraConfig(
        width=16,
        height=16,
        fx=6.0,
        fy=6.0,
        cx=7.5,
        cy=7.5,
        world_from_camera=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        camera_position_world=(0.0, 0.0, 0.0),
    )
    config.base.bev = BevConfig(center_xy=(0.0, 1.0), size_m=4.0, meters_per_pixel=0.1)
    config.model = EdgeModelConfig(
        point_hidden_dim=16, cell_hidden_dim=32, query_hidden_dim=32, context_blocks=1
    )
    config.inference.coarse_threshold = 0.0
    config.inference.query_edge_threshold = 0.0
    config.inference.confidence_threshold = 0.0
    config.inference.candidate_dilation_cells = 0
    config.inference.max_candidate_cells = 32
    config.inference.coarse_batch_size = 16
    config.inference.query_batch_size = 16
    config.inference.hash_guard_evaluation_depth = False
    image_path = tmp_path / "input.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)
    evaluation_depth = np.full((16, 16), 2.0, dtype=np.float32)
    evaluation_depth[:, 8:] = 4.0
    evaluation_depth_path = tmp_path / "depth.npy"
    np.save(evaluation_depth_path, evaluation_depth)
    model = EdgeEstimateModel(config.model, "rgb_local")
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, None, 0, {})

    run_dir = run_inference(
        config,
        {"rgb_local": checkpoint},
        input_rgb=image_path,
        evaluation_depth_path=evaluation_depth_path,
    )

    variant = run_dir / "rgb_local"
    assert (variant / "edge_queries.npz").exists()
    assert (variant / "edge_2d_prior.png").exists()
    assert (variant / "edge_3d_preview.png").exists()
    assert (variant / "edge_only" / "bev_edge_near.png").exists()
    assert (variant / "edge_only" / "bev_edge_polyline.png").exists()
    assert (variant / "edge_only" / "bev_edge_occupancy.png").exists()
    assert (variant / "edge_only" / "bev_edge_projected_with_gt_depth.png").exists()
    assert (variant / "gt" / "bev_edge_gt.png").exists()
    assert (run_dir / "index.html").exists()
    with np.load(variant / "edge_queries.npz") as queries:
        assert len(queries["ray_dir"]) > 0
        assert np.all(queries["completed"] == ~queries["unknown"])
