from __future__ import annotations

import numpy as np
import torch
import inspect

from wide_fov_supervision_v2.config import CompletionConfig, LossConfig, StageToggles
from wide_fov_supervision_v2.datasets.nyu.quad_dataset import _confidence_targets
from wide_fov_supervision_v2.loss.completion_loss import QuadCompletionLoss
from wide_fov_supervision_v2.modules.quad_completion.geometry import (
    CORNER_RELATIVE_UV,
    bilinear_quad_jacobian,
    bilinear_quad_map,
    is_convex_quad,
    quad_is_valid,
)
from wide_fov_supervision_v2.modules.quad_completion.model import QuadCompletionResult, QuadRayCompletionModel
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths


def _inputs(batch: int = 2, queries: int = 7):
    torch.manual_seed(3)
    support_rays = torch.nn.functional.normalize(torch.rand(batch, 4, 3) + torch.tensor([0.0, 0.0, 2.0]), dim=-1)
    support_rgb = torch.rand(batch, 4, 3)
    support_depth = torch.tensor([[1.0, 2.0, 4.0, 3.0]], dtype=torch.float32).expand(batch, -1).clone()
    support_valid = torch.ones(batch, 4, dtype=torch.bool)
    query_rays = torch.nn.functional.normalize(torch.rand(batch, queries, 3) + torch.tensor([0.0, 0.0, 2.0]), dim=-1)
    query_uv = torch.rand(batch, queries, 2)
    query_mask = torch.ones(batch, queries, dtype=torch.bool)
    return support_rays, support_rgb, support_depth, support_valid, query_rays, query_uv, query_mask


def test_convex_quad_mapping_and_corner_order() -> None:
    corners = np.array([[10, 20], [80, 18], [75, 90], [12, 84]], dtype=np.float32)
    assert is_convex_quad(corners)
    assert quad_is_valid(corners, image_width=100, image_height=100)
    mapped = bilinear_quad_map(corners[None], CORNER_RELATIVE_UV[None])[0]
    np.testing.assert_allclose(mapped, corners, atol=1.0e-6)
    assert np.all(bilinear_quad_jacobian(corners, np.array([[0.2, 0.3], [0.8, 0.7]], dtype=np.float32)) > 0.0)


def test_self_intersection_and_concave_quad_are_rejected() -> None:
    bow_tie = np.array([[0, 0], [2, 2], [2, 0], [0, 2]], dtype=np.float32)
    concave = np.array([[0, 0], [2, 0], [0.5, 0.5], [0, 2]], dtype=np.float32)
    assert not is_convex_quad(bow_tie)
    assert not is_convex_quad(concave)


def test_zero_residual_matches_bilinear_baseline_and_scale_equivariance() -> None:
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1)).eval()
    inputs = _inputs()
    first = model(*inputs)
    scaled = list(inputs)
    scaled[2] = inputs[2] * 7.0
    second = model(*scaled)
    torch.testing.assert_close(first.rgb, first.base_rgb)
    torch.testing.assert_close(first.depth_z, first.base_depth_z)
    torch.testing.assert_close(second.depth_z, first.depth_z * 7.0, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(second.delta_log_depth, first.delta_log_depth)


def test_arbitrary_query_count_and_padding_mask() -> None:
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=2))
    inputs = list(_inputs(batch=3, queries=11))
    inputs[-1][1, 7:] = False
    result = model(*inputs)
    assert result.rgb.shape == (3, 11, 3)
    assert result.depth_z.shape == (3, 11)
    assert torch.all(result.delta_log_depth[1, 7:] == 0.0)


def test_public_interface_contains_only_four_support_and_query_inputs() -> None:
    parameters = list(inspect.signature(QuadRayCompletionModel.forward).parameters)
    assert parameters == [
        "self",
        "support_ray_dir",
        "support_rgb",
        "support_depth_z",
        "support_valid",
        "query_ray_dir",
        "query_relative_uv",
        "query_mask",
    ]


def test_continuous_confidence_is_one_and_depth_step_is_zero() -> None:
    continuous = np.ones((20, 20), dtype=np.float32) * 2.0
    xy = np.array([[5.5, 5.5], [10.0, 10.0]], dtype=np.float32)
    valid, confidence = _confidence_targets(continuous, xy, 0.20)
    assert np.all(valid & confidence)
    step = continuous.copy()
    step[:, 10:] = 4.0
    _, confidence = _confidence_targets(step, np.array([[9.5, 8.0]], dtype=np.float32), 0.20)
    assert not bool(confidence[0])


def test_loss_masks_nan_targets_and_backpropagates_finite_gradients() -> None:
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))
    inputs = _inputs(batch=1, queries=6)
    result = model(*inputs)
    target_depth = result.depth_z.detach().clone()
    target_depth[0, 2] = float("nan")
    target_valid = torch.ones(1, 6, dtype=torch.bool)
    target_valid[0, 2] = False
    loss = QuadCompletionLoss(LossConfig(), StageToggles())(
        result=result,
        support_depth_z=inputs[2],
        support_valid=inputs[3],
        target_rgb=result.rgb.detach(),
        target_depth_z=target_depth,
        target_valid=target_valid,
        target_confidence=target_valid,
        query_mask=inputs[-1],
        corner_query_mask=torch.zeros_like(target_valid),
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_cycle_reconstruction_loss_uses_internal_queries_and_backpropagates() -> None:
    inputs = list(_inputs(batch=1, queries=6))
    inputs[5] = torch.tensor(
        [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.50, 0.25], [0.50, 0.75]]],
        dtype=torch.float32,
    )
    baseline = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))(*inputs)
    pred_rgb = (baseline.rgb.detach() + 0.05 * inputs[5][..., :1]).clone().requires_grad_()
    pred_depth = (baseline.depth_z.detach() * (1.0 + 0.10 * inputs[5][..., 0])).clone().requires_grad_()
    result = QuadCompletionResult(
        rgb=pred_rgb,
        depth_z=pred_depth,
        valid_logit=torch.zeros_like(pred_depth),
        confidence_logit=torch.zeros_like(pred_depth),
        delta_log_depth=torch.zeros_like(pred_depth),
        rgb_residual=torch.zeros_like(pred_rgb),
        base_rgb=baseline.base_rgb.detach(),
        base_depth_z=baseline.base_depth_z.detach(),
    )
    toggles = StageToggles()
    toggles.enable_depth_loss = False
    toggles.enable_rgb_loss = False
    toggles.enable_valid_loss = False
    toggles.enable_confidence_loss = False
    toggles.enable_normal_loss = False
    loss = QuadCompletionLoss(LossConfig(cycle_reconstruction_weight=1.0, residual_weight=0.0), toggles)(
        result=result,
        support_rgb=inputs[1],
        support_depth_z=inputs[2],
        support_valid=inputs[3],
        query_relative_uv=inputs[5],
        target_rgb=baseline.rgb.detach(),
        target_depth_z=baseline.depth_z.detach(),
        target_valid=torch.ones(1, 6, dtype=torch.bool),
        target_confidence=torch.ones(1, 6, dtype=torch.bool),
        query_mask=inputs[-1],
        corner_query_mask=torch.zeros(1, 6, dtype=torch.bool),
    )
    loss.total.backward()
    assert loss.cycle_count == 4
    assert torch.isfinite(loss.cycle)
    assert pred_rgb.grad is not None and float(pred_rgb.grad.abs().sum()) > 0.0
    assert pred_depth.grad is not None and float(pred_depth.grad.abs().sum()) > 0.0


def test_cycle_reconstruction_loss_ignores_corner_queries() -> None:
    inputs = _inputs(batch=1, queries=4)
    result = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))(*inputs)
    loss = QuadCompletionLoss(LossConfig(cycle_reconstruction_weight=1.0), StageToggles())(
        result=result,
        support_rgb=inputs[1],
        support_depth_z=inputs[2],
        support_valid=inputs[3],
        query_relative_uv=inputs[5],
        target_rgb=result.rgb.detach(),
        target_depth_z=result.depth_z.detach(),
        target_valid=torch.ones(1, 4, dtype=torch.bool),
        target_confidence=torch.ones(1, 4, dtype=torch.bool),
        query_mask=inputs[-1],
        corner_query_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert loss.cycle_count == 0
    assert float(loss.cycle.detach()) == 0.0


def test_normal_loss_backpropagates_to_depth_residual_head() -> None:
    model = QuadRayCompletionModel(CompletionConfig(hidden_dim=32, attention_heads=4, attention_blocks=1))
    support_rays, support_rgb, support_depth, support_valid, _, _, _ = _inputs(batch=1, queries=1)
    center_ray = torch.tensor([[[0.0, 0.0, 1.0]]])
    center_uv = torch.tensor([[[0.5, 0.5]]])
    stencil_rays = torch.nn.functional.normalize(
        torch.tensor([[[[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, -0.1, 1.0], [0.0, 0.1, 1.0]]]]),
        dim=-1,
    )
    stencil_uv = torch.tensor([[[[0.4, 0.5], [0.6, 0.5], [0.5, 0.4], [0.5, 0.6]]]])
    center = model(support_rays, support_rgb, support_depth, support_valid, center_ray, center_uv, torch.ones(1, 1, dtype=torch.bool))
    stencil = model(
        support_rays,
        support_rgb,
        support_depth,
        support_valid,
        stencil_rays.reshape(1, 4, 3),
        stencil_uv.reshape(1, 4, 2),
        torch.ones(1, 4, dtype=torch.bool),
    )
    pred_normal, normal_valid = normals_from_stencil_depths(
        center.depth_z, stencil.depth_z.reshape(1, 1, 4), center_ray, stencil_rays
    )
    losses = QuadCompletionLoss(LossConfig(), StageToggles())(
        result=center,
        support_depth_z=support_depth,
        support_valid=support_valid,
        target_rgb=center.rgb.detach(),
        target_depth_z=center.depth_z.detach(),
        target_valid=torch.ones(1, 1, dtype=torch.bool),
        target_confidence=torch.ones(1, 1, dtype=torch.bool),
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        corner_query_mask=torch.zeros(1, 1, dtype=torch.bool),
        pred_normal=pred_normal,
        target_normal=torch.tensor([[[1.0, 0.0, 0.0]]]),
        normal_mask=normal_valid,
    )
    losses.total.backward()
    assert model.depth_head.weight.grad is not None
    assert torch.isfinite(model.depth_head.weight.grad).all()
    assert float(model.depth_head.weight.grad.abs().sum()) > 0.0


def test_degenerate_stencil_normal_is_masked_before_loss() -> None:
    center_depth = torch.ones(1, 1, requires_grad=True)
    stencil_depth = torch.ones(1, 1, 4, requires_grad=True)
    center_ray = torch.tensor([[[0.0, 0.0, 1.0]]])
    stencil_rays = center_ray[:, :, None, :].expand(1, 1, 4, 3).clone()

    normal, normal_valid = normals_from_stencil_depths(center_depth, stencil_depth, center_ray, stencil_rays)
    assert torch.isfinite(normal).all()
    assert not bool(normal_valid.item())
