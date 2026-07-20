from __future__ import annotations

import torch

from wide_fov_supervision_v2.config import LossConfig, RefinerConfig, StageToggles
from wide_fov_supervision_v2.loss.partial_loss import PartialLossMasks, PartialSupervisionLoss
from wide_fov_supervision_v2.modules.query_geometry import normals_from_stencil_depths, query_stencil_rays
from wide_fov_supervision_v2.modules.refiner import RayAwareQueryRefiner, scale_pixel_center_uv


def test_stride_pixel_center_coordinate_conversion() -> None:
    uv = torch.tensor([[[0.5, 0.5], [2.5, 4.5], [6.5, 10.5]]])
    expected_stride2 = torch.tensor([[[0.5, 0.5], [1.5, 2.5], [3.5, 5.5]]])
    expected_stride4 = torch.tensor([[[0.5, 0.5], [1.0, 1.5], [2.0, 3.0]]])
    torch.testing.assert_close(scale_pixel_center_uv(uv, 2), expected_stride2)
    torch.testing.assert_close(scale_pixel_center_uv(uv, 4), expected_stride4)


def test_shared_encode_decode_matches_forward_and_backpropagates() -> None:
    """공유 pyramid decode가 forward와 같고 encoder까지 gradient를 전달한다."""

    torch.manual_seed(7)
    model = RayAwareQueryRefiner(RefinerConfig(base_channels=16, hidden_dim=32))
    final_layer = model.decoder[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    # 기본 zero-init 성질은 별도 테스트가 검증한다. 여기서는 encoder gradient까지
    # 확인하기 위해 decoder가 sampled source feature에 민감하도록 작은 값을 준다.
    with torch.no_grad():
        final_layer.weight.normal_(mean=0.0, std=0.01)

    b, h, w, q = 1, 24, 24, 5
    rgb = torch.rand(b, 3, h, w)
    depth0 = torch.rand(b, 1, h, w) + 1.0
    normal0 = torch.zeros(b, 3, h, w)
    normal0[:, 2] = -1.0
    rays = torch.zeros(b, 3, h, w)
    rays[:, 2] = 1.0
    valid = torch.ones(b, 1, h, w)
    query_rays = torch.zeros(b, q, 3)
    query_rays[..., 2] = 1.0
    uv = torch.rand(b, q, 2) * 14.0 + 4.5
    relative_uv = torch.rand(b, q, 2)
    sampling_features = torch.rand(b, q, 3)
    observed = torch.ones(b, q)

    with torch.no_grad():
        forward_result = model(
            rgb,
            depth0,
            normal0,
            rays,
            valid,
            query_rays,
            uv,
            relative_uv,
            sampling_features,
            observed,
        )

    source_features = model.encode_source(rgb, depth0, normal0, rays, valid)
    shared_result = model.decode_queries(
        depth0,
        normal0,
        rays,
        valid,
        source_features,
        query_rays,
        uv,
        relative_uv,
        sampling_features,
        observed,
    )
    for key in forward_result:
        torch.testing.assert_close(shared_result[key], forward_result[key], equal_nan=True)

    torch.nan_to_num(shared_result["depth_final_z"]).sum().backward()
    stem_conv = model.stem[0]
    assert isinstance(stem_conv, torch.nn.Conv2d)
    assert stem_conv.weight.grad is not None
    assert torch.count_nonzero(stem_conv.weight.grad).item() > 0
    assert final_layer.weight.grad is not None


def test_zero_residual_depth_equals_d0() -> None:
    torch.manual_seed(0)
    cfg = RefinerConfig(base_channels=16, hidden_dim=32)
    model = RayAwareQueryRefiner(cfg)
    b, h, w, q = 1, 32, 32, 8
    rgb = torch.rand(b, 3, h, w)
    depth0 = torch.ones(b, 1, h, w) * 2.0
    normal0 = torch.zeros(b, 3, h, w)
    normal0[:, 2] = -1.0
    rays = torch.zeros(b, 3, h, w)
    rays[:, 2] = 1.0
    valid = torch.ones(b, 1, h, w)
    query_rays = torch.zeros(b, q, 3)
    query_rays[..., 2] = 1.0
    uv = torch.rand(b, q, 2) * 20.0 + 6.0
    rel = torch.zeros(b, q, 2)
    sampling_features = torch.zeros(b, q, 3)
    sampling_features[..., 0] = 1.1
    sampling_features[..., 1] = 2.0
    sampling_features[..., 2] = 3.0
    obs = torch.ones(b, q)
    out = model(rgb, depth0, normal0, rays, valid, query_rays, uv, rel, sampling_features, obs)
    assert torch.allclose(out["depth_final_z"], torch.ones_like(out["depth_final_z"]) * 2.0, atol=1.0e-5)


def test_stencil_plane_normal_is_differentiable() -> None:
    center_rays = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    stencil = query_stencil_rays(center_rays, 0.01)
    center_depth = torch.ones(1, 1, requires_grad=True)
    stencil_depth = torch.ones(1, 1, 4, requires_grad=True)
    normal, valid = normals_from_stencil_depths(center_depth, stencil_depth, center_rays, stencil)
    loss = normal[..., 2].sum()
    loss.backward()
    assert valid.item()
    assert stencil_depth.grad is not None


def test_partial_mask_empty_returns_connected_zero() -> None:
    pred = torch.ones(1, 4, requires_grad=True)
    loss_fn = PartialSupervisionLoss(LossConfig(), StageToggles())
    result = loss_fn(
        pred_depth_z=pred,
        target_depth_z=torch.ones_like(pred),
        pred_normal=torch.zeros(1, 4, 3),
        target_normal=torch.zeros(1, 4, 3),
        query_rays=torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(1, 4, 1),
        source_rays_query=torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(1, 4, 1),
        depth0_query_z=torch.ones_like(pred),
        delta_log_depth=torch.zeros_like(pred),
        masks=PartialLossMasks(
            depth=torch.zeros(1, 4, dtype=torch.bool),
            normal=torch.zeros(1, 4, dtype=torch.bool),
            radial=torch.zeros(1, 4, dtype=torch.bool),
            delta=torch.zeros(1, 4, dtype=torch.bool),
        ),
    )
    result.total.backward()
    assert result.valid_count == 0
    assert result.depth_valid_count == 0
    assert result.normal_valid_count == 0
    assert result.radial_valid_count == 0
    assert result.delta_valid_count == 0
    assert pred.grad is not None


def test_normal_mask_does_not_block_center_depth_loss() -> None:
    """stencil이 무효여도 독립 depth mask가 켜진 중심 query는 감독한다."""

    pred = torch.tensor([[2.0, 2.0]], requires_grad=True)
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    loss_fn = PartialSupervisionLoss(LossConfig(), StageToggles())
    result = loss_fn(
        pred_depth_z=pred,
        target_depth_z=torch.tensor([[4.0, 4.0]]),
        pred_normal=torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]]),
        target_normal=torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]]),
        query_rays=ray,
        source_rays_query=ray,
        depth0_query_z=torch.ones_like(pred) * 2.0,
        delta_log_depth=torch.zeros_like(pred),
        masks=PartialLossMasks(
            depth=torch.ones(1, 2, dtype=torch.bool),
            normal=torch.zeros(1, 2, dtype=torch.bool),
            radial=torch.tensor([[True, False]]),
            delta=torch.tensor([[False, True]]),
        ),
    )
    assert result.depth_valid_count == 2
    assert result.normal_valid_count == 0
    assert result.radial_valid_count == 1
    assert result.delta_valid_count == 1
    assert result.depth.item() > 0.0


def test_invalid_nan_values_do_not_create_nan_gradients() -> None:
    """mask 밖 NaN은 loss 값과 gradient 모두에 영향을 주지 않아야 한다."""

    pred_depth = torch.tensor([[2.0, float("nan"), 3.0]], requires_grad=True)
    pred_normal = torch.tensor(
        [[[0.1, 0.0, -0.99], [float("nan"), 0.0, 1.0], [0.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    rays = torch.tensor(
        [[[0.0, 0.0, 1.0], [float("nan"), 0.0, 1.0], [0.0, 0.0, -1.0]]]
    )
    valid = torch.tensor([[True, False, False]])
    result = PartialSupervisionLoss(LossConfig(), StageToggles())(
        pred_depth_z=pred_depth,
        target_depth_z=torch.tensor([[2.5, float("nan"), 4.0]]),
        pred_normal=pred_normal,
        target_normal=torch.tensor(
            [[[0.0, 0.0, -1.0], [float("nan"), 0.0, 1.0], [0.0, 0.0, -1.0]]]
        ),
        query_rays=rays,
        source_rays_query=rays,
        depth0_query_z=torch.tensor([[2.0, float("nan"), 3.0]]),
        delta_log_depth=torch.zeros_like(pred_depth),
        masks=PartialLossMasks(depth=valid, normal=valid, radial=valid, delta=valid),
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert pred_depth.grad is not None
    assert pred_normal.grad is not None
    assert torch.isfinite(pred_depth.grad).all()
    assert torch.isfinite(pred_normal.grad).all()


def test_partial_normal_loss_backpropagates_to_refiner() -> None:
    """N* cosine loss의 gradient가 Refiner 마지막 depth residual layer에 도달한다."""

    torch.manual_seed(1)
    model = RayAwareQueryRefiner(RefinerConfig(base_channels=16, hidden_dim=32))
    b, h, w = 1, 32, 32
    rgb = torch.rand(b, 3, h, w)
    depth0 = torch.ones(b, 1, h, w) * 2.0
    normal0 = torch.zeros(b, 3, h, w)
    normal0[:, 2] = -1.0
    source_rays = torch.zeros(b, 3, h, w)
    source_rays[:, 2] = 1.0
    source_valid = torch.ones(b, 1, h, w)

    center_ray = torch.tensor([[[0.0, 0.0, 1.0]]])
    stencil_rays = query_stencil_rays(center_ray, 0.01)
    center_uv = torch.tensor([[[16.5, 16.5]]])
    stencil_uv = torch.tensor([[[15.5, 16.5], [17.5, 16.5], [16.5, 15.5], [16.5, 17.5]]])
    center_features = torch.ones(1, 1, 3)
    stencil_features = center_features.repeat_interleave(4, dim=1)
    center = model(
        rgb,
        depth0,
        normal0,
        source_rays,
        source_valid,
        center_ray,
        center_uv,
        torch.zeros(1, 1, 2),
        center_features,
        torch.ones(1, 1),
    )
    stencil = model(
        rgb,
        depth0,
        normal0,
        source_rays,
        source_valid,
        stencil_rays.reshape(1, 4, 3),
        stencil_uv,
        torch.zeros(1, 4, 2),
        stencil_features,
        torch.ones(1, 4),
    )
    normal_star, normal_valid = normals_from_stencil_depths(
        center["depth_final_z"],
        stencil["depth_final_z"].reshape(1, 1, 4),
        center_ray,
        stencil_rays,
    )

    false_mask = torch.zeros(1, 1, dtype=torch.bool)
    result = PartialSupervisionLoss(LossConfig(), StageToggles())(
        pred_depth_z=center["depth_final_z"],
        target_depth_z=None,
        pred_normal=normal_star,
        target_normal=torch.tensor([[[0.5, 0.0, -0.8660254]]]),
        query_rays=center_ray,
        source_rays_query=center["source_ray_query"],
        depth0_query_z=center["depth0_query_z"],
        delta_log_depth=center["delta_log_depth"],
        masks=PartialLossMasks(
            depth=false_mask,
            normal=normal_valid,
            radial=false_mask,
            delta=false_mask,
        ),
    )
    result.total.backward()
    final_layer = model.decoder[-1]
    assert result.normal_valid_count == 1
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad).item() > 0
