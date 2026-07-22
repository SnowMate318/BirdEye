from __future__ import annotations

import numpy as np


CORNER_RELATIVE_UV = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


def bilinear_weights(relative_uv: np.ndarray) -> np.ndarray:
    """상대 좌표 ``(..., 2)``의 네 corner bilinear 가중치를 반환한다.

    corner 순서는 전체 코드에서 ``p00, p10, p11, p01``로 고정한다. 따라서
    반환 shape은 ``(..., 4)``이고 네 가중치의 합은 1이다.
    """

    uv = np.asarray(relative_uv, dtype=np.float32)
    if uv.shape[-1] != 2:
        raise ValueError(f"relative_uv must end with two coordinates, got {uv.shape}")
    u, v = uv[..., 0], uv[..., 1]
    return np.stack(
        [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v],
        axis=-1,
    ).astype(np.float32)


def bilinear_quad_map(corners_xy: np.ndarray, relative_uv: np.ndarray) -> np.ndarray:
    """일반 사각형의 상대 좌표를 영상 좌표로 bilinear mapping한다.

    Args:
        corners_xy: ``(..., 4, 2)``. ``p00,p10,p11,p01`` 순서의 array-index
            좌표 ``(x,y)``. pixel center ray를 만들 때는 각 좌표에 0.5를 더한다.
        relative_uv: ``(..., Q, 2)`` 또는 ``(Q,2)`` 상대 좌표.
    """

    corners = np.asarray(corners_xy, dtype=np.float32)
    uv = np.asarray(relative_uv, dtype=np.float32)
    if corners.shape[-2:] != (4, 2):
        raise ValueError(f"corners_xy must end with (4,2), got {corners.shape}")
    weights = bilinear_weights(uv)
    return np.einsum("...qk,...kc->...qc", weights, corners).astype(np.float32)


def bilinear_quad_jacobian(corners_xy: np.ndarray, relative_uv: np.ndarray) -> np.ndarray:
    """Bilinear quad mapping의 signed Jacobian determinant를 계산한다.

    양수이면 ``u``와 ``v`` 방향의 국소 순서가 뒤집히지 않는다. 학습 manifest는
    내부 query에서 이 값이 모두 양수인 quad만 사용한다.
    """

    p = np.asarray(corners_xy, dtype=np.float32)
    uv = np.asarray(relative_uv, dtype=np.float32)
    if p.shape != (4, 2):
        raise ValueError(f"corners_xy must be (4,2), got {p.shape}")
    u, v = uv[..., 0], uv[..., 1]
    p00, p10, p11, p01 = p
    d_du = (1.0 - v)[..., None] * (p10 - p00) + v[..., None] * (p11 - p01)
    d_dv = (1.0 - u)[..., None] * (p01 - p00) + u[..., None] * (p11 - p10)
    return (d_du[..., 0] * d_dv[..., 1] - d_du[..., 1] * d_dv[..., 0]).astype(np.float32)


def _cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    bc = c - b
    return float(ab[0] * bc[1] - ab[1] * bc[0])


def is_convex_quad(corners_xy: np.ndarray, *, min_area: float = 1.0e-4) -> bool:
    """사각형이 self-intersection 없는 strictly convex polygon인지 검사한다.

    입력 순서는 ``p00→p10→p11→p01``이어야 한다. 네 연속 edge의 cross product가
    모두 같은 부호이고 shoelace area가 양수인 경우만 허용한다.
    """

    p = np.asarray(corners_xy, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return False
    crosses = np.array([_cross_2d(p[i], p[(i + 1) % 4], p[(i + 2) % 4]) for i in range(4)])
    if np.any(np.abs(crosses) <= min_area):
        return False
    area2 = float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))
    return bool(area2 > 2.0 * min_area and np.all(crosses > 0.0))


def quad_is_valid(
    corners_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    jacobian_grid_size: int = 5,
) -> bool:
    """영상 내부·convex·양의 Jacobian 조건을 한 번에 검사한다."""

    p = np.asarray(corners_xy, dtype=np.float32)
    if not is_convex_quad(p):
        return False
    if (
        np.any(p[:, 0] < 0.0)
        or np.any(p[:, 0] > image_width - 1.0)
        or np.any(p[:, 1] < 0.0)
        or np.any(p[:, 1] > image_height - 1.0)
    ):
        return False
    axis = np.linspace(0.0, 1.0, jacobian_grid_size, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    query = np.stack([u, v], axis=-1)
    return bool(np.all(bilinear_quad_jacobian(p, query) > 1.0e-5))


def pinhole_rays_from_xy(
    xy: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """array-index ``(x,y)``를 OpenCV camera 좌표계 단위 ray로 바꾼다."""

    coords = np.asarray(xy, dtype=np.float32)
    x = (coords[..., 0] + 0.5 - float(cx)) / float(fx)
    y = (coords[..., 1] + 0.5 - float(cy)) / float(fy)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1.0e-12)
    return rays.astype(np.float32)
