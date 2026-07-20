from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from wide_fov_supervision_v2.config import FisheyeCameraConfig, PathConfig
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split


NYU_FX = 518.857901
NYU_FY = 519.469611
NYU_CX = 325.582449
NYU_CY = 253.736166


@dataclass(frozen=True)
class NYUFrame:
    """NYU 원본 RGB-D frame.

    Attributes:
        rgb: ``(480, 640, 3)`` uint8 RGB.
        depth_z: ``(480, 640)`` float32 meter. NYU pinhole 카메라의 z-depth.
        index: ``nyu_depth_v2_labeled.mat``의 0-based frame index.
    """

    rgb: np.ndarray
    depth_z: np.ndarray
    index: int


@dataclass(frozen=True)
class VirtualFisheyeOrientation:
    """가상 fisheye 카메라에서 NYU 카메라로 가는 회전.

    ``rotation_nyu_from_fisheye``는 column-vector 기준으로
    ``r_nyu = R @ r_fisheye``를 만족한다. OpenCV 좌표계(+x 오른쪽, +y 아래,
    +z 전방)를 사용하며 양의 yaw는 오른쪽, 양의 pitch는 위쪽을 바라본다.
    """

    name: str
    yaw_degrees: float
    pitch_degrees: float
    rotation_nyu_from_fisheye: np.ndarray


@dataclass(frozen=True)
class FisheyeDepthSamples:
    """임의의 가상 fisheye ray에서 읽은 NYU GT.

    Attributes:
        depth_z: 입력 fisheye 좌표계의 z-depth. shape은 ``rays_fisheye.shape[:-1]``.
        radial_t: 두 카메라가 같은 원점을 쓴다고 보았을 때 ray 위의 거리 ``t``.
        source_observed: ray가 NYU pinhole 전방이며 영상 내부에 투영되는지 여부.
            RGB 관측 가능 여부이므로 GT depth 유효성과 분리한다.
        gt_valid: 관측 가능하면서 bilinear depth 이웃이 모두 유효한지 여부.
        uv_nyu: NYU 영상의 pixel-center 좌표 ``(..., 2)``.
    """

    depth_z: np.ndarray
    radial_t: np.ndarray
    source_observed: np.ndarray
    gt_valid: np.ndarray
    uv_nyu: np.ndarray


@dataclass(frozen=True)
class VirtualFisheyeFrame:
    """NYU frame을 한 orientation의 가상 fisheye 영상으로 변환한 결과."""

    rgb: np.ndarray
    depth_gt_z: np.ndarray
    source_observed: np.ndarray
    gt_valid: np.ndarray
    orientation: VirtualFisheyeOrientation


class NYURawDataset(Dataset):
    """``nyu_depth_v2_labeled.mat``의 cleaned depth loader.

    HDF5 내부 ``images``는 ``(N, 3, 640, 480)``, ``depths``는
    ``(N, 640, 480)``이므로 화면 convention에 맞춰 ``(480, 640, ...)``로
    transpose해서 반환한다.
    """

    def __init__(self, paths: PathConfig, split: Literal["train", "test"] = "train", max_items: int | None = None) -> None:
        self.paths = paths
        self.split = split
        split_path = paths.nyu_split_train if split == "train" else paths.nyu_split_test
        self.indices = read_nyu_split(split_path)
        if max_items is not None:
            self.indices = self.indices[: int(max_items)]
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            if not self.paths.nyu_mat.exists():
                raise FileNotFoundError(f"NYU MAT file not found: {self.paths.nyu_mat}")
            self._h5 = h5py.File(self.paths.nyu_mat, "r")
        return self._h5

    def __getitem__(self, item: int) -> NYUFrame:
        idx = int(self.indices[item])
        h5 = self._file()
        rgb = np.asarray(h5["images"][idx], dtype=np.uint8).transpose(2, 1, 0)
        depth = np.asarray(h5["depths"][idx], dtype=np.float32).T
        return NYUFrame(rgb=rgb.copy(), depth_z=depth.copy(), index=idx)


def nyu_pinhole_rays(height: int = 480, width: int = 640) -> np.ndarray:
    """NYU pinhole RGB-D 카메라의 pixel-center 단위 ray map."""

    u, v = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5, np.arange(height, dtype=np.float32) + 0.5)
    x = (u - NYU_CX) / NYU_FX
    y = (v - NYU_CY) / NYU_FY
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1.0e-12)
    return rays.astype(np.float32)


def make_virtual_fisheye_orientation(yaw_degrees: float, pitch_degrees: float, *, name: str | None = None) -> VirtualFisheyeOrientation:
    """yaw/pitch로 ``NYU-from-fisheye`` 회전을 만든다.

    현재 기본 orientation은 한 번에 yaw 또는 pitch 하나만 사용하지만, 조합된
    값도 명확하게 정의되도록 ``R_yaw @ R_pitch`` 순서를 사용한다.
    """

    yaw = np.deg2rad(float(yaw_degrees))
    pitch = np.deg2rad(float(pitch_degrees))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotation_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    # OpenCV의 +y가 아래이므로 +pitch에서 optical axis는 화면 위(-y)로 향한다.
    rotation_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rotation = (rotation_yaw @ rotation_pitch).astype(np.float32)
    if name is None:
        name = f"yaw_{float(yaw_degrees):+g}_pitch_{float(pitch_degrees):+g}"
    return VirtualFisheyeOrientation(
        name=name,
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        rotation_nyu_from_fisheye=rotation,
    )


def orientation_for_frame(frame_index: int, orientations_degrees: Sequence[Sequence[float]]) -> tuple[int, VirtualFisheyeOrientation]:
    """MAT frame index로 center/yaw/pitch orientation을 결정적으로 순환한다.

    split 내 순번이 아니라 원본 ``frame_index``를 사용하므로 DataLoader shuffle이나
    cache 재시작 여부와 무관하게 같은 frame은 항상 같은 orientation을 갖는다.
    각 설정 원소는 ``(yaw_degrees, pitch_degrees)``이다.
    """

    if not orientations_degrees:
        raise ValueError("NYU virtual fisheye orientation set must not be empty.")
    orientation_index = int(frame_index) % len(orientations_degrees)
    values = orientations_degrees[orientation_index]
    if len(values) != 2:
        raise ValueError(f"Orientation must be (yaw, pitch), got: {values}")
    yaw, pitch = float(values[0]), float(values[1])
    if yaw == 0.0 and pitch == 0.0:
        name = "center"
    elif pitch == 0.0:
        name = f"yaw_{'right' if yaw > 0.0 else 'left'}_{abs(yaw):g}"
    elif yaw == 0.0:
        name = f"pitch_{'up' if pitch > 0.0 else 'down'}_{abs(pitch):g}"
    else:
        name = f"yaw_{yaw:+g}_pitch_{pitch:+g}"
    return orientation_index, make_virtual_fisheye_orientation(yaw, pitch, name=name)


def project_fisheye_rays_to_nyu(
    rays_fisheye: np.ndarray,
    orientation: VirtualFisheyeOrientation,
    *,
    image_height: int,
    image_width: int,
    geometry_z_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가상 fisheye ray를 NYU pinhole pixel로 투영한다.

    Returns:
        uv_nyu: ``(..., 2)`` pixel-center 좌표.
        rays_nyu: ``(..., 3)`` NYU camera-frame 단위 ray.
        source_observed: fisheye/NYU 양쪽에서 z-depth geometry가 유효하고, 투영점이
            RGB 영상의 bilinear sampling 범위 안에 있는지 나타내는 bool mask.
    """

    rays = np.asarray(rays_fisheye, dtype=np.float32)
    if rays.shape[-1] != 3:
        raise ValueError(f"rays_fisheye must end with 3 channels, got {rays.shape}")
    rotation = np.asarray(orientation.rotation_nyu_from_fisheye, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"orientation rotation must be (3,3), got {rotation.shape}")
    rays_nyu = rays @ rotation.T
    ray_norm = np.linalg.norm(rays, axis=-1)
    finite = np.isfinite(rays).all(axis=-1) & np.isfinite(rays_nyu).all(axis=-1)
    forward = (rays[..., 2] > float(geometry_z_eps)) & (rays_nyu[..., 2] > float(geometry_z_eps))

    rz = rays_nyu[..., 2]
    safe_rz = np.where(rz > float(geometry_z_eps), rz, 1.0)
    uv = np.stack(
        [NYU_FX * rays_nyu[..., 0] / safe_rz + NYU_CX, NYU_FY * rays_nyu[..., 1] / safe_rz + NYU_CY],
        axis=-1,
    ).astype(np.float32)
    # uv는 pixel-center 좌표다. uv-0.5가 OpenCV array index 범위에 들어가야 한다.
    inside = (
        (uv[..., 0] >= 0.5)
        & (uv[..., 0] <= float(image_width) - 0.5)
        & (uv[..., 1] >= 0.5)
        & (uv[..., 1] <= float(image_height) - 0.5)
    )
    observed = finite & (ray_norm > 0.5) & forward & inside
    uv[~observed] = np.nan
    return uv, rays_nyu.astype(np.float32), observed


def _remap_bilinear(image: np.ndarray, uv_pixel_center: np.ndarray, *, border_value: float | tuple[float, ...]) -> np.ndarray:
    """임의 shape의 pixel-center 좌표에서 OpenCV bilinear sampling을 수행한다.

    OpenCV remap은 한 축 길이가 매우 크면 제한에 걸리므로 query/stencil 배열은
    30,000개 단위의 1-row map으로 나눠 처리한다.
    """

    uv = np.asarray(uv_pixel_center, dtype=np.float32)
    original_shape = uv.shape[:-1]
    flat = uv.reshape(-1, 2)
    channels = () if image.ndim == 2 else (image.shape[2],)
    output = np.empty((len(flat),) + channels, dtype=image.dtype)
    for start in range(0, len(flat), 30_000):
        stop = min(start + 30_000, len(flat))
        map_x = (flat[start:stop, 0] - 0.5).reshape(1, -1)
        map_y = (flat[start:stop, 1] - 0.5).reshape(1, -1)
        sampled = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        output[start:stop] = sampled.reshape((stop - start,) + channels)
    return output.reshape(original_shape + channels)


def sample_nyu_depth_at_fisheye_rays(
    depth_nyu_z: np.ndarray,
    rays_fisheye: np.ndarray,
    orientation: VirtualFisheyeOrientation,
    *,
    geometry_z_eps: float = 1.0e-3,
) -> FisheyeDepthSamples:
    """임의의 fisheye ray에서 NYU GT를 읽고 target z-depth로 변환한다.

    두 가상 카메라는 같은 원점을 공유한다. 회전된 ray ``r_nyu``에서 읽은 NYU
    z-depth를 ``z_fisheye``로 바꾸는 식은 다음과 같다.

    ``t = z_nyu / r_nyu.z``
    ``z_fisheye = t * r_fisheye.z``

    따라서 orientation이 center가 아닐 때 NYU z-depth를 그대로 복사하면 안 된다.
    """

    depth = np.asarray(depth_nyu_z, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_nyu_z must be (H,W), got {depth.shape}")
    rays = np.asarray(rays_fisheye, dtype=np.float32)
    uv, rays_nyu, observed = project_fisheye_rays_to_nyu(
        rays,
        orientation,
        image_height=depth.shape[0],
        image_width=depth.shape[1],
        geometry_z_eps=geometry_z_eps,
    )

    source_depth_valid = np.isfinite(depth) & (depth > 0.0)
    safe_depth = np.where(source_depth_valid, depth, 0.0).astype(np.float32)
    sample_uv = np.nan_to_num(uv, nan=-1.0)
    sampled_nyu_z = _remap_bilinear(safe_depth, sample_uv, border_value=0.0).astype(np.float32)
    # 네 bilinear 이웃 중 하나라도 invalid라면 GT supervision을 끈다.
    valid_weight = _remap_bilinear(source_depth_valid.astype(np.float32), sample_uv, border_value=0.0)
    gt_valid = observed & (valid_weight >= 1.0 - 1.0e-5) & np.isfinite(sampled_nyu_z) & (sampled_nyu_z > 0.0)

    r_nyu_z = rays_nyu[..., 2]
    radial_t = np.divide(
        sampled_nyu_z,
        r_nyu_z,
        out=np.full_like(sampled_nyu_z, np.nan, dtype=np.float32),
        where=gt_valid,
    )
    depth_fisheye_z = radial_t * rays[..., 2]
    valid_final = gt_valid & np.isfinite(depth_fisheye_z) & (depth_fisheye_z > 0.0)
    radial_t[~valid_final] = np.nan
    depth_fisheye_z[~valid_final] = np.nan
    return FisheyeDepthSamples(
        depth_z=depth_fisheye_z.astype(np.float32),
        radial_t=radial_t.astype(np.float32),
        source_observed=observed.astype(bool),
        gt_valid=valid_final.astype(bool),
        uv_nyu=uv.astype(np.float32),
    )


def virtual_fisheye_from_nyu(
    frame: NYUFrame,
    target_camera: FisheyeCameraConfig,
    source_rays: np.ndarray,
    orientation: VirtualFisheyeOrientation | None = None,
) -> VirtualFisheyeFrame:
    """NYU RGB-D를 지정 orientation의 target fisheye source view로 ray-warp한다.

    ``source_observed``는 RGB 투영 가능 여부이고 ``gt_valid``는 여기에 depth 유효성을
    추가한 mask다. frozen teacher의 유효성은 이 함수에서 섞지 않고 cache 단계에서
    별도 ``teacher_valid``로 저장한다.
    """

    if orientation is None:
        orientation = make_virtual_fisheye_orientation(0.0, 0.0, name="center")
    if source_rays.shape != (target_camera.height, target_camera.width, 3):
        raise ValueError(
            f"source_rays must match target camera {(target_camera.height, target_camera.width, 3)}, got {source_rays.shape}"
        )

    depth_samples = sample_nyu_depth_at_fisheye_rays(
        frame.depth_z,
        source_rays,
        orientation,
        geometry_z_eps=target_camera.geometry_z_eps,
    )
    sample_uv = np.nan_to_num(depth_samples.uv_nyu, nan=-1.0)
    rgb = _remap_bilinear(frame.rgb, sample_uv, border_value=(0.0, 0.0, 0.0)).astype(np.uint8)
    rgb[~depth_samples.source_observed] = 0
    return VirtualFisheyeFrame(
        rgb=rgb,
        depth_gt_z=depth_samples.depth_z,
        source_observed=depth_samples.source_observed,
        gt_valid=depth_samples.gt_valid,
        orientation=orientation,
    )


class TeacherCacheDataset(Dataset):
    """NYU virtual fisheye와 frozen teacher cache를 읽는 학습 dataset.

    새 cache는 ``source_observed``, ``teacher_valid``, ``gt_valid``를 분리해서
    저장한다. ``source_valid``가 포함된 구 cache도 읽을 수 있지만, 새 학습 코드는
    세 mask의 교집합 목적을 명시적으로 선택하는 것이 안전하다.
    """

    def __init__(self, cache_dir: Path, split: Literal["train", "test"], indices: list[int], max_items: int | None = None) -> None:
        self.cache_dir = cache_dir
        self.split = split
        self.indices = indices[: int(max_items)] if max_items is not None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        idx = int(self.indices[item])
        path = self.cache_dir / self.split / f"{idx:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Teacher cache not found: {path}")
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}
