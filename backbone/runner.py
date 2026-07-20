from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wide_fov_supervision_v2.backbone.depth_anything import DepthAnythingMetricWrapper
from wide_fov_supervision_v2.backbone.dsine import DSINEWrapper
from wide_fov_supervision_v2.backbone.tangent_views import build_tangent_views, fuse_tangent_normals, fuse_tangent_predictions
from wide_fov_supervision_v2.config import BackboneConfig, FisheyeCameraConfig, PathConfig


@dataclass
class BackbonePrediction:
    """branch별 teacher prediction."""

    branch: str
    depth0_z: np.ndarray
    normal0: np.ndarray
    valid: np.ndarray
    metadata: dict


class BackboneRunner:
    """DA-V2와 DSINE을 순차 실행하는 helper."""

    def __init__(self, paths: PathConfig, config: BackboneConfig, camera: FisheyeCameraConfig) -> None:
        self.paths = paths
        self.config = config
        self.camera = camera
        # external NPY mode에서 DA-V2 checkpoint/GPU memory를 사용하지 않도록
        # foundation model wrapper를 실제 사용 시점에 지연 생성한다.
        self._depth_model: DepthAnythingMetricWrapper | None = None
        self._normal_model: DSINEWrapper | None = None

    @property
    def depth_model(self) -> DepthAnythingMetricWrapper:
        if self._depth_model is None:
            self._depth_model = DepthAnythingMetricWrapper(self.paths, self.config)
        return self._depth_model

    @property
    def normal_model(self) -> DSINEWrapper:
        if self._normal_model is None:
            self._normal_model = DSINEWrapper(self.paths, self.config)
        return self._normal_model

    def run_direct(
        self,
        rgb: np.ndarray,
        *,
        depth_override_z: np.ndarray | None = None,
        depth_metadata: dict | None = None,
    ) -> BackbonePrediction:
        """원본 fisheye RGB를 그대로 DA-V2/DSINE에 넣는 baseline branch."""

        if depth_override_z is None:
            depth = self.depth_model.predict(rgb)
            source_metadata = {"depth_source": "da_v2"}
        else:
            depth = np.asarray(depth_override_z, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"depth override shape {depth.shape} does not match RGB shape {rgb.shape[:2]}")
            source_metadata = dict(depth_metadata or {"depth_source": "external_npy"})
        normal = self.normal_model.predict(rgb, camera=self.camera)
        valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(normal).all(axis=-1)
        return BackbonePrediction(
            branch="direct",
            depth0_z=depth.astype(np.float32),
            normal0=normal.astype(np.float32),
            valid=valid,
            metadata={
                "note": "fisheye RGB direct input; pinhole-pretrained teacher baseline",
                **source_metadata,
            },
        )

    def run_tangent(
        self,
        rgb: np.ndarray,
        source_rays: np.ndarray,
        *,
        depth_override_z: np.ndarray | None = None,
        depth_metadata: dict | None = None,
    ) -> BackbonePrediction:
        """5개 tangent pinhole view에서 teacher를 실행하고 source fisheye 좌표로 융합한다."""

        views = build_tangent_views(rgb, self.camera, self.config)
        depth_by_view: dict[str, np.ndarray] = {}
        normal_by_view: dict[str, np.ndarray] = {}
        for view in views:
            if depth_override_z is None:
                depth_by_view[view.name] = self.depth_model.predict(view.rgb)
            normal_by_view[view.name] = self.normal_model.predict(view.rgb, intrinsics=view.intrinsics)
        if depth_override_z is None:
            depth, normal, valid = fuse_tangent_predictions(source_rays, views, depth_by_view, normal_by_view)
            source_metadata = {"depth_source": "da_v2"}
        else:
            depth = np.asarray(depth_override_z, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"depth override shape {depth.shape} does not match RGB shape {rgb.shape[:2]}")
            normal, normal_valid = fuse_tangent_normals(source_rays, views, normal_by_view)
            valid = normal_valid & np.isfinite(depth) & (depth > 0.0)
            source_metadata = dict(depth_metadata or {"depth_source": "external_npy"})
        return BackbonePrediction(
            branch="tangent",
            depth0_z=depth,
            normal0=normal,
            valid=valid,
            metadata={
                "view_names": [view.name for view in views],
                "tangent_resolution": int(self.config.tangent_resolution),
                "tangent_fov_degrees": float(self.config.tangent_fov_degrees),
                "tangent_polar_degrees": float(self.config.tangent_polar_degrees),
                **source_metadata,
            },
        )
