from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from wide_fov_supervision_v2.backbone.depth_anything import DepthAnythingMetricWrapper
from wide_fov_supervision_v2.datasets.nyu.dataset import NYUFrame, orientation_for_frame, virtual_fisheye_from_nyu
from wide_fov_supervision_v2.datasets.nyu.splits import read_nyu_split
from wide_fov_supervision_v2.modules.camera_geometry import build_fisheye_rays

from .config import DepthRefineV4Config, save_v4_config
from .edge_condition import depth_edge_condition, load_v2_edge_condition


def _resize_float(value: np.ndarray, size: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(np.asarray(value, dtype=np.float32), (size, size), interpolation=interpolation).astype(np.float32)


def _resize_rgb(value: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _resize_rays(rays: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    resized = _resize_float(rays, size)
    norm = np.linalg.norm(resized, axis=-1, keepdims=True)
    resized = np.divide(resized, norm, out=np.zeros_like(resized), where=norm > 1.0e-8)
    valid_resized = cv2.resize(valid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    resized[~valid_resized] = 0.0
    return resized.astype(np.float32), valid_resized


def build_v4_cache(config: DepthRefineV4Config, split: str, *, include_da: bool | None = None) -> Path:
    """Create per-frame NYU virtual-fisheye cache for V4 training/evaluation."""

    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    include_da = config.data.cache_da_v2 if include_da is None else bool(include_da)
    root = config.cache_root / split
    index_path = root / "index.json"
    if index_path.exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    save_v4_config(config, root / "config.json")

    split_path = config.base.paths.nyu_split_train if split == "train" else config.base.paths.nyu_split_test
    indices = read_nyu_split(split_path)
    limit = config.data.train_frames if split == "train" else config.data.test_frames
    if limit is not None:
        indices = indices[: int(limit)]
    rays_full = build_fisheye_rays(config.base.camera).rays_cv
    valid_full = build_fisheye_rays(config.base.camera).valid
    rays_small, valid_small = _resize_rays(rays_full, valid_full, int(config.data.image_size))
    da = DepthAnythingMetricWrapper(config.base.paths, config.base.backbone) if include_da else None
    teacher = _V2Teacher(config) if include_da and config.data.v2_teacher_for_cache else None

    entries: list[str] = []
    with h5py.File(config.base.paths.nyu_mat, "r") as h5:
        for item, frame_index in enumerate(tqdm(indices, desc=f"v4 {split} cache")):
            _, orientation = orientation_for_frame(frame_index, ((0.0, 0.0), (-55.0, 0.0), (55.0, 0.0), (0.0, -55.0), (0.0, 55.0)))
            rgb = np.asarray(h5["images"][frame_index], dtype=np.uint8).transpose(2, 1, 0)
            depth = np.asarray(h5["depths"][frame_index], dtype=np.float32).T
            virtual = virtual_fisheye_from_nyu(NYUFrame(rgb=rgb, depth_z=depth, index=frame_index), config.base.camera, rays_full, orientation)
            rgb_small = _resize_rgb(virtual.rgb, int(config.data.image_size))
            depth_small = _resize_float(virtual.depth_gt_z, int(config.data.image_size))
            source_valid = valid_small & _resize_float(virtual.gt_valid.astype(np.float32), int(config.data.image_size), cv2.INTER_NEAREST).astype(bool)
            da_features = np.zeros(
                (
                    int(config.model.da_feature_layers),
                    int(config.model.da_feature_channels),
                    1,
                    1,
                ),
                dtype=np.float32,
            )
            if da is not None:
                depth0_full, da_features = da.predict_with_features(virtual.rgb)
                depth0 = _resize_float(depth0_full, int(config.data.image_size))
            else:
                source = np.where(source_valid, depth_small, 0.0).astype(np.float32)
                depth0 = cv2.GaussianBlur(source, (0, 0), sigmaX=float(config.data.synthetic_d0_blur_sigma_px))
                depth0 = np.where(source_valid, depth0, np.nan).astype(np.float32)
            gt_condition, gt_edge_band = depth_edge_condition(
                depth_small,
                source_valid,
                threshold=float(config.data.depth_edge_log_threshold),
                band_radius=int(config.data.edge_band_radius_px),
            )
            if teacher is not None:
                condition = teacher.condition(virtual.rgb, rays_full, valid_full, depth0_full)
                condition = np.stack(
                    [_resize_float(channel, int(config.data.image_size)) for channel in condition],
                    axis=0,
                )
                edge_band = (condition[0] >= 0.20) | gt_edge_band
            else:
                condition, edge_band = gt_condition, gt_edge_band
            name = f"{item:06d}_{frame_index:05d}.npz"
            np.savez_compressed(
                root / name,
                rgb=rgb_small,
                rays=rays_small,
                valid=source_valid,
                depth0_z=depth0.astype(np.float16),
                depth_gt_z=depth_small.astype(np.float16),
                da_features=da_features.astype(np.float16),
                edge_condition=condition.astype(np.float16),
                edge_band=edge_band.astype(np.uint8),
                frame_index=np.int64(frame_index),
            )
            entries.append(name)
    index_path.write_text(
        json.dumps({"split": split, "items": entries, "include_da": include_da, "count": len(entries)}, indent=2),
        encoding="utf-8",
    )
    return root


class DepthRefineV4Dataset(Dataset):
    def __init__(self, config: DepthRefineV4Config, split: str) -> None:
        self.config = config
        self.root = config.cache_root / split
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"V4 cache not found: {index_path}. Run --mode cache first.")
        self.items = json.loads(index_path.read_text(encoding="utf-8"))["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.root / self.items[index])
        rgb = data["rgb"].astype(np.float32) / 255.0
        sample = {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1)),
            "rays": torch.from_numpy(data["rays"].astype(np.float32).transpose(2, 0, 1)),
            "valid": torch.from_numpy(data["valid"].astype(bool)[None]),
            "depth0_z": torch.from_numpy(data["depth0_z"].astype(np.float32)[None]),
            "depth_gt_z": torch.from_numpy(data["depth_gt_z"].astype(np.float32)[None]),
            "edge_condition": torch.from_numpy(data["edge_condition"].astype(np.float32)),
            "da_features": torch.from_numpy(data["da_features"].astype(np.float32)),
            "edge_band": torch.from_numpy(data["edge_band"].astype(bool)[None]),
        }
        return sample


class _V2Teacher:
    """In-memory frozen V2 rgb_context teacher used only for V4 cache creation."""

    def __init__(self, config: DepthRefineV4Config) -> None:
        from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.config import make_edge_config
        from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.model import EdgeEstimateModel
        from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.pipeline import load_checkpoint

        self.edge_config = make_edge_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = _latest_v2_context_checkpoint(config)
        if checkpoint is None:
            raise FileNotFoundError("V2 rgb_context checkpoint not found. Train edge_estimate_v2 first.")
        self.model = EdgeEstimateModel(self.edge_config.model, "rgb_context").to(self.device).eval()
        load_checkpoint(checkpoint, self.model, map_location=self.device)

    @torch.inference_mode()
    def condition(self, rgb: np.ndarray, rays: np.ndarray, valid: np.ndarray, depth0: np.ndarray) -> np.ndarray:
        from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.edge_prior import estimate_2d_edge_prior
        from wide_fov_supervision_v2.modules.prepare.edge_estimate_v2.pipeline import (
            _candidate_cells,
            _coarse_scan,
            _refine_candidates,
            _relative_da,
            _raster_queries,
        )

        edge_prior = estimate_2d_edge_prior(rgb, valid, self.edge_config.edge_prior)
        da_relative = _relative_da(depth0, valid)
        prior_valid = np.isfinite(depth0) & (depth0 > 0.0) & valid
        coarse, coarse_type = _coarse_scan(
            self.model,
            rgb,
            rays,
            valid,
            da_relative,
            edge_prior,
            self.edge_config,
            self.device,
        )
        candidates = _candidate_cells(coarse, rays, valid, self.edge_config, edge_prior)
        queries = _refine_candidates(
            self.model,
            candidates,
            coarse_type,
            rgb,
            rays,
            valid,
            da_relative,
            edge_prior,
            depth0,
            prior_valid,
            self.edge_config,
            self.device,
        )
        selected = (
            (queries["edge_probability"] >= self.edge_config.inference.query_edge_threshold)
            & (queries["confidence"] >= self.edge_config.inference.confidence_threshold)
            & queries["prior_depth_valid"]
        )
        shape = valid.shape
        probability = _raster_queries(shape, queries["source_uv"], queries["edge_probability"], queries["confidence"], selected)
        confidence = _raster_queries(shape, queries["source_uv"], queries["confidence"], queries["confidence"], selected)
        near = _raster_queries(shape, queries["source_uv"], queries["depth_near_z"], queries["confidence"], selected)
        far = _raster_queries(shape, queries["source_uv"], queries["depth_far_z"], queries["confidence"], selected)
        with _temporary_npz_outputs(probability, confidence, near, far) as tmp_dir:
            return load_v2_edge_condition(tmp_dir, shape, depth0)


class _temporary_npz_outputs:
    def __init__(self, probability: np.ndarray, confidence: np.ndarray, near: np.ndarray, far: np.ndarray) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        np.save(self.path / "edge_probability.npy", probability.astype(np.float32))
        np.save(self.path / "edge_confidence.npy", confidence.astype(np.float32))
        np.save(self.path / "edge_depth_near_z.npy", near.astype(np.float32))
        np.save(self.path / "edge_depth_far_z.npy", far.astype(np.float32))

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


def _latest_v2_context_checkpoint(config: DepthRefineV4Config) -> Path | None:
    root = config.base.paths.outputs / "edge_estimate" / "v2" / "train" / "rgb_context"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None
