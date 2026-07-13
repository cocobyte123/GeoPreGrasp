from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from oc_pregrasp.data.features import pca_short_canonical_inputs


@dataclass(frozen=True)
class RolloutSampleRef:
    path: Path
    row: int
    object_id: str


def iter_rollout_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.expanduser().resolve().glob("*/batch_*.npz"))


def angle_features(yaw_deg, tilt_deg, pitch_deg) -> np.ndarray:
    angles = np.deg2rad(
        np.stack([yaw_deg, tilt_deg, pitch_deg], axis=-1).astype(np.float32)
    )
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(
        np.float32
    )


class YawPregraspRolloutDataset(Dataset):
    """Samples from rollout npz files.

    Each item is one candidate label:
        object point cloud, candidate angle feature, object pose, success label.
    """

    def __init__(self, rollout_root: str | Path, feature_mode: str = "raw"):
        self.rollout_root = Path(rollout_root).expanduser().resolve()
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in ("raw", "pca_short"):
            raise ValueError("feature_mode must be raw or pca_short")
        self.files = list(iter_rollout_files(self.rollout_root))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.npz files under {self.rollout_root}")

        self.samples: List[RolloutSampleRef] = []
        self.object_point_clouds: Dict[str, np.ndarray] = {}
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                object_id = str(np.asarray(data["object_id"]).item())
                if object_id not in self.object_point_clouds:
                    self.object_point_clouds[object_id] = data[
                        "object_point_cloud_object"
                    ].astype(np.float32)
                count = int(data["success"].reshape(-1).shape[0])
                self.samples.extend(
                    RolloutSampleRef(path=path, row=row, object_id=object_id)
                    for row in range(count)
                )
        self._cache_path: Path | None = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path):
        if self._cache_path != path:
            if self._cache_data is not None:
                self._cache_data.close()
            self._cache_path = path
            self._cache_data = np.load(path, allow_pickle=False)
        return self._cache_data

    def __getitem__(self, index: int):
        sample = self.samples[index]
        data = self._load(sample.path)
        row = sample.row

        yaw = data["yaw_deg"].reshape(-1)[row]
        tilt = data["tilt_deg"].reshape(-1)[row]
        pitch = data["pitch_deg"].reshape(-1)[row]
        angles = angle_features(
            np.asarray(yaw), np.asarray(tilt), np.asarray(pitch)
        ).reshape(-1)
        angle_deg = np.asarray([yaw, tilt, pitch], dtype=np.float32)

        pose = data["object_pose_world"].reshape(-1, 7)[row].astype(np.float32)
        point_cloud = self.object_point_clouds[sample.object_id]
        pose_feature = pose
        if self.feature_mode == "pca_short":
            point_cloud, pose_feature = pca_short_canonical_inputs(point_cloud, pose)
        label = np.asarray(data["success"].reshape(-1)[row], dtype=np.float32)
        return {
            "object_id": sample.object_id,
            "point_cloud": torch.from_numpy(point_cloud),
            "angle": torch.from_numpy(angles),
            "angle_deg": torch.from_numpy(angle_deg),
            "object_pose": torch.from_numpy(pose_feature),
            "label": torch.from_numpy(label.reshape(1)),
        }

    def close(self) -> None:
        if self._cache_data is not None:
            self._cache_data.close()
            self._cache_data = None
            self._cache_path = None
