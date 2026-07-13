"""Geometry helpers for object-frame rollout processing."""

from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f'quat must end with 4 values, got shape {quat.shape}')
    x, y, z, w = np.moveaxis(quat, -1, 0)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    norm = np.maximum(norm, 1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def palm_world_to_object(
    object_pose_world: np.ndarray,
    palm_center_world: np.ndarray,
    palm_toward_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_pose_world = np.asarray(object_pose_world, dtype=np.float64)
    palm_center_world = np.asarray(palm_center_world, dtype=np.float64)
    palm_toward_world = np.asarray(palm_toward_world, dtype=np.float64)
    object_pos = object_pose_world[:, :3]
    object_rot = quat_xyzw_to_matrix(object_pose_world[:, 3:7])
    palm_center_obj = np.einsum('bij,bj->bi', np.swapaxes(object_rot, 1, 2), palm_center_world - object_pos)
    palm_toward_obj = np.einsum('bij,bj->bi', np.swapaxes(object_rot, 1, 2), palm_toward_world)
    return palm_center_obj.astype(np.float32), normalize(palm_toward_obj).astype(np.float32)


def points_world_to_object(points_world: np.ndarray, object_pose_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(object_pose_world, dtype=np.float64).reshape(7)
    object_pos = pose[:3]
    object_rot = quat_xyzw_to_matrix(pose[3:7])
    return ((points - object_pos.reshape(1, 3)) @ object_rot).astype(np.float32)

