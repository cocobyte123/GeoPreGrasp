from __future__ import annotations

import numpy as np


def quat_apply_xyzw(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    p = np.asarray(points, dtype=np.float32)
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz.reshape(1, 3), p)
    return p + q_w * t + np.cross(q_xyz.reshape(1, 3), t)


def quat_z_yaw_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def pca_short_reference_yaw(world_rel_points: np.ndarray, fallback_yaw: float) -> float:
    xy = world_rel_points[:, :2].astype(np.float32)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(int(xy.shape[0]) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 0]
    if axis[0] < 0.0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0.0):
        axis = -axis
    if float(eigvals[1] - eigvals[0]) <= 1e-8:
        return float(fallback_yaw)
    return float(np.arctan2(axis[1], axis[0]))


def rotate_z(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    result = points.astype(np.float32).copy()
    x = result[:, 0].copy()
    y = result[:, 1].copy()
    result[:, 0] = c * x - s * y
    result[:, 1] = s * x + c * y
    return result


def pca_short_canonical_inputs(
    object_point_cloud_object: np.ndarray,
    object_pose_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build scorer inputs aligned with pca_short yaw labels."""

    local_points = np.asarray(object_point_cloud_object, dtype=np.float32)
    pose = np.asarray(object_pose_world, dtype=np.float32)
    quat = pose[3:7]
    object_yaw = quat_z_yaw_xyzw(quat)
    world_rel = quat_apply_xyzw(quat, local_points)
    reference_yaw = pca_short_reference_yaw(world_rel, object_yaw)

    centered = world_rel - world_rel.mean(axis=0, keepdims=True)
    canonical = rotate_z(centered, -reference_yaw).astype(np.float32)
    extents = canonical.max(axis=0) - canonical.min(axis=0)
    short_extent = float(extents[0])
    long_extent = float(extents[1])
    height = float(extents[2])
    anisotropy = (long_extent - short_extent) / max(long_extent + short_extent, 1e-6)
    rel_yaw = object_yaw - reference_yaw
    features = np.asarray(
        [
            short_extent,
            long_extent,
            height,
            anisotropy,
            float(pose[2]),
            np.sin(rel_yaw),
            np.cos(rel_yaw),
        ],
        dtype=np.float32,
    )
    return canonical, features

