"""Label construction helpers for spherical pregrasp fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from oc_pregrasp.field.sphere_template import SphereTemplate, one_ring_direction_weights


@dataclass(frozen=True)
class DirectionLabel:
    direction: np.ndarray
    projection: np.ndarray
    weights: np.ndarray
    nearest_id: int
    nearest_angle_deg: float
    ray_hit: bool

    @property
    def support_ids(self) -> np.ndarray:
        return np.flatnonzero(self.weights > 0.0).astype(np.int64)


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    return vector / max(float(np.linalg.norm(vector)), eps)


def ray_sphere_projection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    radius: float,
    sphere_center: Optional[np.ndarray] = None,
):
    """Project a ray onto a sphere and return the first positive intersection.

    If the ray misses, return the radial projection of the ray direction as a
    fallback. For our pregrasp labels this keeps preprocessing robust while
    exposing the miss through the returned boolean flag.
    """
    center = np.zeros(3, dtype=np.float32) if sphere_center is None else np.asarray(sphere_center, dtype=np.float32)
    origin = np.asarray(ray_origin, dtype=np.float32).reshape(3) - center.reshape(3)
    direction = normalize(ray_direction)
    radius = float(radius)

    b = float(np.dot(origin, direction))
    c = float(np.dot(origin, origin) - radius * radius)
    discriminant = b * b - c
    if discriminant >= 0.0:
        sqrt_disc = float(np.sqrt(discriminant))
        candidates = [t for t in (-b - sqrt_disc, -b + sqrt_disc) if t >= 0.0]
        if candidates:
            t = min(candidates)
            projection = center + origin + direction * t
            return projection.astype(np.float32), t, True

    fallback = center + direction * radius
    return fallback.astype(np.float32), None, False


def hand_back_direction_label(
    palm_center: np.ndarray,
    palm_toward: np.ndarray,
    template: SphereTemplate,
    pregrasp_radius: float,
    object_center: Optional[np.ndarray] = None,
    sigma_deg: float = 8.0,
) -> DirectionLabel:
    """Map one grasp to geodesic nearest + 1-ring direction labels.

    `palm_toward` points out of the palm toward the object in the current hand
    model convention. The pregrasp retreat direction is the opposite direction,
    i.e. the hand-back direction.
    """
    center = np.zeros(3, dtype=np.float32) if object_center is None else np.asarray(object_center, dtype=np.float32)
    hand_back = -normalize(palm_toward)
    projection, _, ray_hit = ray_sphere_projection(
        ray_origin=palm_center,
        ray_direction=hand_back,
        radius=pregrasp_radius,
        sphere_center=center,
    )
    direction = normalize(projection - center)
    weights, nearest_id, nearest_angle = one_ring_direction_weights(
        direction,
        template,
        sigma_deg=sigma_deg,
    )
    return DirectionLabel(
        direction=direction,
        projection=projection,
        weights=weights,
        nearest_id=nearest_id,
        nearest_angle_deg=nearest_angle,
        ray_hit=ray_hit,
    )


def rotation_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic SO(3) distance in degrees."""
    rotation_a = np.asarray(rotation_a, dtype=np.float32).reshape(3, 3)
    rotation_b = np.asarray(rotation_b, dtype=np.float32).reshape(3, 3)
    relative = rotation_a.T @ rotation_b
    cos_angle = (float(np.trace(relative)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))

