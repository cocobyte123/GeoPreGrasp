"""Build dense M/T/P labels from palm-back direction observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from oc_pregrasp.field.label_mapping import DirectionLabel, hand_back_direction_label, normalize
from oc_pregrasp.field.sphere_template import SphereTemplate


@dataclass(frozen=True)
class GraspFieldLabel:
    target_field: np.ndarray
    confidence: np.ndarray
    nearest_id: int
    nearest_angle_deg: float
    ray_hit: bool
    support_ids: np.ndarray


def stable_tangent_frame(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two deterministic tangent axes for a sphere direction."""
    n = normalize(direction)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    tilt_axis = np.cross(ref, n)
    tilt_axis = normalize(tilt_axis)
    pitch_axis = np.cross(n, tilt_axis)
    pitch_axis = normalize(pitch_axis)
    return tilt_axis.astype(np.float32), pitch_axis.astype(np.float32)


def local_angular_offsets_deg(reference_dir: np.ndarray, target_dir: np.ndarray) -> Tuple[float, float]:
    """Express target direction as signed tilt/pitch offsets around a reference direction."""
    reference = normalize(reference_dir)
    target = normalize(target_dir)
    tilt_axis, pitch_axis = stable_tangent_frame(reference)
    forward = float(np.clip(np.dot(target, reference), -1.0, 1.0))
    tilt = float(np.degrees(np.arctan2(float(np.dot(target, tilt_axis)), forward)))
    pitch = float(np.degrees(np.arctan2(float(np.dot(target, pitch_axis)), forward)))
    return tilt, pitch


def gaussian_angle_weights(delta_deg: float, angle_bins_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    bins = np.asarray(angle_bins_deg, dtype=np.float32)
    sigma = max(float(sigma_deg), 1e-6)
    weights = np.exp(-0.5 * ((bins - float(delta_deg)) / sigma) ** 2).astype(np.float32)
    total = float(weights.sum())
    if total <= 1e-8:
        nearest = int(np.argmin(np.abs(bins - float(delta_deg))))
        weights[:] = 0.0
        weights[nearest] = 1.0
        return weights
    return weights / total


def field_label_from_direction(
    direction_label: DirectionLabel,
    template: SphereTemplate,
    local_angle_sigma_deg: float = 12.0,
) -> GraspFieldLabel:
    """Convert one direction label to a dense [M,T,P] distribution."""
    m_count, t_count, p_count = template.shape
    field = np.zeros((m_count, t_count, p_count), dtype=np.float32)
    support_ids = direction_label.support_ids
    for m_id in support_ids:
        m_weight = float(direction_label.weights[int(m_id)])
        if m_weight <= 0.0:
            continue
        tilt_delta, pitch_delta = local_angular_offsets_deg(template.dirs[int(m_id)], direction_label.direction)
        tilt_weights = gaussian_angle_weights(tilt_delta, template.tilt_angles, local_angle_sigma_deg)
        pitch_weights = gaussian_angle_weights(pitch_delta, template.pitch_angles, local_angle_sigma_deg)
        field[int(m_id)] += m_weight * np.outer(tilt_weights, pitch_weights).astype(np.float32)
    return GraspFieldLabel(
        target_field=field,
        confidence=field.copy(),
        nearest_id=direction_label.nearest_id,
        nearest_angle_deg=direction_label.nearest_angle_deg,
        ray_hit=direction_label.ray_hit,
        support_ids=support_ids,
    )


def hand_back_field_label(
    palm_center: np.ndarray,
    palm_toward: np.ndarray,
    template: SphereTemplate,
    pregrasp_radius: float,
    object_center=None,
    direction_sigma_deg: float = 8.0,
    local_angle_sigma_deg: float = 12.0,
) -> GraspFieldLabel:
    direction_label = hand_back_direction_label(
        palm_center=palm_center,
        palm_toward=palm_toward,
        template=template,
        pregrasp_radius=pregrasp_radius,
        object_center=object_center,
        sigma_deg=direction_sigma_deg,
    )
    return field_label_from_direction(
        direction_label=direction_label,
        template=template,
        local_angle_sigma_deg=local_angle_sigma_deg,
    )


def normalize_field(field: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    total = float(field.sum())
    if total <= eps:
        return field.copy()
    return (field / total).astype(np.float32)
