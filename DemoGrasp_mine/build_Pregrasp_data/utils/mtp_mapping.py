"""Map rollout palm states to PregraspPrior M/T/P bins."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from build_Pregrasp_data.utils.geometry import normalize


def add_pregrasp_root(pregrasp_root: str | Path) -> Path:
    root = Path(pregrasp_root).expanduser().resolve()
    import_root = root / "src" if (root / "src").is_dir() else root
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    return root


def parse_angle_list(spec: str | Sequence[float]) -> list[float]:
    if isinstance(spec, str):
        return [float(item) for item in spec.split(',') if item.strip()]
    return [float(item) for item in spec]


def make_template(pregrasp_root: str | Path, kind: str, subdivision: int, tilt_angles, pitch_angles):
    add_pregrasp_root(pregrasp_root)
    from oc_pregrasp.field.sphere_template import make_sphere_template

    return make_sphere_template(
        kind=kind,
        subdivision=int(subdivision),
        tilt_angles=parse_angle_list(tilt_angles),
        pitch_angles=parse_angle_list(pitch_angles),
    )


def nearest_angle_bin(value_deg: float, bins_deg: np.ndarray) -> int:
    bins = np.asarray(bins_deg, dtype=np.float64)
    return int(np.argmin(np.abs(bins - float(value_deg))))


def map_palm_object_to_mtp(
    palm_center_obj: np.ndarray,
    palm_toward_obj: np.ndarray,
    template,
    pregrasp_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    add_pregrasp_root(pregrasp_root)
    from oc_pregrasp.field.field_label import local_angular_offsets_deg
    from oc_pregrasp.field.sphere_template import nearest_direction

    centers = np.asarray(palm_center_obj, dtype=np.float32).reshape(-1, 3)
    palm_toward = normalize(np.asarray(palm_toward_obj, dtype=np.float32).reshape(-1, 3)).astype(np.float32)
    hand_back = -palm_toward
    mtp = np.zeros((len(centers), 3), dtype=np.int64)
    angle_info = np.zeros((len(centers), 3), dtype=np.float32)

    for index, (center, back_dir) in enumerate(zip(centers, hand_back)):
        position_dir = normalize(center.reshape(1, 3))[0].astype(np.float32)
        m_id, nearest_angle = nearest_direction(position_dir, template.dirs)
        tilt_deg, pitch_deg = local_angular_offsets_deg(template.dirs[m_id], back_dir)
        t_id = nearest_angle_bin(tilt_deg, template.tilt_angles)
        p_id = nearest_angle_bin(pitch_deg, template.pitch_angles)
        mtp[index] = [m_id, t_id, p_id]
        angle_info[index] = [nearest_angle, tilt_deg, pitch_deg]
    return mtp, angle_info
