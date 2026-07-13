"""Schema helpers for PPO rollout batch npz files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

SCHEMA = 'PregraspPPORollout/v1'

REQUIRED_FIELDS = (
    'object_name',
    'object_pose_world',
    'table_height',
    'initial_qpos33',
    'initial_palm_center_world',
    'initial_palm_toward_world',
    'success',
)

OPTIONAL_FIELDS = (
    'sample_id',
    'object_point_cloud_world',
    'object_point_cloud_center_world',
    'candidate_index',
    'field_mtp_index',
    'candidate_score',
    'episode_reward',
    'failure_code',
)


def scalar_to_str(value) -> str:
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def load_rollout_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def infer_batch_size(batch: Dict[str, np.ndarray]) -> int:
    pose = np.asarray(batch['object_pose_world'])
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f'object_pose_world must have shape [B,7], got {pose.shape}')
    return int(pose.shape[0])


def validate_rollout_batch(batch: Dict[str, np.ndarray], strict: bool = True) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_FIELDS:
        if key not in batch:
            errors.append(f'missing required field: {key}')
    if errors:
        if strict:
            raise ValueError('; '.join(errors))
        return errors

    batch_size = infer_batch_size(batch)
    expected_shapes = {
        'table_height': (batch_size,),
        'initial_qpos33': (batch_size, 33),
        'initial_palm_center_world': (batch_size, 3),
        'initial_palm_toward_world': (batch_size, 3),
        'success': (batch_size,),
    }
    for key, expected in expected_shapes.items():
        shape = np.asarray(batch[key]).shape
        if shape != expected:
            errors.append(f'{key} must have shape {expected}, got {shape}')

    if 'field_mtp_index' in batch:
        shape = np.asarray(batch['field_mtp_index']).shape
        if shape != (batch_size, 3):
            errors.append(f'field_mtp_index must have shape {(batch_size, 3)}, got {shape}')
    if 'candidate_index' in batch:
        shape = np.asarray(batch['candidate_index']).shape
        if shape != (batch_size,):
            errors.append(f'candidate_index must have shape {(batch_size,)}, got {shape}')

    if errors and strict:
        raise ValueError('; '.join(errors))
    return errors


def require_fields(batch: Dict[str, np.ndarray], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in batch]
    if missing:
        raise ValueError(f'missing fields: {missing}')

