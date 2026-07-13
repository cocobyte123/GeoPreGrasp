"""Aggregate PPO rollout records into a PregraspPrior field store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from build_Pregrasp_data.io.rollout_schema import (
    infer_batch_size,
    load_rollout_npz,
    scalar_to_str,
    validate_rollout_batch,
)
from build_Pregrasp_data.utils.geometry import palm_world_to_object, points_world_to_object
from build_Pregrasp_data.utils.mtp_mapping import make_template, map_palm_object_to_mtp, parse_angle_list
from build_Pregrasp_data.utils.stats import aggregate_counts, target_and_confidence

SCHEMA = 'PregraspPriorStore/v1'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_PREGRASP_ROOT = WORKSPACE_ROOT / 'PregraspPrior'
DEFAULT_ROLLOUT_ROOT = DEFAULT_PREGRASP_ROOT / 'data' / 'rollouts' / 'ppo_random_mtp'
DEFAULT_OUTPUT_ROOT = DEFAULT_PREGRASP_ROOT / 'data' / 'fields' / 'ppo_random_mtp'
DEFAULT_SUBDIVISION = 3
DEFAULT_TILT_PITCH = '0,15,30'


def _add_pregrasp_root(pregrasp_root: str | Path) -> Path:
    root = Path(pregrasp_root).expanduser().resolve()
    import_root = root / 'src' if (root / 'src').is_dir() else root
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    return root


def _safe_object_key(pregrasp_root: str | Path, object_name: str) -> str:
    _add_pregrasp_root(pregrasp_root)
    from oc_pregrasp.field.field_store import safe_object_key

    return safe_object_key(object_name)


def _batch_paths_for_object(root: Path, object_name: str | None = None) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    if object_name:
        object_dir = root / object_name
        result[object_name] = sorted(object_dir.glob('batch_*.npz'))
        return result
    for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        batch_paths = sorted(object_dir.glob('batch_*.npz'))
        if batch_paths:
            result[object_dir.name] = batch_paths
    return result


def _points_from_batches(batches: Iterable[Dict[str, np.ndarray]], fallback_pose: np.ndarray) -> np.ndarray:
    for batch in batches:
        if 'object_point_cloud_object' in batch:
            points = np.asarray(batch['object_point_cloud_object'], dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 3:
                return points
        if 'points' in batch:
            points = np.asarray(batch['points'], dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 3:
                return points
        if 'object_point_cloud_world' in batch:
            points = np.asarray(batch['object_point_cloud_world'], dtype=np.float32)
            if points.ndim == 3:
                points = points[0]
            if points.ndim == 2 and points.shape[1] == 3:
                return points_world_to_object(points, fallback_pose)
    return np.empty((0, 3), dtype=np.float32)


def _indices_from_batch(batch: Dict[str, np.ndarray], template, pregrasp_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if 'field_mtp_index' in batch:
        mtp = np.asarray(batch['field_mtp_index'], dtype=np.int64)
        if mtp.ndim == 2 and mtp.shape[1] == 3 and np.all(mtp >= 0):
            return mtp, np.full((len(mtp), 3), np.nan, dtype=np.float32)

    palm_center_obj, palm_toward_obj = palm_world_to_object(
        object_pose_world=np.asarray(batch['object_pose_world'], dtype=np.float32),
        palm_center_world=np.asarray(batch['initial_palm_center_world'], dtype=np.float32),
        palm_toward_world=np.asarray(batch['initial_palm_toward_world'], dtype=np.float32),
    )
    return map_palm_object_to_mtp(
        palm_center_obj=palm_center_obj,
        palm_toward_obj=palm_toward_obj,
        template=template,
        pregrasp_root=pregrasp_root,
    )


def aggregate_object(
    object_name: str,
    batch_paths: List[Path],
    output_root: Path,
    pregrasp_root: Path,
    template,
    overwrite: bool,
) -> Dict:
    if not batch_paths:
        raise ValueError(f'No rollout batches found for object: {object_name}')

    object_attempt = np.zeros(template.shape, dtype=np.float32)
    object_success = np.zeros(template.shape, dtype=np.float32)
    batches: List[Dict[str, np.ndarray]] = []
    first_pose = None
    success_total = 0
    env_total = 0

    for batch_path in batch_paths:
        batch = load_rollout_npz(batch_path)
        validate_rollout_batch(batch, strict=True)
        batch_object = scalar_to_str(batch['object_name'])
        if batch_object != object_name:
            raise ValueError(f'{batch_path} object_name={batch_object}, expected {object_name}')
        if first_pose is None:
            first_pose = np.asarray(batch['object_pose_world'], dtype=np.float32)[0]

        mtp, _ = _indices_from_batch(batch, template=template, pregrasp_root=pregrasp_root)
        success = np.asarray(batch['success']).reshape(-1).astype(np.float32)
        attempt_count, success_count = aggregate_counts(mtp, success, template.shape)
        object_attempt += attempt_count
        object_success += success_count
        success_total += int(success.sum())
        env_total += infer_batch_size(batch)
        batches.append(batch)

    target, confidence = target_and_confidence(object_attempt, object_success)
    points = _points_from_batches(batches, fallback_pose=first_pose)

    object_key = _safe_object_key(pregrasp_root, object_name)
    field_dir = output_root / 'fields'
    field_dir.mkdir(parents=True, exist_ok=True)
    field_path = field_dir / f'{object_key}.npz'
    if field_path.exists() and not overwrite:
        raise FileExistsError(f'Field exists: {field_path}')

    np.savez_compressed(
        field_path,
        schema=np.asarray(SCHEMA),
        object_key=np.asarray(object_key),
        object_name=np.asarray(object_name),
        aliases=np.asarray([], dtype=object),
        points=np.asarray(points, dtype=np.float32),
        target_field=target.astype(np.float32),
        confidence=confidence.astype(np.float32),
        attempt_count=object_attempt.astype(np.float32),
        success_count=object_success.astype(np.float32),
        sphere_dirs=np.asarray(template.dirs, dtype=np.float32),
        sphere_faces=np.asarray(template.faces if template.faces is not None else np.empty((0, 3)), dtype=np.int64),
        tilt_angles=np.asarray(template.tilt_angles, dtype=np.float32),
        pitch_angles=np.asarray(template.pitch_angles, dtype=np.float32),
        template_kind=np.asarray(template.kind),
        subdivision=np.asarray(-1 if template.subdivision is None else int(template.subdivision), dtype=np.int64),
        grasp_count=np.asarray(int(object_attempt.sum()), dtype=np.int64),
        used_grasp_count=np.asarray(int(object_attempt.sum()), dtype=np.int64),
        rollout_count=np.asarray(len(batch_paths), dtype=np.int64),
        env_count=np.asarray(env_total, dtype=np.int64),
        success_total=np.asarray(success_total, dtype=np.int64),
        attempt_total=np.asarray(int(object_attempt.sum()), dtype=np.int64),
    )
    return {
        'object_key': object_key,
        'object_name': object_name,
        'field_path': str(field_path.relative_to(output_root)),
        'aliases': [],
        'rollout_count': len(batch_paths),
        'grasp_count': int(object_attempt.sum()),
        'used_grasp_count': int(object_attempt.sum()),
        'env_count': env_total,
        'success_total': success_total,
        'attempt_total': int(object_attempt.sum()),
        'success_rate': float(success_total / max(env_total, 1)),
    }


def write_manifest(output_root: Path, dataset_name: str, entries: List[Dict], config: Dict) -> Path:
    payload = {
        'schema': SCHEMA,
        'dataset': dataset_name,
        'root': str(output_root),
        'entries': entries,
        'config': config,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / 'manifest.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build PregraspPrior field store from PPO rollout batches.')
    parser.add_argument('--pregrasp-root', default=str(DEFAULT_PREGRASP_ROOT))
    parser.add_argument('--rollout-root', default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--dataset-name', default='ppo_rollout_fields')
    parser.add_argument('--object-name', default=None)
    parser.add_argument('--sphere-template', default='geodesic', choices=['geodesic', 'fibonacci'])
    parser.add_argument('--subdivision', type=int, default=DEFAULT_SUBDIVISION)
    parser.add_argument('--tilt-angles', default=DEFAULT_TILT_PITCH)
    parser.add_argument('--pitch-angles', default=DEFAULT_TILT_PITCH)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pregrasp_root = Path(args.pregrasp_root).expanduser().resolve()
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    template = make_template(
        pregrasp_root=pregrasp_root,
        kind=args.sphere_template,
        subdivision=args.subdivision,
        tilt_angles=parse_angle_list(args.tilt_angles),
        pitch_angles=parse_angle_list(args.pitch_angles),
    )
    object_batches = _batch_paths_for_object(rollout_root, object_name=args.object_name)
    if not object_batches:
        raise ValueError(f'No rollout batches found under {rollout_root}')

    entries = []
    for object_name, batch_paths in object_batches.items():
        entry = aggregate_object(
            object_name=object_name,
            batch_paths=batch_paths,
            output_root=output_root,
            pregrasp_root=pregrasp_root,
            template=template,
            overwrite=args.overwrite,
        )
        entries.append(entry)
        print(
            f"{object_name}: rollouts={entry['rollout_count']}, envs={entry['env_count']}, "
            f"success_rate={entry['success_rate']:.6f}, field={entry['field_path']}"
        )

    manifest = write_manifest(
        output_root=output_root,
        dataset_name=args.dataset_name,
        entries=entries,
        config={
            'source': 'build_Pregrasp_data.aggregate.build_field_store_from_rollouts',
            'rollout_root': str(rollout_root),
            'pregrasp_root': str(pregrasp_root),
            'sphere_template': args.sphere_template,
            'subdivision': int(args.subdivision),
            'tilt_angles': parse_angle_list(args.tilt_angles),
            'pitch_angles': parse_angle_list(args.pitch_angles),
            'assignment': 'hard_nearest',
        },
    )
    print(f'wrote manifest: {manifest}')


if __name__ == '__main__':
    main()
