"""Inspect one PPO rollout batch npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_Pregrasp_data.io.rollout_schema import (
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    infer_batch_size,
    load_rollout_npz,
    scalar_to_str,
    validate_rollout_batch,
)


def _range_text(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float64)
    return f'min={array.min(axis=0)}, max={array.max(axis=0)}, mean={array.mean(axis=0)}'


def inspect(path: str | Path, strict: bool = True) -> None:
    batch = load_rollout_npz(path)
    errors = validate_rollout_batch(batch, strict=strict)
    print(f'file: {Path(path).expanduser().resolve()}')
    if errors:
        print('schema_errors:')
        for error in errors:
            print(f'  - {error}')

    print(f'fields: {sorted(batch.keys())}')
    object_name = scalar_to_str(batch.get('object_name', 'UNKNOWN'))
    print(f'object_name: {object_name}')
    if 'sample_id' in batch:
        print(f"sample_id: {scalar_to_str(batch['sample_id'])}")

    if all(key in batch for key in REQUIRED_FIELDS):
        batch_size = infer_batch_size(batch)
        success = np.asarray(batch['success']).reshape(-1).astype(np.float32)
        print(f'batch_size: {batch_size}')
        print(f'success_rate: {float(success.mean()):.6f} ({int(success.sum())}/{batch_size})')
        print(f"object_xyz: {_range_text(np.asarray(batch['object_pose_world'])[:, :3])}")
        print(f"table_height: {_range_text(np.asarray(batch['table_height']).reshape(-1, 1))}")
        print(f"palm_center_world: {_range_text(np.asarray(batch['initial_palm_center_world']))}")
        palm_toward_norm = np.linalg.norm(np.asarray(batch['initial_palm_toward_world']), axis=1)
        print(f'palm_toward_norm: min={palm_toward_norm.min():.6f}, max={palm_toward_norm.max():.6f}, mean={palm_toward_norm.mean():.6f}')

    if 'candidate_index' in batch:
        values, counts = np.unique(np.asarray(batch['candidate_index']).reshape(-1), return_counts=True)
        pairs = ', '.join(f'{int(v)}:{int(c)}' for v, c in zip(values[:20], counts[:20]))
        suffix = '' if len(values) <= 20 else f' ... total_unique={len(values)}'
        print(f'candidate_index_counts: {pairs}{suffix}')
    if 'field_mtp_index' in batch:
        mtp = np.asarray(batch['field_mtp_index'])
        print(f'field_mtp_index: shape={mtp.shape}, min={mtp.min(axis=0)}, max={mtp.max(axis=0)}')

    missing_optional = [field for field in OPTIONAL_FIELDS if field not in batch]
    if missing_optional:
        print(f'missing_optional: {missing_optional}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect one PPO rollout batch npz.')
    parser.add_argument('batch_npz')
    parser.add_argument('--no-strict', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect(args.batch_npz, strict=not args.no_strict)


if __name__ == '__main__':
    main()

