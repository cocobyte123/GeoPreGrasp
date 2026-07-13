"""Aggregate yaw pregrasp rollout labels into per-object angle fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from build_Pregrasp_data.yaw_pregrasp.io.schema import FIELD_REQUIRED_KEYS


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROLLOUT_ROOT = (
    WORKSPACE_ROOT / "PregraspPrior" / "data" / "rollouts" / "yaw_pregrasp"
)
DEFAULT_FIELD_ROOT = (
    WORKSPACE_ROOT / "PregraspPrior" / "data" / "fields" / "yaw_pregrasp"
)


def _float_key(value: float) -> float:
    return round(float(value), 6)


def _load_manifest_config(rollout_root: Path):
    manifest_path = rollout_root / "manifest.json"
    if not manifest_path.is_file():
        return {}, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        grid = (
            [float(v) for v in config["yaw_angles"]],
            [float(v) for v in config["tilt_angles"]],
            [float(v) for v in config["pitch_angles"]],
        )
        return config, grid
    except Exception:
        return {}, None


def _iter_rollout_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/batch_*.npz"))


def _infer_grid(files: list[Path]):
    yaw_values = set()
    tilt_values = set()
    pitch_values = set()
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            yaw_values.update(_float_key(v) for v in data["yaw_deg"].reshape(-1))
            tilt_values.update(_float_key(v) for v in data["tilt_deg"].reshape(-1))
            pitch_values.update(_float_key(v) for v in data["pitch_deg"].reshape(-1))
    return sorted(yaw_values), sorted(tilt_values), sorted(pitch_values)


def _object_id_from_file(path: Path, data) -> str:
    if "object_id" in data:
        return str(np.asarray(data["object_id"]).item())
    return path.parent.name


def aggregate_yaw_fields(
    rollout_root: Path,
    field_root: Path,
    overwrite: bool = True,
) -> Path:
    rollout_root = rollout_root.expanduser().resolve()
    field_root = field_root.expanduser().resolve()
    files = list(_iter_rollout_files(rollout_root))
    if not files:
        raise FileNotFoundError(f"No rollout files found under {rollout_root}")

    rollout_config, grid = _load_manifest_config(rollout_root)
    if grid is None:
        grid = _infer_grid(files)
    yaw_angles, tilt_angles, pitch_angles = grid

    yaw_index = {_float_key(value): index for index, value in enumerate(yaw_angles)}
    tilt_index = {_float_key(value): index for index, value in enumerate(tilt_angles)}
    pitch_index = {_float_key(value): index for index, value in enumerate(pitch_angles)}
    shape = (len(yaw_angles), len(tilt_angles), len(pitch_angles))

    fields_dir = field_root / "fields"
    if overwrite and fields_dir.exists():
        for old_path in fields_dir.glob("*.npz"):
            old_path.unlink()
    fields_dir.mkdir(parents=True, exist_ok=True)

    objects: Dict[str, Dict] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            object_id = _object_id_from_file(path, data)
            object_asset = str(np.asarray(data["object_asset"]).item())
            entry = objects.setdefault(
                object_id,
                {
                    "object_asset": object_asset,
                    "success_count": np.zeros(shape, dtype=np.int64),
                    "trial_count": np.zeros(shape, dtype=np.int64),
                    "hit_count": np.zeros(shape, dtype=np.int64),
                    "clearance_sum": np.zeros(shape, dtype=np.float64),
                    "clearance_count": np.zeros(shape, dtype=np.int64),
                    "rollout_files": [],
                },
            )
            entry["rollout_files"].append(str(path))

            yaw = data["yaw_deg"].reshape(-1)
            tilt = data["tilt_deg"].reshape(-1)
            pitch = data["pitch_deg"].reshape(-1)
            success = data["success"].reshape(-1).astype(bool)
            hit_table = data["hit_table"].reshape(-1).astype(bool)
            min_clearance = data["min_clearance"].reshape(-1).astype(np.float64)

            for i in range(success.shape[0]):
                key = (
                    yaw_index[_float_key(yaw[i])],
                    tilt_index[_float_key(tilt[i])],
                    pitch_index[_float_key(pitch[i])],
                )
                entry["trial_count"][key] += 1
                entry["success_count"][key] += int(success[i])
                entry["hit_count"][key] += int(hit_table[i])
                if np.isfinite(min_clearance[i]):
                    entry["clearance_sum"][key] += float(min_clearance[i])
                    entry["clearance_count"][key] += 1

    manifest_entries = []
    for object_id, entry in sorted(objects.items()):
        trial_count = entry["trial_count"]
        success_count = entry["success_count"]
        success_rate = np.divide(
            success_count,
            trial_count,
            out=np.zeros_like(success_count, dtype=np.float32),
            where=trial_count > 0,
        ).astype(np.float32)
        hit_rate = np.divide(
            entry["hit_count"],
            trial_count,
            out=np.zeros_like(success_count, dtype=np.float32),
            where=trial_count > 0,
        ).astype(np.float32)
        clearance_mean = np.divide(
            entry["clearance_sum"],
            entry["clearance_count"],
            out=np.full(shape, np.nan, dtype=np.float64),
            where=entry["clearance_count"] > 0,
        ).astype(np.float32)

        path = fields_dir / f"{object_id}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray("YawPregraspPrior/v1"),
            required_keys=np.asarray(FIELD_REQUIRED_KEYS),
            object_id=np.asarray(object_id),
            object_asset=np.asarray(entry["object_asset"]),
            yaw_frame=np.asarray(str(rollout_config.get("yaw_frame", "unknown"))),
            canonical_obs=np.asarray(bool(rollout_config.get("canonical_obs", False))),
            canonical_obs_scope=np.asarray(
                str(rollout_config.get("canonical_obs_scope", "unknown"))
            ),
            direct_base_qpos=np.asarray(
                bool(rollout_config.get("direct_base_qpos", False))
            ),
            yaw_angles=np.asarray(yaw_angles, dtype=np.float32),
            tilt_angles=np.asarray(tilt_angles, dtype=np.float32),
            pitch_angles=np.asarray(pitch_angles, dtype=np.float32),
            success_count=success_count,
            trial_count=trial_count,
            success_rate=success_rate,
            hit_count=entry["hit_count"],
            hit_rate=hit_rate,
            min_clearance_mean=clearance_mean,
        )
        valid = trial_count > 0
        mean_success = float(success_count.sum() / max(trial_count.sum(), 1))
        manifest_entries.append(
            {
                "object_id": object_id,
                "object_asset": entry["object_asset"],
                "path": str(path),
                "rollout_files": len(entry["rollout_files"]),
                "trials": int(trial_count.sum()),
                "filled_cells": int(valid.sum()),
                "mean_success": mean_success,
            }
        )
        print(
            f"{object_id}: trials={int(trial_count.sum())}, "
            f"filled={int(valid.sum())}/{trial_count.size}, "
            f"success_rate={mean_success:.6f}, field={path}",
            flush=True,
        )

    manifest = {
        "schema": "YawPregraspPriorManifest/v1",
        "rollout_root": str(rollout_root),
        "field_root": str(field_root),
        "yaw_angles": yaw_angles,
        "yaw_frame": str(rollout_config.get("yaw_frame", "unknown")),
        "canonical_obs": bool(rollout_config.get("canonical_obs", False)),
        "canonical_obs_scope": str(
            rollout_config.get("canonical_obs_scope", "unknown")
        ),
        "direct_base_qpos": bool(rollout_config.get("direct_base_qpos", False)),
        "tilt_angles": tilt_angles,
        "pitch_angles": pitch_angles,
        "objects": manifest_entries,
    }
    manifest_path = field_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT
    )
    parser.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing field files instead of deleting them first.",
    )
    args = parser.parse_args()
    manifest = aggregate_yaw_fields(
        rollout_root=args.rollout_root,
        field_root=args.field_root,
        overwrite=not args.no_overwrite,
    )
    print(f"Yaw pregrasp field manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
