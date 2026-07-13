"""Evaluate SE residual PPO with a learned yaw-pregrasp scorer.

This keeps the validated yaw-expanded execution path, but chooses each env's
yaw/tilt/pitch candidate by scoring the requested candidate grid with the MLP
trained under ``PregraspPrior``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ISAACGYM_ROOT = PROJECT_ROOT / "thirdparty" / "isaacgym" / "python"
ISAACGYM_ENVS_ROOT = PROJECT_ROOT / "thirdparty" / "IsaacGymEnvs"
OC_PREGRASP_SRC = WORKSPACE_ROOT / "PregraspPrior" / "src"


def _add_project_paths() -> None:
    for path in (PROJECT_ROOT, ISAACGYM_ROOT, ISAACGYM_ENVS_ROOT, OC_PREGRASP_SRC):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


_add_project_paths()


def _convert_csv_arg_to_hydra_list(value: str) -> str:
    text = str(value).strip()
    if text.startswith("["):
        return text
    return "[" + text + "]"


def _normalize_cli_aliases(argv: Sequence[str]) -> None:
    converted = [argv[0]]
    skip_next = False
    aliases = {
        "--yaw_angles": "+pregrasp_yaw_angles=",
        "--tilt_angles": "+se_tilt_angles=",
        "--pitch_angles": "+se_pitch_angles=",
    }
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        matched = False
        for cli_name, hydra_name in aliases.items():
            if arg.startswith(cli_name + "="):
                value = arg.split("=", 1)[1]
                converted.append(hydra_name + _convert_csv_arg_to_hydra_list(value))
                matched = True
                break
            if arg == cli_name and index + 1 < len(argv):
                value = argv[index + 1]
                converted.append(hydra_name + _convert_csv_arg_to_hydra_list(value))
                skip_next = True
                matched = True
                break
        if not matched:
            converted.append(arg)
    sys.argv[:] = converted


def _as_float_list(value) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json

            return [float(item) for item in json.loads(text)]
        return [float(item) for item in text.split(",") if item.strip()]
    return [float(item) for item in value]


def hydra_main():
    _add_project_paths()

    import gym
    import hydra
    import isaacgymenvs

    from isaacgym import gymapi  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.torch_jit_utils import (
        quat_apply,
        quat_conjugate,
        quat_from_angle_axis,
        quat_mul,
        unscale,
    )
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from omegaconf import DictConfig, open_dict

    import tasks  # noqa: F401
    from residual_se_grasp.se_residual_grasp import SEResidualGrasp
    from residual_se_grasp.train_se_reslearn import (
        TASK_NAME,
        build_runner,
        configure_se_training,
    )
    from residual_tilt_grasp.train_tilted_hand_only_reslearn import log_viewer_device
    from residual_tilt_grasp.tilted_hand_only_grasp import _quat_to_base_angles_zyx
    from oc_pregrasp.models.scorer_mlp import PointNetScorer
    import torch
    class YawScorerSEResidualGrasp(SEResidualGrasp):
        """Tilt/pitch/yaw SE residual task driven by a learned scorer."""

        def init_configs(self, cfg):
            env_cfg = cfg["env"]
            self.pregrasp_yaw_scorer_checkpoint = str(
                env_cfg.get("pregraspYawScorerCheckpoint", "")
            )
            self.pregrasp_yaw_frame = str(env_cfg.get("pregraspYawFrame", "pca_short"))
            if self.pregrasp_yaw_frame not in (
                "absolute",
                "object",
                "pca_long",
                "pca_short",
            ):
                raise ValueError(
                    "pregraspYawFrame must be absolute, object, pca_long, or pca_short"
                )
            self.pregrasp_yaw_scorer_top_k = int(
                env_cfg.get("pregraspYawScorerTopK", 1)
            )
            self.pregrasp_yaw_scorer_chunk_size = int(
                env_cfg.get("pregraspYawScorerChunkSize", 8192)
            )
            compare_base_angle = env_cfg.get("pregraspYawCompareBaseAngle", None)
            self.pregrasp_yaw_compare_base_angle_cfg = (
                _as_float_list(compare_base_angle)
                if compare_base_angle is not None
                else None
            )
            if (
                self.pregrasp_yaw_compare_base_angle_cfg is not None
                and len(self.pregrasp_yaw_compare_base_angle_cfg) != 3
            ):
                raise ValueError("pregraspYawCompareBaseAngle must have 3 values")
            self.pregrasp_yaw_compare_base_sampling = str(
                env_cfg.get("pregraspYawCompareBaseSampling", "random")
            )
            if self.pregrasp_yaw_compare_base_sampling not in ("random", "cycle"):
                raise ValueError(
                    "pregraspYawCompareBaseSampling must be random or cycle"
                )
            self.pregrasp_yaw_compare_base_mode = str(
                env_cfg.get("pregraspYawCompareBaseMode", "yaw0")
            )
            if self.pregrasp_yaw_compare_base_mode not in ("yaw0", "all"):
                raise ValueError(
                    "pregraspYawCompareBaseMode must be yaw0 or all"
                )
            self.pregrasp_yaw_scorer_sampling = str(
                env_cfg.get("pregraspYawScorerSampling", "scorer_top1")
            )
            if self.pregrasp_yaw_scorer_sampling not in (
                "scorer_top1",
                "scorer_topk",
                "scorer_sample",
            ):
                raise ValueError(
                    "pregraspYawScorerSampling must be scorer_top1, scorer_topk, or scorer_sample"
                )
            if self.pregrasp_yaw_scorer_top_k < 1:
                raise ValueError("pregraspYawScorerTopK must be >= 1")
            if self.pregrasp_yaw_scorer_chunk_size < 1:
                raise ValueError("pregraspYawScorerChunkSize must be >= 1")
            self.pregrasp_yaw_angles_cfg = _as_float_list(
                env_cfg.get("pregraspYawAngles", [0.0])
            )
            self.pregrasp_yaw_canonical_obs = bool(
                env_cfg.get("pregraspYawCanonicalObs", True)
            )
            self.pregrasp_yaw_canonical_obs_scope = str(
                env_cfg.get("pregraspYawCanonicalObsScope", "scene")
            )
            if self.pregrasp_yaw_canonical_obs_scope not in ("hand", "scene"):
                raise ValueError(
                    "pregraspYawCanonicalObsScope must be 'hand' or 'scene'"
                )
            self.pregrasp_yaw_direct_base_qpos = bool(
                env_cfg.get("pregraspYawDirectBaseQpos", True)
            )
            if not self.pregrasp_yaw_angles_cfg:
                raise ValueError("pregraspYawAngles must not be empty")
            super().init_configs(cfg)
            if self.se_guide_mode != "tilt_pitch":
                raise ValueError(
                    "Yaw-expanded eval currently expects seGuideMode=tilt_pitch"
                )
            self.pregrasp_yaw_angle_triples_cfg = [
                (yaw, float(tilt), float(pitch))
                for yaw in self.pregrasp_yaw_angles_cfg
                for tilt, pitch in self.se_angle_pairs_cfg
            ]
            self.world_tilt_angles = [
                float(index)
                for index in range(len(self.pregrasp_yaw_angle_triples_cfg))
            ]
            self._angle_success_sum = {
                angle: 0.0 for angle in self.world_tilt_angles
            }
            self._angle_success_count = {
                angle: 0 for angle in self.world_tilt_angles
            }
            self._angle_hit_sum = {
                angle: 0.0 for angle in self.world_tilt_angles
            }
            self._angle_clearance_sum = {
                angle: 0.0 for angle in self.world_tilt_angles
            }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pregrasp_yaw_angle_triples = torch.tensor(
                self.pregrasp_yaw_angle_triples_cfg,
                dtype=torch.float32,
                device=self.device,
            )
            self.env_pregrasp_yaw = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.env_pregrasp_world_yaw = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.env_pregrasp_object_yaw = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.env_pregrasp_reference_yaw = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.env_pregrasp_yaw_quat = self.identity_quat.view(1, 4).repeat(
                self.num_envs, 1
            )
            self.env_pregrasp_yaw_quat_inv = (
                self.identity_quat.view(1, 4).repeat(self.num_envs, 1)
            )
            self.env_traj_normal_canonical = self.table_normal.view(
                1, 3
            ).repeat(self.num_envs, 1)
            self._scorer_cycle_index = 0
            self._compare_base_cycle_index = 0
            self.pregrasp_yaw_force_base_angle = False
            self.pregrasp_yaw_compare_base_angle_id = None
            if self.pregrasp_yaw_compare_base_angle_cfg is not None:
                self.pregrasp_yaw_compare_base_angle_id = (
                    self._find_angle_id_by_triple(
                        self.pregrasp_yaw_compare_base_angle_cfg
                    )
                )
            self.pregrasp_yaw_compare_base_angle_ids = (
                self._find_compare_base_angle_ids()
            )
            self._load_yaw_scorer()
            print(
                "Yaw-scorer SE residual eval env: "
                f"yaw={self.pregrasp_yaw_angles_cfg}, "
                f"tilt/pitch={self.se_angle_pairs_cfg}, "
                f"candidates={len(self.pregrasp_yaw_angle_triples_cfg)}, "
                f"sampling={self.pregrasp_yaw_scorer_sampling}, "
                f"yaw_frame={self.pregrasp_yaw_frame}, "
                f"scorer_checkpoint={self.pregrasp_yaw_scorer_checkpoint}, "
                f"scorer_top_k={self.pregrasp_yaw_scorer_top_k}, "
                f"canonical_obs={self.pregrasp_yaw_canonical_obs}, "
                f"canonical_scope={self.pregrasp_yaw_canonical_obs_scope}, "
                f"direct_base_qpos={self.pregrasp_yaw_direct_base_qpos}",
                flush=True,
            )

        def _find_angle_id_by_triple(self, triple):
            target = self._angle_key(triple[0], triple[1], triple[2])
            for index, candidate in enumerate(self.pregrasp_yaw_angle_triples_cfg):
                if self._angle_key(*candidate) == target:
                    return int(index)
            raise ValueError(
                "compare base angle is not in requested yaw/tilt/pitch grid: "
                f"{triple}"
            )

        def _find_compare_base_angle_ids(self):
            if self.pregrasp_yaw_compare_base_mode == "all":
                return torch.arange(
                    len(self.pregrasp_yaw_angle_triples_cfg),
                    dtype=torch.long,
                    device=self.device,
                )
            ids = [
                index
                for index, candidate in enumerate(self.pregrasp_yaw_angle_triples_cfg)
                if abs(float(candidate[0])) < 1e-6
            ]
            if not ids:
                raise ValueError(
                    "compare base requires yaw=0 in --yaw_angles/pregrasp_yaw_angles"
                )
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        def set_pregrasp_yaw_force_base_angle(self, enabled: bool):
            self.pregrasp_yaw_force_base_angle = bool(enabled)

        def _object_name_for_env_id(self, env_id: int) -> str:
            object_fn = self.object_fns[int(env_id) % len(self.object_fns)]
            return Path(object_fn).stem

        def _angle_key(self, yaw, tilt, pitch):
            return (round(float(yaw), 6), round(float(tilt), 6), round(float(pitch), 6))

        def _load_yaw_scorer(self):
            if not self.pregrasp_yaw_scorer_checkpoint:
                raise ValueError(
                    "yaw scorer eval requires +pregrasp_yaw_scorer_checkpoint=..."
                )
            checkpoint = Path(self.pregrasp_yaw_scorer_checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Yaw scorer checkpoint not found: {checkpoint}")
            try:
                payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint, map_location=self.device)
            state = (
                payload["model"]
                if isinstance(payload, dict) and "model" in payload
                else payload
            )
            self.pregrasp_yaw_scorer = PointNetScorer().to(self.device)
            self.pregrasp_yaw_scorer.load_state_dict(state)
            self.pregrasp_yaw_scorer.eval()
            for parameter in self.pregrasp_yaw_scorer.parameters():
                parameter.requires_grad_(False)
            self.pregrasp_yaw_scorer_feature_mode = "raw"
            if isinstance(payload, dict):
                self.pregrasp_yaw_scorer_feature_mode = str(
                    payload.get(
                        "feature_mode",
                        payload.get("args", {}).get("feature_mode", "raw"),
                    )
                )
            if self.pregrasp_yaw_scorer_feature_mode not in ("raw", "pca_short"):
                raise ValueError(
                    "Unsupported yaw scorer feature mode: "
                    f"{self.pregrasp_yaw_scorer_feature_mode}"
                )
            epoch = payload.get("epoch", "unknown") if isinstance(payload, dict) else "unknown"
            val = payload.get("val", {}) if isinstance(payload, dict) else {}
            val_acc = val.get("accuracy", "unknown") if isinstance(val, dict) else "unknown"
            print(
                f"Loaded yaw scorer from {checkpoint} "
                f"(epoch={epoch}, val_acc={val_acc}, "
                f"feature_mode={self.pregrasp_yaw_scorer_feature_mode})",
                flush=True,
            )

        def _candidate_angle_features(self):
            angles = torch.deg2rad(self.pregrasp_yaw_angle_triples)
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        def _rotate_z_points(self, points, yaw_rad):
            c = torch.cos(yaw_rad).view(-1, 1)
            s = torch.sin(yaw_rad).view(-1, 1)
            result = points.clone()
            x = points[:, :, 0]
            y = points[:, :, 1]
            result[:, :, 0] = c * x - s * y
            result[:, :, 1] = s * x + c * y
            return result

        def _scorer_inputs(self, env_ids, object_pose):
            point_cloud = self.obj_pcl_buf[env_ids].to(dtype=torch.float32)
            if getattr(self, "pregrasp_yaw_scorer_feature_mode", "raw") == "raw":
                return point_cloud, object_pose

            count, point_count, _ = point_cloud.shape
            quat = object_pose[:, 3:7].unsqueeze(1).expand(count, point_count, 4)
            world_rel = quat_apply(
                quat.reshape(-1, 4), point_cloud.reshape(-1, 3)
            ).reshape(count, point_count, 3)
            reference_yaw = self.env_pregrasp_reference_yaw[env_ids]
            centered = world_rel - world_rel.mean(dim=1, keepdim=True)
            canonical = self._rotate_z_points(centered, -reference_yaw)
            extents = canonical.max(dim=1).values - canonical.min(dim=1).values
            short_extent = extents[:, 0]
            long_extent = extents[:, 1]
            height = extents[:, 2]
            anisotropy = (long_extent - short_extent) / (
                long_extent + short_extent
            ).clamp_min(1e-6)
            object_yaw = self._quat_z_yaw(object_pose[:, 3:7])
            rel_yaw = object_yaw - reference_yaw
            pose_features = torch.stack(
                [
                    short_extent,
                    long_extent,
                    height,
                    anisotropy,
                    object_pose[:, 2],
                    torch.sin(rel_yaw),
                    torch.cos(rel_yaw),
                ],
                dim=-1,
            )
            return canonical, pose_features

        def _score_scorer_candidates(self, env_ids, object_pose):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            object_pose = object_pose.to(device=self.device, dtype=torch.float32)
            point_cloud, pose_features = self._scorer_inputs(env_ids, object_pose)
            angle_features = self._candidate_angle_features()
            count = int(env_ids.numel())
            candidate_count = int(angle_features.shape[0])
            scores = torch.empty(
                (count, candidate_count), dtype=torch.float32, device=self.device
            )
            flat_scores = scores.reshape(-1)
            total = count * candidate_count
            chunk_size = int(self.pregrasp_yaw_scorer_chunk_size)
            with torch.no_grad():
                for start in range(0, total, chunk_size):
                    stop = min(start + chunk_size, total)
                    flat_ids = torch.arange(
                        start, stop, dtype=torch.long, device=self.device
                    )
                    local_env_ids = torch.div(
                        flat_ids, candidate_count, rounding_mode="floor"
                    )
                    candidate_ids = flat_ids % candidate_count
                    logits = self.pregrasp_yaw_scorer(
                        point_cloud[local_env_ids],
                        angle_features[candidate_ids],
                        pose_features[local_env_ids],
                    ).reshape(-1)
                    flat_scores[start:stop] = torch.sigmoid(logits)
            return scores

        def _sample_scorer_angle_ids(self, env_ids, object_pose):
            mode = str(self.pregrasp_yaw_scorer_sampling)
            scores = self._score_scorer_candidates(env_ids, object_pose)
            sorted_ids = torch.argsort(scores, dim=-1, descending=True)
            if mode == "scorer_top1":
                return sorted_ids[:, 0]
            top_k = min(self.pregrasp_yaw_scorer_top_k, int(sorted_ids.shape[1]))
            ids = sorted_ids[:, :top_k]
            if mode == "scorer_topk":
                offsets = (
                    torch.arange(len(env_ids), device=self.device)
                    + self._scorer_cycle_index
                ) % top_k
                self._scorer_cycle_index = (
                    self._scorer_cycle_index + len(env_ids)
                ) % max(1, top_k)
                return ids[torch.arange(len(env_ids), device=self.device), offsets]
            if mode == "scorer_sample":
                weights = scores.gather(1, ids).clamp_min(0.0) + 1e-6
                chosen = torch.multinomial(weights, 1).squeeze(-1)
                return ids[torch.arange(len(env_ids), device=self.device), chosen]
            raise ValueError(
                "scorer eval sampling must be scorer_top1, scorer_topk, or scorer_sample"
            )

        def _sample_compare_base_angle_ids(self, env_ids):
            if self.pregrasp_yaw_compare_base_angle_id is not None:
                return torch.full(
                    (len(env_ids),),
                    int(self.pregrasp_yaw_compare_base_angle_id),
                    dtype=torch.long,
                    device=self.device,
                )
            count = len(env_ids)
            base_ids = self.pregrasp_yaw_compare_base_angle_ids
            if base_ids.numel() == 1:
                return base_ids.expand(count)
            if self.pregrasp_yaw_compare_base_sampling == "cycle":
                ids = (
                    torch.arange(count, device=self.device)
                    + self._compare_base_cycle_index
                ) % int(base_ids.numel())
                self._compare_base_cycle_index = (
                    self._compare_base_cycle_index + count
                ) % int(base_ids.numel())
                return base_ids[ids]
            ids = torch.randint(
                0,
                int(base_ids.numel()),
                (count,),
                dtype=torch.long,
                device=self.device,
            )
            return base_ids[ids]

        def _sample_trajectory_angle_ids(self, env_ids):
            if self.pregrasp_yaw_force_base_angle:
                return self._sample_compare_base_angle_ids(env_ids)
            return torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

        def format_angle_id(self, angle_id):
            yaw, tilt, pitch = self.pregrasp_yaw_angle_triples_cfg[
                int(angle_id)
            ]
            return f"yaw{yaw:g}_tilt{tilt:g}_pitch{pitch:g}"

        def _quat_z_yaw(self, quats):
            x = quats[:, 0]
            y = quats[:, 1]
            z = quats[:, 2]
            w = quats[:, 3]
            return torch.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )

        def _pca_table_axis_yaw(self, env_ids, object_pose, axis_kind: str):
            if not hasattr(self, "obj_pcl_buf"):
                return self._quat_z_yaw(object_pose[:, 3:7])
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            local_points = self.obj_pcl_buf[env_ids].to(dtype=torch.float32)
            count, point_count, _ = local_points.shape
            quat = object_pose[:, 3:7].unsqueeze(1).expand(count, point_count, 4)
            world_rel = quat_apply(
                quat.reshape(-1, 4), local_points.reshape(-1, 3)
            ).reshape(count, point_count, 3)
            xy = world_rel[:, :, :2]
            xy = xy - xy.mean(dim=1, keepdim=True)
            cov = torch.matmul(xy.transpose(1, 2), xy) / max(point_count - 1, 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            axis_index = 1 if axis_kind == "pca_long" else 0
            axis = eigvecs[:, :, axis_index]
            flip = (axis[:, 0] < 0.0) | (
                (axis[:, 0].abs() < 1e-6) & (axis[:, 1] < 0.0)
            )
            axis = torch.where(flip.unsqueeze(-1), -axis, axis)
            yaw = torch.atan2(axis[:, 1], axis[:, 0])
            fallback = self._quat_z_yaw(object_pose[:, 3:7])
            anisotropy = eigvals[:, 1] - eigvals[:, 0]
            return torch.where(anisotropy > 1e-8, yaw, fallback)

        def _prepare_tilted_reset(self, env_ids):
            object_pose = super()._prepare_tilted_reset(env_ids)
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            if not hasattr(self, "env_pregrasp_object_yaw"):
                self.env_pregrasp_object_yaw = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
                self.env_pregrasp_world_yaw = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
                self.env_pregrasp_reference_yaw = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
            object_yaw = self._quat_z_yaw(object_pose[:, 3:7])
            self.env_pregrasp_object_yaw[env_ids] = object_yaw
            if self.pregrasp_yaw_frame == "object":
                reference_yaw = object_yaw
            elif self.pregrasp_yaw_frame in ("pca_long", "pca_short"):
                reference_yaw = self._pca_table_axis_yaw(
                    env_ids, object_pose, self.pregrasp_yaw_frame
                )
            else:
                reference_yaw = torch.zeros_like(object_yaw)
            self.env_pregrasp_reference_yaw[env_ids] = reference_yaw
            if self.pregrasp_yaw_force_base_angle:
                angle_ids = self._sample_compare_base_angle_ids(env_ids)
            else:
                angle_ids = self._sample_scorer_angle_ids(env_ids, object_pose)
            self._set_env_trajectory_rotations(env_ids, angle_ids)
            return object_pose

        def _set_env_trajectory_rotations(self, env_ids, angle_ids):
            self._ensure_base_tracking_reference()
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            angle_ids = angle_ids.to(device=self.device, dtype=torch.long)
            triples = self.pregrasp_yaw_angle_triples[angle_ids]
            yaw_deg = triples[:, 0]
            tilt_deg = triples[:, 1]
            pitch_deg = triples[:, 2]
            reference_yaw_rad = (
                self.env_pregrasp_reference_yaw[env_ids]
                if hasattr(self, "env_pregrasp_reference_yaw")
                else torch.zeros_like(yaw_deg)
            )
            world_yaw_deg = yaw_deg + torch.rad2deg(reference_yaw_rad)

            base_guide_quat = self._guide_quat_from_tilt_pitch(
                env_ids, tilt_deg, pitch_deg
            )
            z_axis = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device
            ).expand(len(env_ids), -1)
            yaw_quat = quat_from_angle_axis(torch.deg2rad(world_yaw_deg), z_axis)
            yaw_quat_inv = quat_conjugate(yaw_quat)
            guide_quat = quat_mul(yaw_quat, base_guide_quat)

            self.env_angle_ids[env_ids] = angle_ids
            self.env_pregrasp_yaw[env_ids] = yaw_deg
            self.env_pregrasp_world_yaw[env_ids] = world_yaw_deg
            self.env_pregrasp_yaw_quat[env_ids] = yaw_quat
            self.env_pregrasp_yaw_quat_inv[env_ids] = yaw_quat_inv
            self.env_se_yaw[env_ids] = tilt_deg
            self.env_se_pitch[env_ids] = pitch_deg
            self.env_traj_angles[env_ids] = torch.linalg.vector_norm(
                torch.stack([yaw_deg, tilt_deg, pitch_deg], dim=-1), dim=-1
            )
            self.env_traj_quat[env_ids] = guide_quat
            self.env_traj_quat_inv[env_ids] = quat_conjugate(guide_quat)

            table_z = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device
            ).expand(len(env_ids), -1)
            self.env_traj_normal[env_ids] = quat_apply(guide_quat, table_z)
            self.env_traj_normal_canonical[env_ids] = quat_apply(
                base_guide_quat, table_z
            )

            tilt_rad = torch.deg2rad(tilt_deg)
            pitch_rad = torch.deg2rad(pitch_deg)
            self.env_se_guide_obs[env_ids] = torch.stack(
                [
                    torch.sin(tilt_rad),
                    torch.cos(tilt_rad),
                    torch.sin(pitch_rad),
                    torch.cos(pitch_rad),
                ],
                dim=-1,
            )
            self.current_world_tilt = float(
                self.env_traj_angles[env_ids].float().mean().item()
            )

            for key, value in self.base_tracking_reference.items():
                self.tracking_reference[key][env_ids] = value[env_ids]
            quat_seq = guide_quat.unsqueeze(1).expand(-1, self.T_ref, -1)
            self.tracking_reference["wrist_initobj_pos"][env_ids] = quat_apply(
                quat_seq,
                self.base_tracking_reference["wrist_initobj_pos"][env_ids],
            )
            self.tracking_reference["wrist_quat"][env_ids] = quat_mul(
                quat_seq,
                self.base_tracking_reference["wrist_quat"][env_ids],
            )

        def compute_reward(self):
            super().compute_reward()
            self.extras["pregrasp_yaw_deg"] = self.env_pregrasp_yaw.clone()

        def _canonical_yaw_centers(self):
            if hasattr(self, "object_init_states"):
                return self.object_init_states[:, 0:3]
            if hasattr(self, "object_pos"):
                return self.object_pos
            return torch.zeros((self.num_envs, 3), device=self.device)

        def _canonicalize_world_positions(self, positions):
            centers = self._canonical_yaw_centers()
            inv_quat = self.env_pregrasp_yaw_quat_inv
            if positions.dim() == 2:
                return centers + quat_apply(inv_quat, positions - centers)

            shape = positions.shape
            flat_positions = positions.reshape(self.num_envs, -1, 3)
            flat_centers = centers.unsqueeze(1).expand_as(flat_positions)
            flat_quat = inv_quat.unsqueeze(1).expand(
                self.num_envs, flat_positions.shape[1], 4
            )
            canonical = flat_centers + quat_apply(
                flat_quat.reshape(-1, 4),
                (flat_positions - flat_centers).reshape(-1, 3),
            ).reshape_as(flat_positions)
            return canonical.reshape(shape)

        def _canonicalize_world_quats(self, quats):
            inv_quat = self.env_pregrasp_yaw_quat_inv
            if quats.dim() == 2:
                canonical = quat_mul(inv_quat, quats)
                sign = torch.where(
                    canonical[:, 3:4] < 0.0,
                    -torch.ones_like(canonical[:, 3:4]),
                    torch.ones_like(canonical[:, 3:4]),
                )
                return canonical * sign

            shape = quats.shape
            flat_quats = quats.reshape(self.num_envs, -1, 4)
            flat_inv = inv_quat.unsqueeze(1).expand_as(flat_quats)
            canonical = quat_mul(
                flat_inv.reshape(-1, 4),
                flat_quats.reshape(-1, 4),
            ).reshape_as(flat_quats)
            sign = torch.where(
                canonical[..., 3:4] < 0.0,
                -torch.ones_like(canonical[..., 3:4]),
                torch.ones_like(canonical[..., 3:4]),
            )
            canonical = canonical * sign
            return canonical.reshape(shape)

        def _apply_yaw_canonical_observation(self, obs_buf, obs_type, num_obs):
            if not self.pregrasp_yaw_canonical_obs:
                return

            parts = set(str(obs_type).split("+"))
            obs_end = 0
            ordered_parts = (
                "armdof",
                "handdof",
                "fulldof",
                "eefpose",
                "ftpos",
                "palmpose",
                "lastact",
                "objxyz",
                "objpose",
                "objinitpose",
            )
            for part in ordered_parts:
                if part not in parts:
                    continue
                dim = int(self.num_obs_dict[part])
                start = obs_end
                stop = start + dim
                view = obs_buf[:, start:stop]
                if part in ("eefpose", "palmpose"):
                    view[:, 0:3] = self._canonicalize_world_positions(
                        view[:, 0:3]
                    )
                    view[:, 3:7] = self._canonicalize_world_quats(
                        view[:, 3:7]
                    )
                elif part == "ftpos":
                    view[:] = self._canonicalize_world_positions(
                        view.reshape(self.num_envs, -1, 3)
                    ).reshape(self.num_envs, -1)
                elif (
                    self.pregrasp_yaw_canonical_obs_scope == "scene"
                    and part in ("objpose", "objinitpose")
                ):
                    view[:, 0:3] = self._canonicalize_world_positions(
                        view[:, 0:3]
                    )
                    view[:, 3:7] = self._canonicalize_world_quats(
                        view[:, 3:7]
                    )
                elif (
                    self.pregrasp_yaw_canonical_obs_scope == "scene"
                    and part == "objxyz"
                ):
                    view[:] = self._canonicalize_world_positions(view)
                obs_end = stop

            if "seguide" in parts:
                obs_end += self.GUIDE_OBS_DIM

            if "objpcl" in parts:
                dim = int(self.num_obs_dict["objpcl"])
                start = obs_end
                stop = start + dim
                if self.pregrasp_yaw_canonical_obs_scope == "scene":
                    pcl = obs_buf[:, start:stop].reshape(self.num_envs, -1, 3)
                    obs_buf[:, start:stop] = self._canonicalize_world_positions(
                        pcl
                    ).reshape(self.num_envs, -1)
                obs_end = stop

            if obs_end != num_obs:
                raise RuntimeError(
                    "Yaw canonical observation layout mismatch: "
                    f"parsed={obs_end}, expected={num_obs}, obs_type={obs_type}"
                )

        def compute_required_observations(self, obs_buf, obs_type, num_obs):
            super().compute_required_observations(obs_buf, obs_type, num_obs)
            self._apply_yaw_canonical_observation(obs_buf, obs_type, num_obs)

        def get_baseline_observation(self, residual_obs=None):
            if not self.pregrasp_yaw_canonical_obs:
                return super().get_baseline_observation(residual_obs)
            if residual_obs is None:
                residual_obs = self.obs_dict["obs"]
            if getattr(self, "baseline_uses_residual_observation", False):
                return residual_obs
            if "seguide" not in self.obs_type.split("+"):
                return super().get_baseline_observation(residual_obs)

            pcl_dim = (
                self.points_per_object * 3
                if "objpcl" in self.obs_type.split("+")
                else 0
            )
            state_dim = residual_obs.shape[-1] - pcl_dim - self.GUIDE_OBS_DIM
            state_obs = residual_obs[:, :state_dim]
            pcl_obs = residual_obs[:, state_dim + self.GUIDE_OBS_DIM:]
            if "objpcl" in self.obs_type.split("+"):
                return torch.cat(
                    [state_obs, self.env_traj_normal_canonical, pcl_obs],
                    dim=-1,
                )
            return torch.cat(
                [state_obs, self.env_traj_normal_canonical],
                dim=-1,
            )

        def compute_reference_actions(self):
            if (
                not self.pregrasp_yaw_direct_base_qpos
                or self.arm_controller != "qpos"
                or self.use_relative_control
                or not self._is_new_sr_hand_simple()
            ):
                return super().compute_reference_actions()

            env_ids = torch.arange(self.num_envs, device=self.device)
            reaching_plan_timestep_ids = torch.minimum(
                self.progress_buf, self.reaching_plan_timesteps
            )
            tracking_timestep_ids = (
                self.progress_buf - self.reaching_plan_timesteps
            ).clamp(min=0, max=self.T_ref - 1)
            hand_qpos_target = self.current_tracking_reference["hand_qpos"][
                env_ids, tracking_timestep_ids
            ]

            reaching_arm_qpos = self.reaching_plan_base_qpos[
                env_ids, reaching_plan_timestep_ids
            ]
            tracking_arm_qpos = self.tracking_reference_base_qpos[
                env_ids, tracking_timestep_ids
            ]
            reaching_mask = (
                self.progress_buf < self.reaching_plan_timesteps
            ).unsqueeze(-1)
            arm_qpos_target = torch.where(
                reaching_mask, reaching_arm_qpos, tracking_arm_qpos
            )
            qpos_target = torch.cat([arm_qpos_target, hand_qpos_target], dim=-1)
            return unscale(
                qpos_target,
                self.robot_dof_lower_limits[self.active_robot_dof_indices],
                self.robot_dof_upper_limits[self.active_robot_dof_indices],
            )

        def _unwrap_angle_sequence(self, angles):
            if angles.shape[1] <= 1:
                return angles
            unwrapped = angles.clone()
            for index in range(1, angles.shape[1]):
                delta = angles[:, index] - unwrapped[:, index - 1]
                delta = torch.atan2(torch.sin(delta), torch.cos(delta))
                unwrapped[:, index] = unwrapped[:, index - 1] + delta
            return unwrapped

        def _wrist_pose_to_base_qpos_sequence(self, positions, quats):
            angles = _quat_to_base_angles_zyx(
                quats.reshape(-1, 4)
            ).reshape(positions.shape[0], positions.shape[1], 3)
            angles = self._unwrap_angle_sequence(angles)
            return torch.cat([positions, angles], dim=-1)

        def _prepare_direct_base_qpos_plans(self, env_ids):
            if not hasattr(self, "reaching_plan_base_qpos"):
                self.reaching_plan_base_qpos = torch.zeros(
                    (
                        self.num_envs,
                        self.max_episode_length,
                        self.num_arm_dofs,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.tracking_reference_base_qpos = torch.zeros(
                    (self.num_envs, self.T_ref, self.num_arm_dofs),
                    dtype=torch.float32,
                    device=self.device,
                )

            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            tracking_positions = (
                self.current_tracking_reference["wrist_initobj_pos"][env_ids]
                + self.object_init_states[env_ids, 0:3].unsqueeze(1)
            )
            tracking_quats = self.current_tracking_reference["wrist_quat"][
                env_ids
            ]
            tracking_qpos = self._wrist_pose_to_base_qpos_sequence(
                tracking_positions, tracking_quats
            )
            start_qpos = self.robot_dof_pos[:, self.arm_dof_indices][
                env_ids
            ]
            angle_offset = (
                torch.round(
                    (start_qpos[:, 3:6] - tracking_qpos[:, 0, 3:6])
                    / (2.0 * torch.pi)
                )
                * (2.0 * torch.pi)
            )
            tracking_qpos[:, :, 3:6] = (
                tracking_qpos[:, :, 3:6] + angle_offset.unsqueeze(1)
            )
            self.tracking_reference_base_qpos[env_ids] = tracking_qpos

            target_qpos = tracking_qpos[:, 0]
            delta = target_qpos - start_qpos
            delta[:, 3:6] = torch.atan2(
                torch.sin(delta[:, 3:6]), torch.cos(delta[:, 3:6])
            )
            steps = torch.arange(
                self.max_episode_length,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1, 1)
            denom = self.reaching_plan_timesteps[env_ids].float().clamp_min(
                1.0
            ).view(-1, 1, 1)
            fraction = ((steps + 1.0) / denom).clamp(max=1.0)
            self.reaching_plan_base_qpos[env_ids] = (
                start_qpos.unsqueeze(1) + fraction * delta.unsqueeze(1)
            )

        def generate_residual_reaching_plan_idx(
            self, env_ids, baseline_actions, residual_actions
        ):
            result = super().generate_residual_reaching_plan_idx(
                env_ids, baseline_actions, residual_actions
            )
            if (
                self.pregrasp_yaw_direct_base_qpos
                and self.arm_controller == "qpos"
                and not self.use_relative_control
                and self._is_new_sr_hand_simple()
            ):
                self._prepare_direct_base_qpos_plans(env_ids)
            return result

    def _obs_slices(env):
        parts = set(str(env.obs_type).split("+"))
        ordered_parts = (
            "armdof",
            "handdof",
            "fulldof",
            "eefpose",
            "ftpos",
            "palmpose",
            "lastact",
            "objxyz",
            "objpose",
            "objinitpose",
        )
        slices = {}
        obs_end = 0
        for part in ordered_parts:
            if part not in parts:
                continue
            dim = int(env.num_obs_dict[part])
            slices[part] = slice(obs_end, obs_end + dim)
            obs_end += dim
        if "seguide" in parts:
            slices["seguide"] = slice(obs_end, obs_end + env.GUIDE_OBS_DIM)
            obs_end += env.GUIDE_OBS_DIM
        if "worldtilt" in parts:
            slices["worldtilt"] = slice(obs_end, obs_end + 3)
            obs_end += 3
        if "objpcl" in parts:
            dim = int(env.num_obs_dict["objpcl"])
            slices["objpcl"] = slice(obs_end, obs_end + dim)
            obs_end += dim
        if obs_end != env.num_observations:
            raise RuntimeError(
                "Debug observation layout mismatch: "
                f"parsed={obs_end}, expected={env.num_observations}, "
                f"obs_type={env.obs_type}"
            )
        return slices

    def _max_abs_diff(a, b):
        return (a - b).abs().max().item()

    def _run_yaw_equiv_debug(env, runner):
        if env.num_envs < 2:
            raise ValueError("pregrasp_yaw_debug_equiv requires num_envs >= 2")

        device = env.device
        env_ids = torch.arange(env.num_envs, device=device)
        with torch.no_grad():
            obs = env.reset_idx(env_ids)["obs"]
            states = env.get_state()
            baseline_actions = runner._baseline_actions(obs, states)
            residual_actions = runner.actor_critic(
                obs, states, inference=True
            )

        left = 0
        right = 1
        left_angle_id = int(env.env_angle_ids[left].item())
        right_angle_id = int(env.env_angle_ids[right].item())
        left_label = (
            env.format_angle_id(left_angle_id)
            if hasattr(env, "format_angle_id")
            else str(left_angle_id)
        )
        right_label = (
            env.format_angle_id(right_angle_id)
            if hasattr(env, "format_angle_id")
            else str(right_angle_id)
        )
        print("Yaw equivalence debug", flush=True)
        print(
            f"  pair: env{left}={left_label}, env{right}={right_label}",
            flush=True,
        )
        print(
            "  yaw_deg: "
            f"env{left}={env.env_pregrasp_yaw[left].item():.6g}, "
            f"env{right}={env.env_pregrasp_yaw[right].item():.6g}",
            flush=True,
        )
        print(
            "  object_init_pos_diff="
            f"{_max_abs_diff(env.object_init_states[left, :3], env.object_init_states[right, :3]):.6g}, "
            "object_init_quat_diff="
            f"{_max_abs_diff(env.object_init_states[left, 3:7], env.object_init_states[right, 3:7]):.6g}",
            flush=True,
        )

        slices = _obs_slices(env)
        print("  observation max_abs_diff by slice:", flush=True)
        for name, slc in slices.items():
            diff = _max_abs_diff(obs[left, slc], obs[right, slc])
            print(f"    {name}: {diff:.6g}", flush=True)
            if name in ("eefpose", "palmpose", "objpose", "objinitpose"):
                print(
                    f"      {name}.pos={_max_abs_diff(obs[left, slc][0:3], obs[right, slc][0:3]):.6g}, "
                    f"{name}.quat={_max_abs_diff(obs[left, slc][3:7], obs[right, slc][3:7]):.6g}",
                    flush=True,
                )
        print(
            "  policy action max_abs_diff: "
            f"baseline={_max_abs_diff(baseline_actions[left], baseline_actions[right]):.6g}, "
            f"residual={_max_abs_diff(residual_actions[left], residual_actions[right]):.6g}",
            flush=True,
        )
        env.generate_residual_reaching_plan_idx(
            env_ids, baseline_actions, residual_actions
        )
        env_action = env.compute_reference_actions()
        print(
            "  first env action max_abs_diff after real-yaw mapping: "
            f"all={_max_abs_diff(env_action[left], env_action[right]):.6g}, "
            f"base={_max_abs_diff(env_action[left, :env.hand_dof_start_idx], env_action[right, :env.hand_dof_start_idx]):.6g}, "
            f"hand={_max_abs_diff(env_action[left, env.hand_dof_start_idx:], env_action[right, env.hand_dof_start_idx:]):.6g}",
            flush=True,
        )

        yaw_quat = env.env_pregrasp_yaw_quat[right].view(1, 4)
        ref0_pos = env.tracking_reference["wrist_initobj_pos"][left]
        ref1_pos = env.tracking_reference["wrist_initobj_pos"][right]
        ref0_quat = env.tracking_reference["wrist_quat"][left]
        ref1_quat = env.tracking_reference["wrist_quat"][right]
        expected_ref1_pos = quat_apply(
            yaw_quat.expand(ref0_pos.shape[0], -1), ref0_pos
        )
        expected_ref1_quat = quat_mul(
            yaw_quat.expand(ref0_quat.shape[0], -1), ref0_quat
        )
        print(
            "  real reference equiv max_abs_diff: "
            f"wrist_initobj_pos={_max_abs_diff(ref1_pos, expected_ref1_pos):.6g}, "
            f"wrist_quat={_max_abs_diff(ref1_quat, expected_ref1_quat):.6g}",
            flush=True,
        )
        raw_eef = env.rigid_body_states.view(-1, 13)[env.eef_idx, 0:7]
        target_eef = torch.cat(
            [
                env.tracking_reference["wrist_initobj_pos"][:, 0]
                + env.object_init_states[:, 0:3],
                env.tracking_reference["wrist_quat"][:, 0],
            ],
            dim=-1,
        )
        canonical_raw_pos = env._canonicalize_world_positions(raw_eef[:, :3])
        canonical_raw_quat = env._canonicalize_world_quats(raw_eef[:, 3:7])
        canonical_target_pos = env._canonicalize_world_positions(
            target_eef[:, :3]
        )
        canonical_target_quat = env._canonicalize_world_quats(
            target_eef[:, 3:7]
        )
        print(
            "  raw reset eef target max_abs_diff: "
            f"env{left}.pos={_max_abs_diff(raw_eef[left, :3], target_eef[left, :3]):.6g}, "
            f"env{left}.quat={_max_abs_diff(raw_eef[left, 3:7], target_eef[left, 3:7]):.6g}, "
            f"env{right}.pos={_max_abs_diff(raw_eef[right, :3], target_eef[right, :3]):.6g}, "
            f"env{right}.quat={_max_abs_diff(raw_eef[right, 3:7], target_eef[right, 3:7]):.6g}",
            flush=True,
        )
        print(
            "  canonical raw/target pair max_abs_diff: "
            f"raw_pos={_max_abs_diff(canonical_raw_pos[left], canonical_raw_pos[right]):.6g}, "
            f"raw_quat={_max_abs_diff(canonical_raw_quat[left], canonical_raw_quat[right]):.6g}, "
            f"target_pos={_max_abs_diff(canonical_target_pos[left], canonical_target_pos[right]):.6g}, "
            f"target_quat={_max_abs_diff(canonical_target_quat[left], canonical_target_quat[right]):.6g}",
            flush=True,
        )
        raise SystemExit(0)

    def _run_compare_test(runner, base_rounds: int, scorer_rounds: int):
        env = runner.vec_env
        env_ids = torch.arange(env.num_envs, device=runner.device)
        total_rounds = int(base_rounds) + int(scorer_rounds)
        if total_rounds <= 0:
            raise ValueError("compare test needs at least one round")
        phase_success = {"base": [], "scorer": []}
        phase_angle_success = {"base": {}, "scorer": {}}
        runner.actor_critic.eval()
        with torch.no_grad():
            for round_idx in range(total_rounds):
                phase = "base" if round_idx < base_rounds else "scorer"
                if hasattr(env, "set_pregrasp_yaw_force_base_angle"):
                    env.set_pregrasp_yaw_force_base_angle(phase == "base")
                obs = env.reset_idx(env_ids)["obs"]
                states = env.get_state()
                baseline_actions = runner._baseline_actions(obs, states)
                residual_actions = runner.actor_critic(
                    obs, states, inference=True
                )
                env.generate_residual_reaching_plan_idx(
                    env_ids,
                    baseline_actions=baseline_actions,
                    residual_actions=residual_actions,
                )
                for step in range(env.max_episode_length):
                    env_action = env.compute_reference_actions()
                    env.step(env_action)
                    if step == env.max_episode_length - 2:
                        effective_success = torch.where(
                            env.has_hit_table,
                            torch.zeros_like(env.successes),
                            env.successes,
                        )
                        success = effective_success.float().mean().item()
                        phase_success[phase].append(success)
                        angle_rows = []
                        if hasattr(env, "env_angle_ids"):
                            for angle_id in torch.unique(env.env_angle_ids):
                                mask = env.env_angle_ids == angle_id
                                angle_index = int(angle_id.item())
                                angle_label = (
                                    env.format_angle_id(angle_index)
                                    if hasattr(env, "format_angle_id")
                                    else str(angle_index)
                                )
                                angle_rate = (
                                    effective_success[mask].float().mean().item()
                                )
                                angle_fraction = mask.float().mean().item()
                                phase_angle_success[phase].setdefault(
                                    angle_label, []
                                ).append(angle_rate)
                                angle_rows.append(
                                    f"{angle_label}={angle_rate:.3f} "
                                    f"({angle_fraction * 100.0:.1f}%)"
                                )
                        print(
                            f"Residual compare {round_idx + 1}/{total_rounds} "
                            f"[{phase}]: trajectory-rot=mixed, "
                            f"success={success:.3f}"
                            + (
                                " | " + ", ".join(angle_rows)
                                if angle_rows
                                else ""
                            ),
                            flush=True,
                        )
                        break
        if hasattr(env, "set_pregrasp_yaw_force_base_angle"):
            env.set_pregrasp_yaw_force_base_angle(False)
        for phase in ("base", "scorer"):
            if not phase_success[phase]:
                continue
            values = torch.tensor(
                phase_success[phase],
                dtype=torch.float32,
                device=runner.device,
            )
            print(
                f"Residual compare summary [{phase}]: "
                f"rounds={len(phase_success[phase])}, "
                f"mean={values.mean().item():.3f}, "
                f"std={values.std(unbiased=False).item():.3f}, "
                f"min={values.min().item():.3f}, "
                f"max={values.max().item():.3f}",
                flush=True,
            )
            rows = []
            for angle_label, rates in sorted(phase_angle_success[phase].items()):
                angle_values = torch.tensor(
                    rates, dtype=torch.float32, device=runner.device
                )
                rows.append(f"{angle_label}={angle_values.mean().item():.3f}")
            if rows:
                print(
                    f"Residual compare per-angle mean [{phase}]: "
                    + ", ".join(rows),
                    flush=True,
                )
        if phase_success["base"] and phase_success["scorer"]:
            base_mean = torch.tensor(
                phase_success["base"], dtype=torch.float32, device=runner.device
            ).mean()
            scorer_mean = torch.tensor(
                phase_success["scorer"], dtype=torch.float32, device=runner.device
            ).mean()
            print(
                "Residual compare delta: "
                f"scorer-base={(scorer_mean - base_mean).item():+.3f}",
                flush=True,
            )

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        with open_dict(cfg):
            cfg.test = True
            if cfg.get("checkpoint", "") and not cfg.get("residual_checkpoint", ""):
                cfg.residual_checkpoint = cfg.checkpoint
        configure_se_training(cfg)
        with open_dict(cfg):
            scorer_sampling = str(cfg.get("pregrasp_yaw_sampling", "scorer_top1"))
            cfg.task.env.pregraspYawAngles = _as_float_list(
                cfg.get("pregrasp_yaw_angles", [0.0])
            )
            cfg.task.env.pregraspYawFrame = str(
                cfg.get("pregrasp_yaw_frame", "pca_short")
            )
            cfg.task.env.pregraspYawScorerSampling = scorer_sampling
            cfg.task.env.worldTiltSampling = "cycle"
            cfg.task.env.seSampling = cfg.task.env.worldTiltSampling
            cfg.task.env.worldTiltLogInterval = int(
                cfg.get("pregrasp_eval_log_interval", 0)
            )
            cfg.task.env.pregraspYawCanonicalObs = bool(
                cfg.get("pregrasp_yaw_canonical_obs", True)
            )
            cfg.task.env.pregraspYawCanonicalObsScope = str(
                cfg.get("pregrasp_yaw_canonical_obs_scope", "scene")
            )
            cfg.task.env.pregraspYawDirectBaseQpos = bool(
                cfg.get("pregrasp_yaw_direct_base_qpos", True)
            )
            cfg.task.env.pregraspYawScorerCheckpoint = str(
                cfg.get("pregrasp_yaw_scorer_checkpoint", "")
            )
            cfg.task.env.pregraspYawScorerTopK = int(
                cfg.get("pregrasp_yaw_scorer_top_k", 1)
            )
            cfg.task.env.pregraspYawScorerChunkSize = int(
                cfg.get("pregrasp_yaw_scorer_chunk_size", 8192)
            )
            compare_base_angle = cfg.get("pregrasp_yaw_compare_base_angle", None)
            if compare_base_angle is not None:
                cfg.task.env.pregraspYawCompareBaseAngle = _as_float_list(
                    compare_base_angle
                )
            cfg.task.env.pregraspYawCompareBaseSampling = str(
                cfg.get("pregrasp_yaw_compare_base_sampling", "random")
            )
            cfg.task.env.pregraspYawCompareBaseMode = str(
                cfg.get("pregrasp_yaw_compare_base_mode", "yaw0")
            )
        log_viewer_device(cfg)
        print(
            "Yaw-scorer SE residual eval: "
            f"yaw={list(cfg.task.env.pregraspYawAngles)}, "
            f"yaw_frame={cfg.task.env.pregraspYawFrame}, "
            f"tilt={list(cfg.task.env.seTiltAngles)}, "
            f"pitch={list(cfg.task.env.sePitchAngles)}, "
            f"sampling={cfg.task.env.pregraspYawScorerSampling}, "
            f"scorer_checkpoint={cfg.task.env.pregraspYawScorerCheckpoint}, "
            f"scorer_top_k={cfg.task.env.pregraspYawScorerTopK}, "
            f"scorer_chunk={cfg.task.env.pregraspYawScorerChunkSize}, "
            f"compare_base_mode={cfg.task.env.pregraspYawCompareBaseMode}, "
            f"compare_base_sampling={cfg.task.env.pregraspYawCompareBaseSampling}, "
            f"canonical_obs={cfg.task.env.pregraspYawCanonicalObs}, "
            f"canonical_scope={cfg.task.env.pregraspYawCanonicalObsScope}, "
            f"direct_base_qpos={cfg.task.env.pregraspYawDirectBaseQpos}, "
            f"checkpoint={cfg.checkpoint}, "
            f"residual_checkpoint={cfg.get('residual_checkpoint', '')}",
            flush=True,
        )

        isaacgym_task_map[TASK_NAME] = YawScorerSEResidualGrasp
        rank = int(os.getenv("RANK", "0"))
        cfg.seed = set_seed(
            cfg.seed,
            torch_deterministic=cfg.torch_deterministic,
            rank=rank,
        )
        env = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
        )
        if cfg.capture_video:
            env.is_vector_env = True
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{TASK_NAME}_yaw_scorer_eval",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner = build_runner(cfg, env)
        if bool(cfg.get("pregrasp_yaw_debug_equiv", False)):
            _run_yaw_equiv_debug(env, runner)
        compare_base_rounds = int(cfg.get("pregrasp_yaw_compare_base_rounds", 0))
        compare_scorer_rounds = int(
            cfg.get(
                "pregrasp_yaw_compare_scorer_rounds",
                cfg.get("pregrasp_yaw_compare_field_rounds", 0),
            )
        )
        if compare_base_rounds > 0 or compare_scorer_rounds > 0:
            if compare_base_rounds <= 0:
                compare_base_rounds = 5
            if compare_scorer_rounds <= 0:
                compare_scorer_rounds = 5
            _run_compare_test(runner, compare_base_rounds, compare_scorer_rounds)
            return
        runner.run()

    _main()


if __name__ == "__main__":
    _normalize_cli_aliases(sys.argv)
    hydra_main()
