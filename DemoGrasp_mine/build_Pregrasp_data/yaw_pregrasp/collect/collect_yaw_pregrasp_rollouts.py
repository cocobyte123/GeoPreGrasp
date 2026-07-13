"""Collect yaw-expanded pregrasp rollout labels.

This collector samples yaw-expanded pregrasp candidates in the object-centric
frame that matched the validated rollout behavior:

* The default yaw frame is the projected object short PCA axis.
* real_yaw = reference_yaw + candidate_yaw.
* PPO receives a scene-canonical observation rotated by -real_yaw.
* The virtual base is controlled in continuous qpos space to avoid Euler
  branch flips.

The exported labels estimate:

    P(success | object, object_pose, yaw, tilt, pitch)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ISAACGYM_ROOT = PROJECT_ROOT / "thirdparty" / "isaacgym" / "python"
ISAACGYM_ENVS_ROOT = PROJECT_ROOT / "thirdparty" / "IsaacGymEnvs"
DEFAULT_ROLLOUT_ROOT = (
    WORKSPACE_ROOT / "PregraspPrior" / "data" / "rollouts" / "yaw_pregrasp"
)
DEFAULT_YAW_ANGLES = [float(v) for v in range(0, 360, 30)]
DEFAULT_TILT_PITCH = [0.0, 15.0, 30.0]

for path in (PROJECT_ROOT, ISAACGYM_ROOT, ISAACGYM_ENVS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import isaacgym  # noqa: E402,F401
import torch  # noqa: E402
from isaacgymenvs.utils.torch_jit_utils import (  # noqa: E402
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    unscale,
)

from build_Pregrasp_data.yaw_pregrasp.io.schema import (  # noqa: E402
    ROLLOUT_REQUIRED_KEYS,
)
from residual_se_grasp.se_residual_grasp import SEResidualGrasp  # noqa: E402
from residual_se_grasp.train_se_reslearn import (  # noqa: E402
    TASK_NAME,
    _prepare_embedded_baseline_if_needed,
    configure_se_training,
    se_residual_metadata,
)
from residual_tilt_grasp.tilted_hand_only_grasp import _quat_to_base_angles_zyx  # noqa: E402
from residual_tilt_grasp.train_tilted_hand_only_reslearn import (  # noqa: E402
    get_checkpoint_paths,
    log_viewer_device,
)


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
                converted.append(
                    hydra_name
                    + _convert_csv_arg_to_hydra_list(arg.split("=", 1)[1])
                )
                matched = True
                break
            if arg == cli_name and index + 1 < len(argv):
                converted.append(
                    hydra_name + _convert_csv_arg_to_hydra_list(argv[index + 1])
                )
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
            return [float(item) for item in json.loads(text)]
        return [float(item) for item in text.split(",") if item.strip()]
    return [float(item) for item in value]


def _plain_config(value):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def _clear_rollout_root(root: Path) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob("*/batch_*.npz"):
        path.unlink()
        removed += 1
    manifest = root / "manifest.json"
    if manifest.is_file():
        manifest.unlink()
        removed += 1
    return removed


class YawPregraspCollectGrasp(SEResidualGrasp):
    """Yaw-expanded SE residual env with rollout export helpers."""

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.pregrasp_yaw_angles_cfg = _as_float_list(
            env_cfg.get("pregraspYawAngles", DEFAULT_YAW_ANGLES)
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
        self.yaw_collect_rollout_root = str(
            env_cfg.get("yawPregraspCollectRolloutRoot", DEFAULT_ROLLOUT_ROOT)
        )
        self.yaw_collect_overwrite_root = bool(
            env_cfg.get("yawPregraspCollectOverwriteRoot", True)
        )
        if not self.pregrasp_yaw_angles_cfg:
            raise ValueError("pregraspYawAngles must not be empty")
        super().init_configs(cfg)
        if self.se_guide_mode != "tilt_pitch":
            raise ValueError("Yaw pregrasp collection expects seGuideMode=tilt_pitch")
        self.pregrasp_yaw_angle_triples_cfg = [
            (yaw, float(tilt), float(pitch))
            for yaw in self.pregrasp_yaw_angles_cfg
            for tilt, pitch in self.se_angle_pairs_cfg
        ]
        self.world_tilt_angles = [
            float(index) for index in range(len(self.pregrasp_yaw_angle_triples_cfg))
        ]
        self._angle_success_sum = {angle: 0.0 for angle in self.world_tilt_angles}
        self._angle_success_count = {angle: 0 for angle in self.world_tilt_angles}
        self._angle_hit_sum = {angle: 0.0 for angle in self.world_tilt_angles}
        self._angle_clearance_sum = {angle: 0.0 for angle in self.world_tilt_angles}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(self.yaw_collect_rollout_root).expanduser().resolve()
        if self.yaw_collect_overwrite_root:
            removed = _clear_rollout_root(root)
            if removed:
                print(
                    f"Cleared yaw pregrasp rollout root: {root}, removed={removed}",
                    flush=True,
                )
        root.mkdir(parents=True, exist_ok=True)
        self.yaw_collect_object_batch_counts: Dict[str, int] = {}
        self.yaw_collect_entries: List[Dict] = []
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
        self.env_pregrasp_yaw_quat_inv = self.identity_quat.view(1, 4).repeat(
            self.num_envs, 1
        )
        self.env_traj_normal_canonical = self.table_normal.view(1, 3).repeat(
            self.num_envs, 1
        )
        self.initial_object_pose_world = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self.initial_real_palm_pose_world = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self.initial_canonical_palm_pose_world = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self.initial_table_height = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        print(
            "Yaw pregrasp collector env: "
            f"yaw={self.pregrasp_yaw_angles_cfg}, "
            f"tilt/pitch={self.se_angle_pairs_cfg}, "
            f"candidates={len(self.pregrasp_yaw_angle_triples_cfg)}, "
            f"sampling={self.world_tilt_sampling}, "
            f"yaw_frame={self.pregrasp_yaw_frame}, "
            f"canonical_obs={self.pregrasp_yaw_canonical_obs}, "
            f"canonical_scope={self.pregrasp_yaw_canonical_obs_scope}, "
            f"rollout_root={root}",
            flush=True,
        )

    def _sample_trajectory_angle_ids(self, env_ids):
        count = len(env_ids)
        candidate_count = len(self.pregrasp_yaw_angle_triples_cfg)
        if candidate_count == 1:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        if self.world_tilt_sampling == "cycle":
            ids = (
                torch.arange(count, device=self.device) + self._tilt_cycle_index
            ) % candidate_count
            self._tilt_cycle_index = (self._tilt_cycle_index + count) % candidate_count
            return ids.long()
        return torch.randint(
            0, candidate_count, (count,), dtype=torch.long, device=self.device
        )

    def format_angle_id(self, angle_id):
        yaw, tilt, pitch = self.pregrasp_yaw_angle_triples_cfg[int(angle_id)]
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
        # The parent reset samples the angle before the object pose exists.
        # Re-apply after sampling the object pose so geometric frames are valid.
        self._set_env_trajectory_rotations(env_ids, self.env_angle_ids[env_ids])
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
        guide_quat = quat_mul(yaw_quat, base_guide_quat)

        self.env_angle_ids[env_ids] = angle_ids
        self.env_pregrasp_yaw[env_ids] = yaw_deg
        self.env_pregrasp_world_yaw[env_ids] = world_yaw_deg
        self.env_pregrasp_yaw_quat[env_ids] = yaw_quat
        self.env_pregrasp_yaw_quat_inv[env_ids] = quat_conjugate(yaw_quat)
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
            quat_seq, self.base_tracking_reference["wrist_initobj_pos"][env_ids]
        )
        self.tracking_reference["wrist_quat"][env_ids] = quat_mul(
            quat_seq, self.base_tracking_reference["wrist_quat"][env_ids]
        )

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
            flat_inv.reshape(-1, 4), flat_quats.reshape(-1, 4)
        ).reshape_as(flat_quats)
        sign = torch.where(
            canonical[..., 3:4] < 0.0,
            -torch.ones_like(canonical[..., 3:4]),
            torch.ones_like(canonical[..., 3:4]),
        )
        return (canonical * sign).reshape(shape)

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
                view[:, 0:3] = self._canonicalize_world_positions(view[:, 0:3])
                view[:, 3:7] = self._canonicalize_world_quats(view[:, 3:7])
            elif part == "ftpos":
                view[:] = self._canonicalize_world_positions(
                    view.reshape(self.num_envs, -1, 3)
                ).reshape(self.num_envs, -1)
            elif self.pregrasp_yaw_canonical_obs_scope == "scene" and part in (
                "objpose",
                "objinitpose",
            ):
                view[:, 0:3] = self._canonicalize_world_positions(view[:, 0:3])
                view[:, 3:7] = self._canonicalize_world_quats(view[:, 3:7])
            elif self.pregrasp_yaw_canonical_obs_scope == "scene" and part == "objxyz":
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
                f"Yaw canonical observation layout mismatch: {obs_end} != {num_obs}"
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
        pcl_dim = self.points_per_object * 3 if "objpcl" in self.obs_type else 0
        state_dim = residual_obs.shape[-1] - pcl_dim - self.GUIDE_OBS_DIM
        state_obs = residual_obs[:, :state_dim]
        pcl_obs = residual_obs[:, state_dim + self.GUIDE_OBS_DIM :]
        if "objpcl" in self.obs_type.split("+"):
            return torch.cat(
                [state_obs, self.env_traj_normal_canonical, pcl_obs], dim=-1
            )
        return torch.cat([state_obs, self.env_traj_normal_canonical], dim=-1)

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
        reaching_mask = (self.progress_buf < self.reaching_plan_timesteps).unsqueeze(-1)
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
        angles = _quat_to_base_angles_zyx(quats.reshape(-1, 4)).reshape(
            positions.shape[0], positions.shape[1], 3
        )
        angles = self._unwrap_angle_sequence(angles)
        return torch.cat([positions, angles], dim=-1)

    def _prepare_direct_base_qpos_plans(self, env_ids):
        if not hasattr(self, "reaching_plan_base_qpos"):
            self.reaching_plan_base_qpos = torch.zeros(
                (self.num_envs, self.max_episode_length, self.num_arm_dofs),
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
        tracking_quats = self.current_tracking_reference["wrist_quat"][env_ids]
        tracking_qpos = self._wrist_pose_to_base_qpos_sequence(
            tracking_positions, tracking_quats
        )
        start_qpos = self.robot_dof_pos[:, self.arm_dof_indices][env_ids]
        angle_offset = (
            torch.round(
                (start_qpos[:, 3:6] - tracking_qpos[:, 0, 3:6])
                / (2.0 * torch.pi)
            )
            * (2.0 * torch.pi)
        )
        tracking_qpos[:, :, 3:6] = tracking_qpos[:, :, 3:6] + angle_offset.unsqueeze(1)
        self.tracking_reference_base_qpos[env_ids] = tracking_qpos
        target_qpos = tracking_qpos[:, 0]
        delta = target_qpos - start_qpos
        delta[:, 3:6] = torch.atan2(torch.sin(delta[:, 3:6]), torch.cos(delta[:, 3:6]))
        steps = torch.arange(
            self.max_episode_length, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)
        denom = self.reaching_plan_timesteps[env_ids].float().clamp_min(1.0).view(
            -1, 1, 1
        )
        fraction = ((steps + 1.0) / denom).clamp(max=1.0)
        self.reaching_plan_base_qpos[env_ids] = start_qpos.unsqueeze(1) + fraction * delta.unsqueeze(1)

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

    def reset_idx(self, env_ids, object_init_pose=None, **kwargs):
        result = super().reset_idx(env_ids, object_init_pose=object_init_pose, **kwargs)
        self._snapshot_initial_rollout_state(env_ids)
        return result

    def _snapshot_initial_rollout_state(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.initial_object_pose_world[env_ids] = self.object_init_states[
            env_ids, 0:7
        ].to(dtype=torch.float32)
        self.initial_table_height[env_ids] = self.table_heights[env_ids].to(
            dtype=torch.float32
        )
        palm_pos = self.palm_center_pos[env_ids].to(dtype=torch.float32)
        palm_quat = self.palm_rot[env_ids].to(dtype=torch.float32)
        self.initial_real_palm_pose_world[env_ids] = torch.cat(
            [palm_pos, palm_quat], dim=-1
        )
        canonical_pos = self._canonicalize_world_positions(self.palm_center_pos)[
            env_ids
        ].to(dtype=torch.float32)
        canonical_quat = self._canonicalize_world_quats(self.palm_rot)[env_ids].to(
            dtype=torch.float32
        )
        self.initial_canonical_palm_pose_world[env_ids] = torch.cat(
            [canonical_pos, canonical_quat], dim=-1
        )

    def _object_name_for_env_id(self, env_id: int) -> str:
        object_fn = self.object_fns[int(env_id) % len(self.object_fns)]
        return Path(object_fn).stem

    def export_yaw_pregrasp_rollout_batch(
        self, success_values: torch.Tensor, batch_id: int
    ) -> List[Path]:
        root = Path(self.yaw_collect_rollout_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        success = success_values.detach().to(device=self.device).reshape(-1).bool()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        by_object: Dict[str, List[int]] = {}
        for env_id in env_ids.detach().cpu().tolist():
            by_object.setdefault(self._object_name_for_env_id(int(env_id)), []).append(
                int(env_id)
            )

        written: List[Path] = []
        triples = self.pregrasp_yaw_angle_triples[self.env_angle_ids]
        for object_name, ids in sorted(by_object.items()):
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            object_root = root / object_name
            object_root.mkdir(parents=True, exist_ok=True)
            local_batch_id = self.yaw_collect_object_batch_counts.get(object_name, 0)
            self.yaw_collect_object_batch_counts[object_name] = local_batch_id + 1
            path = object_root / f"batch_{local_batch_id:06d}.npz"

            object_asset = self.object_fns[ids[0] % len(self.object_fns)]
            points = np.empty((0, 3), dtype=np.float32)
            if hasattr(self, "obj_pcl_buf"):
                points = (
                    self.obj_pcl_buf[index[0]]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

            np.savez_compressed(
                path,
                schema=np.asarray("YawPregraspRollout/v1"),
                required_keys=np.asarray(ROLLOUT_REQUIRED_KEYS),
                object_id=np.asarray(object_name),
                object_asset=np.asarray(object_asset),
                object_pose_world=self.initial_object_pose_world[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                table_height=self.initial_table_height[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                yaw_deg=triples[index, 0].detach().cpu().numpy().astype(np.float32),
                world_yaw_deg=self.env_pregrasp_world_yaw[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                object_yaw_deg=torch.rad2deg(self.env_pregrasp_object_yaw[index])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                reference_yaw_deg=torch.rad2deg(
                    self.env_pregrasp_reference_yaw[index]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                tilt_deg=triples[index, 1].detach().cpu().numpy().astype(np.float32),
                pitch_deg=triples[index, 2].detach().cpu().numpy().astype(np.float32),
                angle_id=self.env_angle_ids[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                success=success[index].detach().cpu().numpy().astype(bool),
                hit_table=self.has_hit_table[index]
                .detach()
                .cpu()
                .numpy()
                .astype(bool),
                min_clearance=self.episode_min_hand_table_clearance[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                real_palm_pose_world=self.initial_real_palm_pose_world[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                canonical_palm_pose_world=self.initial_canonical_palm_pose_world[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                object_point_cloud_object=points,
            )
            entry = {
                "global_batch_id": int(batch_id),
                "object_id": object_name,
                "object_asset": object_asset,
                "path": str(path),
                "env_count": int(len(ids)),
                "success_total": int(success[index].sum().item()),
            }
            self.yaw_collect_entries.append(entry)
            written.append(path)
        return written

    def write_yaw_pregrasp_manifest(self, config: Dict) -> Path:
        root = Path(self.yaw_collect_rollout_root).expanduser().resolve()
        payload = {
            "schema": "YawPregraspRolloutManifest/v1",
            "root": str(root),
            "object_batch_counts": self.yaw_collect_object_batch_counts,
            "entries": self.yaw_collect_entries,
            "config": config,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class YawPregraspRolloutExportPPO:
    def __init__(self, runner, max_batches: int, manifest_config: Dict):
        self.runner = runner
        self.max_batches = int(max_batches)
        self.manifest_config = manifest_config

    def run(self):
        runner = self.runner
        env = runner.vec_env
        env_ids = torch.arange(env.num_envs, device=runner.device)
        runner.actor_critic.eval()
        success_rates = []
        with torch.no_grad():
            for batch_id in range(self.max_batches):
                obs = env.reset_idx(env_ids)["obs"]
                states = env.get_state()
                baseline_actions = runner._baseline_actions(obs, states)
                residual_actions = runner.actor_critic(obs, states, inference=True)
                env.generate_residual_reaching_plan_idx(
                    env_ids,
                    baseline_actions=baseline_actions,
                    residual_actions=residual_actions,
                )
                effective_success = None
                for step in range(env.max_episode_length):
                    env_action = env.compute_reference_actions()
                    env.step(env_action)
                    if step == env.max_episode_length - 2:
                        effective_success = torch.where(
                            env.has_hit_table,
                            torch.zeros_like(env.successes),
                            env.successes,
                        )
                        break
                if effective_success is None:
                    raise RuntimeError("rollout finished without computing success")
                paths = env.export_yaw_pregrasp_rollout_batch(
                    effective_success, batch_id=batch_id
                )
                success_rate = float(effective_success.float().mean().item())
                success_rates.append(success_rate)
                print(
                    f"Yaw pregrasp rollout batch {batch_id + 1}/{self.max_batches}: "
                    f"success={success_rate:.4f}, files={len(paths)}",
                    flush=True,
                )
        manifest = env.write_yaw_pregrasp_manifest(self.manifest_config)
        values = torch.tensor(success_rates, dtype=torch.float32, device=runner.device)
        print(
            "Yaw pregrasp rollout export summary: "
            f"batches={len(success_rates)}, mean_success={values.mean().item():.4f}, "
            f"manifest={manifest}",
            flush=True,
        )


def _build_runner(cfg, env):
    _prepare_embedded_baseline_if_needed(cfg)
    baseline_checkpoint, residual_checkpoint = get_checkpoint_paths(cfg)
    if not residual_checkpoint:
        raise ValueError("collection requires checkpoint/residual_checkpoint")

    from residual_tilt_grasp.residual_actor_critic import ResidualActorCritic
    from residual_tilt_grasp.residual_ppo import ResidualPPO

    runner = ResidualPPO(
        vec_env=env,
        actor_critic_class=ResidualActorCritic,
        train_param=cfg.train.params,
        log_dir=None,
        apply_reset=False,
        action_dim=env.residual_action_dim,
        baseline_checkpoint=baseline_checkpoint,
        residual_metadata=se_residual_metadata(cfg, env),
        baseline_checkpoint_hash_override=cfg.get(
            "embedded_baseline_checkpoint_hash", ""
        ),
        per_angle_advantage=bool(cfg.get("per_angle_advantage", True)),
        advantage_eps=float(cfg.get("advantage_eps", 1e-8)),
        advantage_min_samples=int(cfg.get("advantage_min_samples", 32)),
        zero_inactive_angle_advantage=bool(
            cfg.get("zero_inactive_angle_advantage", True)
        ),
        freeze_residual_backbone=bool(cfg.get("residual_freeze_backbone", True)),
        initialize_residual_features=bool(
            cfg.get("residual_initialize_from_baseline", True)
        ),
    )
    runner.load_residual(residual_checkpoint, resume=False)
    runner.actor_critic.eval()
    return runner, baseline_checkpoint, residual_checkpoint


def hydra_main():
    import gym
    import hydra
    import isaacgymenvs
    from isaacgym import gymapi  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from omegaconf import DictConfig, open_dict

    import tasks  # noqa: F401

    @hydra.main(version_base="1.3", config_path="../../../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        with open_dict(cfg):
            cfg.test = True
            cfg.train.params.test = True
            if cfg.get("checkpoint", "") and not cfg.get("residual_checkpoint", ""):
                cfg.residual_checkpoint = cfg.checkpoint
            cfg.se_guide_mode = "tilt_pitch"
            cfg.se_sampling = cfg.get("pregrasp_yaw_sampling", "cycle")
        configure_se_training(cfg)
        with open_dict(cfg):
            cfg.task.env.pregraspYawAngles = _as_float_list(
                cfg.get("pregrasp_yaw_angles", DEFAULT_YAW_ANGLES)
            )
            cfg.task.env.pregraspYawFrame = str(
                cfg.get("pregrasp_yaw_frame", "pca_short")
            )
            cfg.task.env.seSampling = cfg.get("pregrasp_yaw_sampling", "cycle")
            cfg.task.env.worldTiltSampling = cfg.task.env.seSampling
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
            cfg.task.env.yawPregraspCollectRolloutRoot = str(
                Path(
                    cfg.get("yaw_pregrasp_collect_rollout_root", DEFAULT_ROLLOUT_ROOT)
                ).expanduser()
            )
            cfg.task.env.yawPregraspCollectOverwriteRoot = bool(
                cfg.get("yaw_pregrasp_collect_overwrite_root", True)
            )
        log_viewer_device(cfg)
        print(
            "Yaw pregrasp rollout collection: "
            f"rollout_root={cfg.task.env.yawPregraspCollectRolloutRoot}, "
            f"yaw={list(cfg.task.env.pregraspYawAngles)}, "
            f"yaw_frame={cfg.task.env.pregraspYawFrame}, "
            f"tilt={list(cfg.task.env.seTiltAngles)}, "
            f"pitch={list(cfg.task.env.sePitchAngles)}, "
            f"sampling={cfg.task.env.worldTiltSampling}, "
            f"batches={cfg.get('yaw_pregrasp_collect_max_batches', 1)}, "
            f"num_envs={cfg.task.env.numEnvs}, "
            f"residual_checkpoint={cfg.get('residual_checkpoint', '')}",
            flush=True,
        )

        isaacgym_task_map[TASK_NAME] = YawPregraspCollectGrasp
        rank = int(os.getenv("RANK", "0"))
        cfg.seed = set_seed(
            cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank
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
                f"videos/{TASK_NAME}_yaw_pregrasp_collect",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner, baseline_checkpoint, residual_checkpoint = _build_runner(cfg, env)
        manifest_config = {
            "source": "build_Pregrasp_data.yaw_pregrasp.collect.collect_yaw_pregrasp_rollouts",
            "residual_checkpoint": str(residual_checkpoint),
            "baseline_checkpoint": str(baseline_checkpoint),
            "num_envs": int(cfg.task.env.numEnvs),
            "max_batches": int(cfg.get("yaw_pregrasp_collect_max_batches", 1)),
            "yaw_angles": list(cfg.task.env.pregraspYawAngles),
            "yaw_frame": str(cfg.task.env.pregraspYawFrame),
            "tilt_angles": list(cfg.task.env.seTiltAngles),
            "pitch_angles": list(cfg.task.env.sePitchAngles),
            "canonical_obs": bool(cfg.task.env.pregraspYawCanonicalObs),
            "canonical_obs_scope": str(cfg.task.env.pregraspYawCanonicalObsScope),
            "direct_base_qpos": bool(cfg.task.env.pregraspYawDirectBaseQpos),
            "multi_object": bool(cfg.task.env.asset.multiObject),
            "multi_object_list": str(cfg.task.env.asset.multiObjectList),
            "reset_position_range": _plain_config(cfg.task.env.resetPositionRange),
            "reset_random_rot": str(cfg.task.env.resetRandomRot),
        }
        YawPregraspRolloutExportPPO(
            runner,
            max_batches=int(cfg.get("yaw_pregrasp_collect_max_batches", 1)),
            manifest_config=manifest_config,
        ).run()

    _main()


if __name__ == "__main__":
    _normalize_cli_aliases(sys.argv)
    hydra_main()
