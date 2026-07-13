"""Evaluate SE residual PPO with an extra object-centered yaw sweep.

The trained SE residual policy uses the original tilt/pitch guide observation.
This entry expands the reset/reference family by rotating the whole tilt/pitch
reference trajectory around the object/table Z axis.  yaw=0 exactly recovers the
original tilt/pitch family.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACGYM_ROOT = PROJECT_ROOT / "thirdparty" / "isaacgym" / "python"
ISAACGYM_ENVS_ROOT = PROJECT_ROOT / "thirdparty" / "IsaacGymEnvs"


def _add_project_paths() -> None:
    for path in (PROJECT_ROOT, ISAACGYM_ROOT, ISAACGYM_ENVS_ROOT):
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
    import torch
    class YawExpandedSEResidualGrasp(SEResidualGrasp):
        """Tilt/pitch SE residual task with an extra yaw around Z."""

        def init_configs(self, cfg):
            env_cfg = cfg["env"]
            self.pregrasp_yaw_angles_cfg = _as_float_list(
                env_cfg.get("pregraspYawAngles", [0.0])
            )
            self.pregrasp_yaw_canonical_obs = bool(
                env_cfg.get("pregraspYawCanonicalObs", True)
            )
            self.pregrasp_yaw_canonical_obs_scope = str(
                env_cfg.get("pregraspYawCanonicalObsScope", "hand")
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
            self.env_pregrasp_yaw_quat = self.identity_quat.view(1, 4).repeat(
                self.num_envs, 1
            )
            self.env_pregrasp_yaw_quat_inv = (
                self.identity_quat.view(1, 4).repeat(self.num_envs, 1)
            )
            self.env_traj_normal_canonical = self.table_normal.view(
                1, 3
            ).repeat(self.num_envs, 1)
            print(
                "Yaw-expanded SE residual eval env: "
                f"yaw={self.pregrasp_yaw_angles_cfg}, "
                f"tilt/pitch={self.se_angle_pairs_cfg}, "
                f"candidates={len(self.pregrasp_yaw_angle_triples_cfg)}, "
                f"sampling={self.world_tilt_sampling}, "
                f"canonical_obs={self.pregrasp_yaw_canonical_obs}, "
                f"canonical_scope={self.pregrasp_yaw_canonical_obs_scope}, "
                f"direct_base_qpos={self.pregrasp_yaw_direct_base_qpos}",
                flush=True,
            )

        def _sample_trajectory_angle_ids(self, env_ids):
            count = len(env_ids)
            candidate_count = len(self.pregrasp_yaw_angle_triples_cfg)
            if candidate_count == 1:
                return torch.zeros(count, dtype=torch.long, device=self.device)
            if self.world_tilt_sampling == "cycle":
                ids = (
                    torch.arange(count, device=self.device)
                    + self._tilt_cycle_index
                ) % candidate_count
                self._tilt_cycle_index = (
                    self._tilt_cycle_index + count
                ) % candidate_count
                return ids.long()
            return torch.randint(
                0, candidate_count, (count,), dtype=torch.long, device=self.device
            )

        def format_angle_id(self, angle_id):
            yaw, tilt, pitch = self.pregrasp_yaw_angle_triples_cfg[
                int(angle_id)
            ]
            return f"yaw{yaw:g}_tilt{tilt:g}_pitch{pitch:g}"

        def _set_env_trajectory_rotations(self, env_ids, angle_ids):
            self._ensure_base_tracking_reference()
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            angle_ids = angle_ids.to(device=self.device, dtype=torch.long)
            triples = self.pregrasp_yaw_angle_triples[angle_ids]
            yaw_deg = triples[:, 0]
            tilt_deg = triples[:, 1]
            pitch_deg = triples[:, 2]

            base_guide_quat = self._guide_quat_from_tilt_pitch(
                env_ids, tilt_deg, pitch_deg
            )
            z_axis = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device
            ).expand(len(env_ids), -1)
            yaw_quat = quat_from_angle_axis(torch.deg2rad(yaw_deg), z_axis)
            yaw_quat_inv = quat_conjugate(yaw_quat)
            guide_quat = quat_mul(yaw_quat, base_guide_quat)

            self.env_angle_ids[env_ids] = angle_ids
            self.env_pregrasp_yaw[env_ids] = yaw_deg
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

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        with open_dict(cfg):
            cfg.test = True
            if cfg.get("checkpoint", "") and not cfg.get("residual_checkpoint", ""):
                cfg.residual_checkpoint = cfg.checkpoint
        configure_se_training(cfg)
        with open_dict(cfg):
            cfg.task.env.pregraspYawAngles = _as_float_list(
                cfg.get("pregrasp_yaw_angles", [0.0])
            )
            cfg.task.env.worldTiltSampling = cfg.get(
                "pregrasp_yaw_sampling", cfg.get("se_sampling", "random")
            )
            cfg.task.env.seSampling = cfg.task.env.worldTiltSampling
            cfg.task.env.worldTiltLogInterval = int(
                cfg.get("pregrasp_eval_log_interval", 0)
            )
            cfg.task.env.pregraspYawCanonicalObs = bool(
                cfg.get("pregrasp_yaw_canonical_obs", True)
            )
            cfg.task.env.pregraspYawCanonicalObsScope = str(
                cfg.get("pregrasp_yaw_canonical_obs_scope", "hand")
            )
            cfg.task.env.pregraspYawDirectBaseQpos = bool(
                cfg.get("pregrasp_yaw_direct_base_qpos", True)
            )
        log_viewer_device(cfg)
        print(
            "Yaw-expanded SE residual eval: "
            f"yaw={list(cfg.task.env.pregraspYawAngles)}, "
            f"tilt={list(cfg.task.env.seTiltAngles)}, "
            f"pitch={list(cfg.task.env.sePitchAngles)}, "
            f"sampling={cfg.task.env.worldTiltSampling}, "
            f"canonical_obs={cfg.task.env.pregraspYawCanonicalObs}, "
            f"canonical_scope={cfg.task.env.pregraspYawCanonicalObsScope}, "
            f"direct_base_qpos={cfg.task.env.pregraspYawDirectBaseQpos}, "
            f"checkpoint={cfg.checkpoint}, "
            f"residual_checkpoint={cfg.get('residual_checkpoint', '')}",
            flush=True,
        )

        isaacgym_task_map[TASK_NAME] = YawExpandedSEResidualGrasp
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
                f"videos/{TASK_NAME}_yaw_expanded_eval",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner = build_runner(cfg, env)
        if bool(cfg.get("pregrasp_yaw_debug_equiv", False)):
            _run_yaw_equiv_debug(env, runner)
        runner.run()

    _main()


if __name__ == "__main__":
    _normalize_cli_aliases(sys.argv)
    hydra_main()
