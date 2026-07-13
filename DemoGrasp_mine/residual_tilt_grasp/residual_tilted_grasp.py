"""Tilted grasp environment with frozen-baseline residual trajectory edits."""

import math
from copy import deepcopy

import torch

from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    tensor_clamp,
)

from residual_tilt_grasp.tilted_hand_only_grasp import TiltedHandOnlyGrasp
from tasks.utils import batch_linear_interpolate_poses


class ResidualTiltedGrasp(TiltedHandOnlyGrasp):
    """Decode bounded wrist/finger residuals without changing robot actions."""

    VALID_MODES = ("wrist", "finger", "hybrid")

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.residual_mode = env_cfg.get("residualMode", "hybrid")
        if self.residual_mode not in self.VALID_MODES:
            raise ValueError(
                f"residualMode must be one of {self.VALID_MODES}, "
                f"got {self.residual_mode}"
            )

        self.residual_frame = env_cfg.get("residualFrame", "table")
        if self.residual_frame != "table":
            raise ValueError("First implementation requires residualFrame=table")
        self.residual_scope = env_cfg.get(
            "residualScope", "full_trajectory"
        )
        if self.residual_scope != "full_trajectory":
            raise ValueError(
                "First implementation requires "
                "residualScope=full_trajectory"
            )
        self.residual_pose_composition = env_cfg.get(
            "residualPoseComposition", "full_se3"
        )
        if self.residual_pose_composition != "full_se3":
            raise ValueError(
                "First implementation requires "
                "residualPoseComposition=full_se3"
            )

        self.residual_translation_scale_cfg = env_cfg.get(
            "residualTranslationScale", [0.02, 0.02, 0.02]
        )
        self.residual_rotation_scale_deg_cfg = env_cfg.get(
            "residualRotationScaleDeg", [10.0, 10.0, 10.0]
        )
        self.finger_residual_alpha = float(
            env_cfg.get("fingerResidualAlpha", 0.1)
        )
        self.finger_residual_min_ratio = float(
            env_cfg.get("fingerResidualMinRatio", 0.01)
        )
        self.finger_residual_max_ratio = float(
            env_cfg.get("fingerResidualMaxRatio", 0.15)
        )

        self.tilt_modulation = env_cfg.get(
            "residualTiltModulation", "normalized_sin"
        )
        if self.tilt_modulation not in (
            "normalized_sin", "sin", "linear", "none"
        ):
            raise ValueError(
                "residualTiltModulation must be normalized_sin, sin, "
                "linear, or none"
            )
        self.tilt_modulation_max_angle = float(
            env_cfg.get(
                "residualTiltModulationMaxAngle",
                max(
                    abs(float(angle))
                    for angle in env_cfg.get("worldTiltAngles", [0.0])
                ),
            )
        )
        self.wrist_tilt_modulation = bool(
            env_cfg.get("wristTiltModulation", True)
        )
        self.finger_tilt_modulation = bool(
            env_cfg.get("fingerTiltModulation", True)
        )

        hand_cfg = cfg.get("hand_config", {})
        hand_obs_dict = hand_cfg.get("num_obs_dict", {})
        num_arm_dofs = int(hand_cfg.get("num_arm_dofs", 6))
        fallback_hand_dofs = int(
            hand_cfg.get("numActions", env_cfg.get("numActions", 30))
        ) - num_arm_dofs
        policy_controls_wrist = bool(
            env_cfg.get("policyControlsStructuralWrist", True)
        )
        self.residual_hand_action_dim = int(
            hand_obs_dict.get("handdof", fallback_hand_dofs)
        )
        if (
            str(hand_cfg.get("name", "")) == "new_sr_hand_simple"
            and not policy_controls_wrist
        ):
            self.residual_hand_action_dim -= len(
                getattr(self, "structural_wrist_dof_names", ("WRJ2", "WRJ1"))
            )
        self.residual_action_dim = {
            "wrist": 6,
            "finger": self.residual_hand_action_dim,
            "hybrid": 6 + self.residual_hand_action_dim,
        }[self.residual_mode]
        super().init_configs(cfg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_new_sr_reference_frames()
        self.residual_translation_scale = torch.tensor(
            self.residual_translation_scale_cfg,
            dtype=torch.float32,
            device=self.device,
        )
        self.residual_rotation_scale = torch.deg2rad(
            torch.tensor(
                self.residual_rotation_scale_deg_cfg,
                dtype=torch.float32,
                device=self.device,
            )
        )
        if self.residual_translation_scale.shape != (3,):
            raise ValueError("residualTranslationScale must contain 3 values")
        if self.residual_rotation_scale.shape != (3,):
            raise ValueError("residualRotationScaleDeg must contain 3 values")
        if torch.any(self.residual_translation_scale < 0):
            raise ValueError("residualTranslationScale must be non-negative")
        if torch.any(self.residual_rotation_scale < 0):
            raise ValueError("residualRotationScaleDeg must be non-negative")
        if not (
            0.0 <= self.finger_residual_min_ratio
            <= self.finger_residual_max_ratio
        ):
            raise ValueError(
                "finger residual ratios must satisfy "
                "0 <= min_ratio <= max_ratio"
            )
        if self.finger_residual_alpha < 0:
            raise ValueError("fingerResidualAlpha must be non-negative")

        lift_idx = self.T_ref_start_lifting - 1
        demo_close = torch.abs(
            self.tracking_reference["hand_qpos"][0, lift_idx]
            - self.tracking_reference["hand_qpos"][0, 0]
        )
        joint_range = (
            self.robot_dof_upper_limits[self.active_hand_dof_indices]
            - self.robot_dof_lower_limits[self.active_hand_dof_indices]
        )
        if demo_close.shape[-1] != joint_range.shape[-1]:
            raise RuntimeError(
                "Residual hand reference dimension mismatch: "
                f"hand_qpos={demo_close.shape[-1]}, "
                f"active_hand_dofs={joint_range.shape[-1]}"
            )
        self.finger_residual_scale = torch.clamp(
            self.finger_residual_alpha * demo_close,
            min=self.finger_residual_min_ratio * joint_range,
            max=self.finger_residual_max_ratio * joint_range,
        )

    def get_baseline_observation(self, residual_obs=None):
        """Return the observation shape expected by the frozen baseline."""
        if residual_obs is None:
            residual_obs = self.obs_dict["obs"]
        if getattr(self, "baseline_uses_residual_observation", False):
            return residual_obs
        if "worldtilt" not in self.obs_type.split("+"):
            return residual_obs

        pcl_dim = self.points_per_object * 3 if "objpcl" in self.obs_type else 0
        state_dim = residual_obs.shape[-1] - pcl_dim - 3
        return torch.cat(
            [
                residual_obs[:, :state_dim],
                residual_obs[:, state_dim + 3:],
            ],
            dim=-1,
        )

    def _tilt_modulation_value(self, env_ids):
        values = [
            self.tilt_modulation_value_for_angle(angle.item())
            for angle in self.env_traj_angles[env_ids]
        ]
        return torch.tensor(
            values, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

    def tilt_modulation_value_for_angle(self, angle):
        angle_deg = abs(float(angle))
        if self.tilt_modulation == "none":
            return 1.0
        if self.tilt_modulation == "sin":
            return math.sin(math.radians(angle_deg))
        if self.tilt_modulation == "linear":
            denominator = max(self.tilt_modulation_max_angle, 1e-6)
            return min(angle_deg / denominator, 1.0)

        denominator = math.sin(
            math.radians(max(self.tilt_modulation_max_angle, 1e-6))
        )
        if abs(denominator) < 1e-8:
            return 0.0 if angle_deg < 1e-8 else 1.0
        return min(
            math.sin(math.radians(angle_deg)) / denominator,
            1.0,
        )

    def residual_is_active_for_angle(self, angle):
        modulation = self.tilt_modulation_value_for_angle(angle)
        wrist_active = (
            self.residual_mode in ("wrist", "hybrid")
            and (not self.wrist_tilt_modulation or modulation > 1e-8)
        )
        finger_active = (
            self.residual_mode in ("finger", "hybrid")
            and (not self.finger_tilt_modulation or modulation > 1e-8)
        )
        return wrist_active or finger_active

    def _split_residual_action(self, residual_actions):
        count = residual_actions.shape[0]
        wrist = torch.zeros((count, 6), device=self.device)
        finger = torch.zeros(
            (count, self.num_active_hand_dofs), device=self.device
        )
        if self.residual_mode == "wrist":
            wrist = residual_actions
        elif self.residual_mode == "finger":
            finger = self._expand_policy_hand_actions(residual_actions)
        else:
            wrist = residual_actions[:, :6]
            finger = self._expand_policy_hand_actions(residual_actions[:, 6:])
        return wrist, finger

    def _decode_wrist_residual(self, env_ids, raw_wrist):
        modulation = (
            self._tilt_modulation_value(env_ids)
            if self.wrist_tilt_modulation
            else 1.0
        )
        translation_table = (
            raw_wrist[:, :3]
            * self.residual_translation_scale
            * modulation
        )
        rotation_vector_table = (
            raw_wrist[:, 3:]
            * self.residual_rotation_scale
            * modulation
        )

        table_quat = self.env_traj_quat[env_ids]
        translation_world = quat_apply(table_quat, translation_table)

        angle = torch.linalg.vector_norm(
            rotation_vector_table, dim=-1
        )
        axis_table = rotation_vector_table / angle.unsqueeze(-1).clamp_min(
            1e-8
        )
        delta_table = quat_from_angle_axis(angle, axis_table)
        delta_world = quat_mul(
            quat_mul(table_quat, delta_table),
            quat_conjugate(table_quat),
        )
        return translation_world, delta_world

    def _decode_finger_residual(self, env_ids, raw_finger):
        modulation = (
            self._tilt_modulation_value(env_ids)
            if self.finger_tilt_modulation
            else 1.0
        )
        return raw_finger * self.finger_residual_scale * modulation

    def generate_residual_reaching_plan_idx(
        self, env_ids, baseline_actions, residual_actions
    ):
        """Apply frozen baseline hand action plus bounded residual edits."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        baseline_actions = baseline_actions[env_ids].to(self.device)
        residual_actions = residual_actions[env_ids].to(self.device)
        baseline_actions = self._expand_policy_hand_actions(baseline_actions)
        if baseline_actions.shape[-1] != self.num_active_hand_dofs:
            raise ValueError(
                "Baseline policy must output "
                f"{self.num_active_hand_dofs} hand actions"
            )
        if residual_actions.shape[-1] != self.residual_action_dim:
            raise ValueError(
                f"Expected {self.residual_action_dim} residual actions, "
                f"got {residual_actions.shape[-1]}"
            )

        self.current_tracking_reference = deepcopy(self.tracking_reference)
        raw_wrist, raw_finger = self._split_residual_action(
            residual_actions
        )
        if (
            self._is_new_sr_hand_simple()
            and self.lock_structural_wrist
            and self.structural_wrist_reference_indices
        ):
            ref_indices = torch.tensor(
                self.structural_wrist_reference_indices,
                dtype=torch.long,
                device=self.device,
            )
            baseline_actions = baseline_actions.clone()
            raw_finger = raw_finger.clone()
            baseline_actions[:, ref_indices] = 0.0
            raw_finger[:, ref_indices] = 0.0
        wrist_translation, wrist_delta_quat = (
            self._decode_wrist_residual(env_ids, raw_wrist)
        )
        finger_delta = self._decode_finger_residual(env_ids, raw_finger)

        lift_idx = self.T_ref_start_lifting - 1
        demo_grasp = self.tracking_reference["hand_qpos"][
            env_ids, lift_idx
        ]
        base_grasp = demo_grasp + (
            baseline_actions * self.randomize_grasp_pose_range
        )
        final_grasp = tensor_clamp(
            base_grasp + finger_delta,
            self.robot_dof_lower_limits[self.active_hand_dof_indices],
            self.robot_dof_upper_limits[self.active_hand_dof_indices],
        )

        hand_ref = self.current_tracking_reference["hand_qpos"][env_ids]
        initial_pose = hand_ref[:, 0]
        denominator = hand_ref[:, lift_idx] - initial_pose
        fraction = (final_grasp - initial_pose) / (denominator + 1e-6)
        if lift_idx > 0:
            self.current_tracking_reference["hand_qpos"][
                env_ids, :lift_idx
            ] = (
                initial_pose.unsqueeze(1)
                + (hand_ref[:, :lift_idx] - initial_pose.unsqueeze(1))
                * fraction.unsqueeze(1)
            )
        self.current_tracking_reference["hand_qpos"][
            env_ids, lift_idx:
        ] = final_grasp.unsqueeze(1)
        self.current_tracking_reference["hand_qpos"][env_ids] = tensor_clamp(
            self.current_tracking_reference["hand_qpos"][env_ids],
            self.robot_dof_lower_limits[self.active_hand_dof_indices],
            self.robot_dof_upper_limits[self.active_hand_dof_indices],
        )
        if (
            self._is_new_sr_hand_simple()
            and self.lock_structural_wrist
            and self.structural_wrist_reference_indices
        ):
            dof_indices = torch.tensor(
                self.structural_wrist_dof_indices,
                dtype=torch.long,
                device=self.device,
            )
            hand_qpos = self.current_tracking_reference["hand_qpos"][env_ids]
            hand_qpos[:, :, ref_indices] = (
                self.robot_dof_default_pos[dof_indices].view(1, 1, -1)
            )
            self.current_tracking_reference["hand_qpos"][env_ids] = hand_qpos

        reference_position = self.current_tracking_reference[
            "wrist_initobj_pos"
        ][env_ids]
        self.current_tracking_reference["wrist_initobj_pos"][env_ids] = (
            quat_apply(
                wrist_delta_quat.unsqueeze(1).expand(
                    -1, reference_position.shape[1], -1
                ),
                reference_position,
            )
            + wrist_translation.unsqueeze(1)
        )
        reference_quat = self.current_tracking_reference["wrist_quat"][
            env_ids
        ]
        self.current_tracking_reference["wrist_quat"][env_ids] = quat_mul(
            wrist_delta_quat.unsqueeze(1).expand_as(reference_quat),
            reference_quat,
        )

        wrist_pose = self.rigid_body_states.view(-1, 13)[
            self.eef_idx[env_ids], 0:7
        ]
        wrist_pose_target = torch.cat(
            [
                self.current_tracking_reference["wrist_initobj_pos"][
                    env_ids, 0
                ]
                + self.object_init_states[env_ids, 0:3],
                self.current_tracking_reference["wrist_quat"][env_ids, 0],
            ],
            dim=-1,
        )
        reaching_plan, reaching_timesteps = batch_linear_interpolate_poses(
            wrist_pose,
            wrist_pose_target,
            max_trans_step=0.04 * self.cfg["env"]["interpolationStepScale"],
            max_rot_step=0.1 * self.cfg["env"]["interpolationStepScale"],
        )
        reaching_plan = reaching_plan[
            :, 1:min(self.max_episode_length, reaching_plan.shape[1])
        ]
        reaching_timesteps -= 1
        self.reaching_plan_ee[env_ids].zero_()
        self.reaching_plan_ee[
            env_ids, :reaching_plan.shape[1]
        ] = reaching_plan
        self.reaching_plan_timesteps[env_ids] = reaching_timesteps

        self.last_baseline_actions = baseline_actions.detach()
        self.last_residual_actions = residual_actions.detach()
        self.last_wrist_translation = wrist_translation.detach()
        self.last_finger_delta = finger_delta.detach()
