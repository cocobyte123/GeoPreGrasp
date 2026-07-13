"""Hand-only Shadow Hand training environment.

The six virtual wrist joints follow the fixed demonstration trajectory. The
policy only adjusts the 18 active Shadow Hand joints at the grasp pose.
"""

from copy import deepcopy

import torch

from isaacgymenvs.utils.torch_jit_utils import tensor_clamp, torch_rand_float

from tasks.grasp import Grasp
from tasks.utils import batch_linear_interpolate_poses


class HandOnlyGrasp(Grasp):
    """Reuse Grasp physics/reward while learning only the hand configuration."""

    def _prepare_robot_asset(self, asset_root, asset_file):
        result = super()._prepare_robot_asset(asset_root, asset_file)
        # shadow_simple has six virtual wrist joints. Keep this local invariant
        # even if the original Grasp implementation uses a pose-action prefix.
        self.hand_dof_start_idx = self.num_arm_dofs
        self._prepare_coarse_grasp_config()
        return result

    def init_configs(self, cfg):
        super().init_configs(cfg)
        # Accept any hand that provides virtual base joints (arm_dof_names).
        # The original check hardcoded shadow_simple; now we verify the
        # required fields exist so sr_shadow_hand and others work directly.
        required_fields = ["arm_dof_names", "palm_link", "fingertips_link"]
        for field in required_fields:
            if field not in self.hand_specific_cfg:
                raise ValueError(
                    f"HandOnlyGrasp requires hand config field '{field}'"
                )
        if self.arm_controller != "qpos":
            raise ValueError(
                "HandOnlyGrasp requires task.env.armController=qpos so the "
                "virtual wrist joints can follow the reference trajectory."
            )
        self.coarse_grasp_pose_cfg = self.cfg["env"].get(
            "coarseGraspHandDofPos", None
        )
        self.grasp_delta_scale_cfg = self.cfg["env"].get(
            "graspDeltaScale",
            self.cfg["env"]["randomizeGraspPoseRange"],
        )

    def _prepare_coarse_grasp_config(self):
        """Validate the optional coarse grasp pose after the asset is loaded."""
        self.coarse_grasp_pose = None
        if self.coarse_grasp_pose_cfg is not None:
            pose = torch.tensor(
                self.coarse_grasp_pose_cfg,
                dtype=torch.float32,
                device=self.device,
            )
            if pose.numel() != self.num_active_hand_dofs:
                raise ValueError(
                    "task.env.coarseGraspHandDofPos must contain "
                    f"{self.num_active_hand_dofs} active hand values, "
                    f"got {pose.numel()}"
                )
            self.coarse_grasp_pose = tensor_clamp(
                pose,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
            print(
                "Using configured coarse grasp pose: "
                f"{self.coarse_grasp_pose.tolist()}"
            )

        scale = torch.tensor(
            self.grasp_delta_scale_cfg,
            dtype=torch.float32,
            device=self.device,
        )
        if scale.numel() == 1:
            scale = scale.repeat(self.num_active_hand_dofs)
        elif scale.numel() != self.num_active_hand_dofs:
            raise ValueError(
                "task.env.graspDeltaScale must be a scalar or contain "
                f"{self.num_active_hand_dofs} values, got {scale.numel()}"
            )
        if torch.any(scale < 0):
            raise ValueError("task.env.graspDeltaScale must be non-negative")
        self.grasp_delta_scale = scale
        print(f"Grasp policy delta scale: {self.grasp_delta_scale.tolist()}")

    def generate_reaching_plan_idx(self, env_ids, actions=None):
        """Build a fixed wrist trajectory with a policy-adjusted grasp pose.

        Policy actions are normalized hand-joint offsets in [-1, 1]. They are
        scaled by graspDeltaScale around the configured coarse grasp pose, then
        clamped to the URDF limits.
        """
        self.current_tracking_reference = deepcopy(self.tracking_reference)

        if actions is None:
            if self.randomize_grasp_pose:
                hand_delta = torch_rand_float(
                    -1,
                    1,
                    (len(env_ids), self.num_active_hand_dofs),
                    device=self.device,
                )
            else:
                hand_delta = torch.zeros(
                    (len(env_ids), self.num_active_hand_dofs),
                    device=self.device,
                )
        else:
            if actions.shape[-1] != self.num_active_hand_dofs:
                raise ValueError(
                    f"Expected {self.num_active_hand_dofs} policy actions, "
                    f"got {actions.shape[-1]}"
                )
            hand_delta = actions[env_ids].to(self.device)

        if self.randomize_grasp_pose:
            lift_idx = self.T_ref_start_lifting - 1
            if self.coarse_grasp_pose is None:
                base_grasp_pose = self.tracking_reference["hand_qpos"][
                    env_ids, lift_idx
                ]
            else:
                base_grasp_pose = self.coarse_grasp_pose.unsqueeze(0).expand(
                    len(env_ids), -1
                )
            grasp_pose = tensor_clamp(
                base_grasp_pose + hand_delta * self.grasp_delta_scale,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )

            hand_ref = self.current_tracking_reference["hand_qpos"][env_ids]
            initial_pose = hand_ref[:, 0]
            denominator = hand_ref[:, lift_idx] - initial_pose
            fraction = (grasp_pose - initial_pose) / (denominator + 1e-6)

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
            ] = grasp_pose.unsqueeze(1)
            self.current_tracking_reference["hand_qpos"][env_ids] = tensor_clamp(
                self.current_tracking_reference["hand_qpos"][env_ids],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )

        wrist_pose = self.rigid_body_states.view(-1, 13)[
            self.eef_idx[env_ids], 0:7
        ]
        wrist_pose_target = torch.cat(
            [
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, 0]
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
        self.reaching_plan_ee[env_ids, :reaching_plan.shape[1]] = reaching_plan
        self.reaching_plan_timesteps[env_ids] = reaching_timesteps
