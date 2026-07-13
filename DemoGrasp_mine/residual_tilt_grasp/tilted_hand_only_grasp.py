"""Hand-only RL environment with per-env object-centered trajectory rotation."""

import math
import os

import numpy as np
import torch

from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    torch_rand_float,
)

from grasp_base.hand_only_grasp import HandOnlyGrasp


def _quat_to_base_angles_xyz(q):
    """Return serial X/Y/Z base-joint angles for sr_shadow_hand."""
    x, y, z, w = q.unbind(-1)
    angle_x = torch.atan2(
        2 * (w * x - y * z),
        1 - 2 * (x * x + y * y),
    )
    angle_y = torch.asin((2 * (w * y + x * z)).clamp(-1, 1))
    angle_z = torch.atan2(
        2 * (w * z - x * y),
        1 - 2 * (y * y + z * z),
    )
    return torch.stack([angle_x, angle_y, angle_z], dim=-1)


def _quat_to_base_angles_zyx(q):
    """Return serial Z/Y/X base-joint angles for shadow_simple-style hands."""
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack([yaw, pitch, roll], dim=-1)


class TiltedHandOnlyGrasp(HandOnlyGrasp):
    """Sample one trajectory rotation per environment at reset.

    Earlier experiments rotated the table and gravity, which forced a single
    shared angle for the whole simulation. The sr hand can instead replay a
    transformed hand trajectory around the object while the physical table and
    gravity stay horizontal. That allows multiple angles in the same vectorized
    rollout.
    """

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.world_tilt_angles = [
            float(angle) for angle in env_cfg.get("worldTiltAngles", [0.0])
        ]
        if not self.world_tilt_angles:
            raise ValueError("task.env.worldTiltAngles must not be empty")

        axis = torch.tensor(
            env_cfg.get("worldTiltAxis", [0.0, 1.0, 0.0]), dtype=torch.float32
        )
        axis_norm = torch.linalg.vector_norm(axis)
        if axis_norm < 1e-8:
            raise ValueError("task.env.worldTiltAxis must be non-zero")
        self.world_tilt_axis_cpu = axis / axis_norm

        self.world_tilt_sampling = env_cfg.get("worldTiltSampling", "random")
        if self.world_tilt_sampling not in ("random", "cycle"):
            raise ValueError(
                "task.env.worldTiltSampling must be 'random' or 'cycle'"
            )
        self.world_tilt_log_interval = int(
            env_cfg.get("worldTiltLogInterval", 100)
        )
        self.world_tilt_pointcloud_frame = env_cfg.get(
            "worldTiltPointCloudFrame", "table"
        )
        if self.world_tilt_pointcloud_frame not in (
            "world", "table", "object"
        ):
            raise ValueError(
                "task.env.worldTiltPointCloudFrame must be "
                "'world', 'table', or 'object'"
            )
        self._tilt_cycle_index = 0
        self._tilt_reset_count = 0
        self.current_world_tilt = 0.0
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
        self.latest_angle_metrics = {}
        self._sr_reference_prepared = False
        self._new_sr_reference_prepared = False
        self.sr_palm_offset_from_forearm = None
        self.base_tracking_reference = None
        self.lock_structural_wrist = bool(
            env_cfg.get("lockStructuralWrist", False)
        )
        self.policy_controls_structural_wrist = bool(
            env_cfg.get("policyControlsStructuralWrist", True)
        )
        self.policy_action_dim = None
        self.policy_hand_action_indices = None
        self.structural_wrist_dof_names = ("WRJ2", "WRJ1")
        self.structural_wrist_dof_indices = []
        self.structural_wrist_action_indices = []
        self.structural_wrist_reference_indices = []
        self._structural_wrist_policy_space_prepared = False

        super().init_configs(cfg)
        self._prepare_new_sr_reference_frames()

    def _is_sr_shadow_hand(self):
        return str(self.hand_specific_cfg.get("name", "")) == "sr_shadow_hand"

    def _is_new_sr_hand_simple(self):
        return (
            str(self.hand_specific_cfg.get("name", ""))
            == "new_sr_hand_simple"
        )

    def _prepare_structural_wrist_policy_space(self):
        if (
            self._structural_wrist_policy_space_prepared
            or not self._is_new_sr_hand_simple()
            or not hasattr(self, "robot_dof_names")
            or not hasattr(self, "active_hand_dof_indices")
        ):
            return

        dof_name_to_index = {
            name: index for index, name in enumerate(self.robot_dof_names)
        }
        active_hand_indices = [
            int(index) for index in self.active_hand_dof_indices
        ]
        self.structural_wrist_dof_indices = []
        self.structural_wrist_action_indices = []
        self.structural_wrist_reference_indices = []
        for name in self.structural_wrist_dof_names:
            dof_index = dof_name_to_index.get(name)
            if dof_index is None:
                continue
            self.structural_wrist_dof_indices.append(dof_index)
            if dof_index in active_hand_indices:
                hand_action_offset = active_hand_indices.index(dof_index)
                self.structural_wrist_reference_indices.append(
                    hand_action_offset
                )
                self.structural_wrist_action_indices.append(
                    self.hand_dof_start_idx + hand_action_offset
                )

        if self.policy_controls_structural_wrist:
            self.policy_hand_action_indices = list(
                range(self.num_active_hand_dofs)
            )
        else:
            excluded = set(self.structural_wrist_reference_indices)
            self.policy_hand_action_indices = [
                index for index in range(self.num_active_hand_dofs)
                if index not in excluded
            ]
        self.policy_action_dim = len(self.policy_hand_action_indices)
        self._structural_wrist_policy_space_prepared = True
        if not self.policy_controls_structural_wrist:
            print(
                "new_sr_hand_simple policy action space excludes "
                f"{self.structural_wrist_dof_names}: "
                f"{self.policy_action_dim} learned hand actions -> "
                f"{self.num_active_hand_dofs} simulated active hand DOFs",
                flush=True,
            )

    def _prepare_new_sr_reference_frames(self):
        """Adapt the 18-DOF shadow_simple reference to WRJ2/WRJ1 + 18 DOFs."""
        if self._new_sr_reference_prepared or not self._is_new_sr_hand_simple():
            return
        if not hasattr(self, "tracking_reference"):
            return
        hand_qpos = self.tracking_reference["hand_qpos"]
        expected = self.num_active_hand_dofs
        if hand_qpos.shape[-1] == expected:
            self._new_sr_reference_prepared = True
        elif hand_qpos.shape[-1] + 2 == expected:
            zeros = torch.zeros(
                (*hand_qpos.shape[:-1], 2),
                dtype=hand_qpos.dtype,
                device=hand_qpos.device,
            )
            self.tracking_reference["hand_qpos"] = torch.cat(
                [zeros, hand_qpos], dim=-1
            )
            self._new_sr_reference_prepared = True
            print(
                "Prepared new_sr_hand_simple reference: "
                "prepended zero WRJ2/WRJ1 targets to shadow_simple hand_qpos",
                flush=True,
            )
        else:
            raise RuntimeError(
                "new_sr_hand_simple reference hand_qpos dimension mismatch: "
                f"reference={hand_qpos.shape[-1]}, active_hand={expected}"
            )

        self.base_tracking_reference = {
            key: value.clone() for key, value in self.tracking_reference.items()
        }

        self._prepare_structural_wrist_policy_space()

    def _expand_policy_hand_actions(self, actions):
        if not (
            self._is_new_sr_hand_simple()
            and not self.policy_controls_structural_wrist
        ):
            return actions
        if actions is None:
            return None
        if actions.shape[-1] == self.num_active_hand_dofs:
            return actions
        if actions.shape[-1] != self.policy_action_dim:
            raise ValueError(
                f"Expected {self.policy_action_dim} policy actions without "
                f"structural wrist, or {self.num_active_hand_dofs} full hand "
                f"actions, got {actions.shape[-1]}"
            )
        expanded = torch.zeros(
            (*actions.shape[:-1], self.num_active_hand_dofs),
            dtype=actions.dtype,
            device=actions.device,
        )
        indices = torch.tensor(
            self.policy_hand_action_indices,
            dtype=torch.long,
            device=actions.device,
        )
        expanded.index_copy_(-1, indices, actions)
        return expanded

    def _sr_wrist_quat_offset(self):
        return torch.tensor(
            [0.0, 0.0, 0.7071068, 0.7071068],
            dtype=torch.float32,
            device=self.device,
        )

    def _sr_palm_offset(self):
        if self.sr_palm_offset_from_forearm is None:
            self.sr_palm_offset_from_forearm = torch.tensor(
                [0.0, -0.01, 0.247],
                dtype=torch.float32,
                device=self.device,
            )
        return self.sr_palm_offset_from_forearm

    def _prepare_sr_reference_frames(self):
        """Convert sr reference targets from palm/wrist frame to forearm/eef.

        The base task tracks `eef_link`. For sr_shadow_hand that link is
        `forearm`, while the reference trajectory is authored in the same
        palm/wrist convention used by shadow_simple. Convert once so reset,
        reaching, and tracking all use the forearm target consistently.
        """
        if self._sr_reference_prepared or not self._is_sr_shadow_hand():
            return
        if not hasattr(self, "tracking_reference"):
            return

        wrist_quat = self.tracking_reference["wrist_quat"]
        quat_offset = self._sr_wrist_quat_offset().view(1, 1, 4)
        wrist_quat = quat_mul(
            wrist_quat,
            quat_offset.expand_as(wrist_quat),
        )
        offset = quat_apply(
            wrist_quat,
            self._sr_palm_offset().view(1, 1, 3).expand(
                wrist_quat.shape[0], wrist_quat.shape[1], 3
            ),
        )
        self.tracking_reference["wrist_quat"] = wrist_quat
        self.tracking_reference["wrist_initobj_pos"] = (
            self.tracking_reference["wrist_initobj_pos"] - offset
        )
        self.sr_reference_frame0_base_angles = _quat_to_base_angles_xyz(
            wrist_quat[:, 0]
        )
        self._sr_reference_prepared = True
        self.base_tracking_reference = {
            key: value.clone() for key, value in self.tracking_reference.items()
        }
        print(
            "Prepared sr_shadow_hand reference in forearm/eef frame: "
            "applied wrist_quat_offset and palm_offset_from_forearm",
            flush=True,
        )

    def _create_ground_plane(self):
        # A horizontal infinite plane would intersect a tilted table.
        return None

    def _prepare_table_asset(self):
        """Load one rotatable actor with the original table+mat geometry.

        The actor origin is the support-surface pivot, so every angle uses the
        exact same collision geometry and rotates around the same point.
        """
        self.table_thickness = 0.3
        default_surface_center = [
            0.51,
            -0.075,
            float(self.cfg["env"]["tableHeightRange"][0]),
        ]
        table_surface_center = self.cfg["env"].get(
            "tableSurfaceCenter", default_surface_center
        )
        if len(table_surface_center) != 3:
            raise ValueError("tableSurfaceCenter must contain xyz")
        self.table_heights = torch.full(
            (self.num_envs,),
            float(table_surface_center[2]),
            dtype=torch.float,
            device=self.device,
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        table_asset = self.gym.load_asset(
            self.sim,
            os.path.join(os.path.dirname(__file__), "assets"),
            "tilted_table.urdf",
            asset_options,
        )
        if table_asset is None:
            raise RuntimeError("Failed to load residual_tilt_grasp tilted table asset")
        # tasks/grasp.py reserves one table shape in each aggregate. This
        # compound actor has a second shape for the mat.
        self.num_object_shapes += 1

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(
            float(table_surface_center[0]),
            float(table_surface_center[1]),
            float(table_surface_center[2]),
        )
        table_start_poses = [table_start_pose] * self.num_envs
        return (
            table_asset,
            table_start_poses,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def _prepare_robot_asset(self, asset_root, asset_file):
        robot_asset, robot_dof_props, robot_start_pose = (
            super()._prepare_robot_asset(asset_root, asset_file)
        )
        self._prepare_structural_wrist_policy_space()
        if self._is_new_sr_hand_simple() and self.lock_structural_wrist:
            for dof_index, name in enumerate(self.robot_dof_names):
                if name not in self.structural_wrist_dof_names:
                    continue
                robot_dof_props["stiffness"][dof_index] = (
                    16000 * self.cfg["env"]["pdParamScale"]
                )
                robot_dof_props["damping"][dof_index] = (
                    600 * self.cfg["env"]["pdParamScale"]
                )
                print(
                    f"Locked structural wrist DOF {name}: "
                    f"stiffness={robot_dof_props['stiffness'][dof_index]:.2f}, "
                    f"damping={robot_dof_props['damping'][dof_index]:.2f}",
                    flush=True,
                )
        return robot_asset, robot_dof_props, robot_start_pose

    def _ensure_base_tracking_reference(self):
        self._prepare_sr_reference_frames()
        self._prepare_new_sr_reference_frames()
        if self.base_tracking_reference is None:
            self.base_tracking_reference = {
                key: value.clone()
                for key, value in self.tracking_reference.items()
            }

    def _sample_trajectory_angle_ids(self, env_ids):
        count = len(env_ids)
        if len(self.world_tilt_angles) == 1:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        if self.world_tilt_sampling == "cycle":
            ids = (
                torch.arange(count, device=self.device)
                + self._tilt_cycle_index
            ) % len(self.world_tilt_angles)
            self._tilt_cycle_index = (
                self._tilt_cycle_index + count
            ) % len(self.world_tilt_angles)
            return ids.long()
        return torch.randint(
            0, len(self.world_tilt_angles), (count,), device=self.device
        )

    def _axis_angle_quat(self, angle_deg):
        angle = torch.tensor(
            math.radians(angle_deg), dtype=torch.float32, device=self.device
        )
        axis = self.world_tilt_axis_cpu.to(self.device)
        return quat_from_angle_axis(angle.view(1), axis.view(1, 3))[0]

    def _set_env_trajectory_rotations(self, env_ids, angle_ids):
        self._ensure_base_tracking_reference()
        angle_values = torch.tensor(
            self.world_tilt_angles, dtype=torch.float32, device=self.device
        )
        angles = angle_values[angle_ids]
        axis = self.world_tilt_axis_cpu.to(self.device)
        angle_rad = torch.deg2rad(angles)
        axes = axis.view(1, 3).expand(len(env_ids), -1)
        quats = quat_from_angle_axis(angle_rad, axes)

        self.env_angle_ids[env_ids] = angle_ids
        self.env_traj_angles[env_ids] = angles
        self.env_traj_quat[env_ids] = quats
        self.env_traj_quat_inv[env_ids] = quat_conjugate(quats)
        self.env_traj_normal[env_ids] = quat_apply(
            quats,
            torch.tensor(
                [[0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            ).expand(len(env_ids), -1),
        )
        self.current_world_tilt = float(angles.float().mean().item())

        for key, value in self.base_tracking_reference.items():
            self.tracking_reference[key][env_ids] = value[env_ids]
        quat_seq = quats.unsqueeze(1).expand(-1, self.T_ref, -1)
        self.tracking_reference["wrist_initobj_pos"][env_ids] = quat_apply(
            quat_seq,
            self.base_tracking_reference["wrist_initobj_pos"][env_ids],
        )
        self.tracking_reference["wrist_quat"][env_ids] = quat_mul(
            quat_seq,
            self.base_tracking_reference["wrist_quat"][env_ids],
        )

    def _sample_local_object_poses(self, env_ids):
        count = len(env_ids)
        reset_range = self.reset_position_range
        samples = reset_range[:, 0] + (
            reset_range[:, 1] - reset_range[:, 0]
        ) * torch.rand(count, 3, device=self.device)

        canonical_top = self.canonical_table_top[env_ids]
        local_offset = samples.clone()
        local_offset[:, 0] -= canonical_top[:, 0]
        local_offset[:, 1] -= canonical_top[:, 1]
        # resetPositionRange Z is defined as height above the tabletop.

        world_quat = self.identity_quat.view(1, 4).expand(count, -1)
        world_pos = canonical_top + local_offset

        rand_axis = np.random.randn(count, 3)
        if self.reset_random_rot == "z":
            rand_axis[:] = np.array([0.0, 0.0, 1.0])
        rand_axis /= np.linalg.norm(rand_axis, axis=1, keepdims=True)
        rand_axis = torch.tensor(
            rand_axis, dtype=torch.float32, device=self.device
        )
        rand_angle = torch_rand_float(
            -math.pi, math.pi, (count, 1), device=self.device
        ).squeeze(-1)
        if self.reset_random_rot == "fixed":
            rand_angle.zero_()
        local_quat = quat_from_angle_axis(rand_angle, rand_axis)
        object_quat = quat_mul(world_quat, local_quat)

        object_pose = torch.cat([world_pos, object_quat], dim=-1)
        return object_pose

    def _prepare_tilted_reset(self, env_ids):
        count = len(env_ids)
        world_quat = self.identity_quat.view(1, 4).expand(count, -1)

        table_pivot = self.canonical_table_top[env_ids]
        self.root_state_tensor[self.table_indices[env_ids], 0:3] = table_pivot
        self.root_state_tensor[self.table_indices[env_ids], 3:7] = world_quat
        self.root_state_tensor[self.table_indices[env_ids], 7:13] = 0.0

        # Grasp.reset_idx assumes the actor origin is half a table thickness
        # below the surface. Offset its synthetic height so it writes our
        # surface-pivot Z back unchanged.
        effective_height = table_pivot[:, 2] + self.table_thickness * 0.5
        self.table_height_range = torch.stack(
            [effective_height.min(), effective_height.max()]
        )

        return self._sample_local_object_poses(env_ids)

    def _prepare_reset_robot_pose(self, env_ids, object_init_pose):
        if not (self._is_sr_shadow_hand() or self._is_new_sr_hand_simple()):
            return None
        self._ensure_base_tracking_reference()
        if self._is_sr_shadow_hand() and not self._sr_reference_prepared:
            return None
        if self._is_new_sr_hand_simple() and not self._new_sr_reference_prepared:
            return None

        robot_pose = self.robot_dof_default_pos.unsqueeze(0).repeat(
            len(env_ids), 1
        )
        base_rot_indices = self.arm_dof_indices[3:6]
        base_xyz_indices = self.arm_dof_indices[0:3]

        frame0_quat = self.tracking_reference["wrist_quat"][env_ids, 0]
        if self._is_sr_shadow_hand():
            frame0_angles = _quat_to_base_angles_xyz(frame0_quat)
        else:
            frame0_angles = _quat_to_base_angles_zyx(frame0_quat)
        robot_pose[:, base_rot_indices] = frame0_angles

        if self._is_sr_shadow_hand():
            offset = quat_apply(
                frame0_quat,
                self._sr_palm_offset().view(1, 3).expand(len(env_ids), 3),
            )
        else:
            offset = torch.zeros(
                (len(env_ids), 3),
                dtype=torch.float32,
                device=self.device,
            )
        object_pos = object_init_pose[:, 0:3]
        raw_default_palm = self.robot_dof_default_pos[
            base_xyz_indices
        ].view(1, 3)
        start_rel = raw_default_palm - object_pos
        rotated_start_rel = quat_apply(
            self.env_traj_quat[env_ids],
            start_rel,
        )
        robot_pose[:, base_xyz_indices] = (
            object_pos + rotated_start_rel - offset
        )
        return robot_pose

    def pre_physics_step(self, actions):
        if (
            self._is_new_sr_hand_simple()
            and self.lock_structural_wrist
            and self.structural_wrist_action_indices
        ):
            actions = actions.clone()
            for dof_index, action_index in zip(
                self.structural_wrist_dof_indices,
                self.structural_wrist_action_indices,
            ):
                if self.use_relative_control:
                    actions[:, action_index] = 0.0
                else:
                    lower = self.robot_dof_lower_limits[dof_index]
                    upper = self.robot_dof_upper_limits[dof_index]
                    default = self.robot_dof_default_pos[dof_index]
                    actions[:, action_index] = (
                        2.0 * (default - lower) / (upper - lower) - 1.0
                    )
        super().pre_physics_step(actions)
        if (
            self._is_new_sr_hand_simple()
            and self.lock_structural_wrist
            and self.structural_wrist_dof_indices
        ):
            indices = torch.tensor(
                self.structural_wrist_dof_indices,
                dtype=torch.long,
                device=self.device,
            )
            self.cur_targets[:, indices] = self.robot_dof_default_pos[indices]

    def generate_reaching_plan_idx(self, env_ids, actions=None):
        actions = self._expand_policy_hand_actions(actions)
        result = super().generate_reaching_plan_idx(env_ids, actions=actions)
        if (
            self._is_new_sr_hand_simple()
            and self.lock_structural_wrist
            and self.structural_wrist_reference_indices
            and hasattr(self, "current_tracking_reference")
        ):
            ref_indices = torch.tensor(
                self.structural_wrist_reference_indices,
                dtype=torch.long,
                device=self.device,
            )
            dof_indices = torch.tensor(
                self.structural_wrist_dof_indices,
                dtype=torch.long,
                device=self.device,
            )
            self.current_tracking_reference["hand_qpos"][:, :, ref_indices] = (
                self.robot_dof_default_pos[dof_indices].view(1, 1, -1)
            )
        return result

    def reset_idx(self, env_ids, object_init_pose=None, **kwargs):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._ensure_base_tracking_reference()
        if hasattr(self, "successes"):
            for angle_id in torch.unique(self.env_angle_ids[env_ids]):
                mask = self.env_angle_ids[env_ids] == angle_id
                scoped_envs = env_ids[mask]
                angle = self.world_tilt_angles[int(angle_id.item())]
                self._angle_success_sum[angle] += (
                    self.successes[scoped_envs].float().mean().item()
                )
                self._angle_success_count[angle] += 1
                self._angle_hit_sum[angle] += (
                    self.has_hit_table[scoped_envs].float().mean().item()
                )
                self._angle_clearance_sum[angle] += (
                    self.episode_min_hand_table_clearance[
                        scoped_envs
                    ].mean().item()
                )

        angle_ids = self._sample_trajectory_angle_ids(env_ids)
        self._set_env_trajectory_rotations(env_ids, angle_ids)

        object_init_pose = self._prepare_tilted_reset(env_ids)
        robot_init_pose = self._prepare_reset_robot_pose(
            env_ids, object_init_pose
        )
        result = super().reset_idx(
            env_ids,
            object_init_pose=object_init_pose,
            robot_init_pose=robot_init_pose,
            **kwargs,
        )
        # The parent reset used a synthetic height only to place the compound
        # actor root at the surface pivot. Keep the public height semantic equal
        # to the actual support surface afterward.
        self.table_heights[env_ids] = self.canonical_table_top[env_ids, 2]
        self.episode_min_hand_table_clearance[env_ids] = torch.inf

        self._tilt_reset_count += 1
        if (
            self.world_tilt_log_interval > 0
            and (
                self._tilt_reset_count == 1
                or self._tilt_reset_count % self.world_tilt_log_interval == 0
            )
        ):
            print(
                f"World tilt reset {self._tilt_reset_count}: "
                f"trajectory angles={self.env_traj_angles[env_ids].tolist()[:8]}"
            )
            summaries = []
            for angle in self.world_tilt_angles:
                count = self._angle_success_count[angle]
                if count:
                    mean_success = self._angle_success_sum[angle] / count
                    mean_hit = self._angle_hit_sum[angle] / count
                    mean_clearance = (
                        self._angle_clearance_sum[angle] / count
                    )
                    summaries.append(
                        f"{angle:g}deg success={mean_success:.3f}, "
                        f"hit={mean_hit:.3f}, "
                        f"min_clearance={mean_clearance:.4f}m ({count})"
                    )
            if summaries:
                print("Per-angle rollout success: " + ", ".join(summaries))
        return result

    def _world_points_to_table_frame(self, points):
        return points

    def compute_reward(self):
        object_pos = self._world_points_to_table_frame(self.object_pos)
        palm_pos = self._world_points_to_table_frame(self.palm_center_pos)
        fingertip_pos = self._world_points_to_table_frame(self.fingertip_pos)

        object_init_states = self.object_init_states.clone()
        object_init_states[:, 0:3] = self._world_points_to_table_frame(
            object_init_states[:, 0:3]
        )
        canonical_table_heights = self.canonical_table_top[:, 2]
        hand_points = torch.cat(
            [palm_pos.unsqueeze(1), fingertip_pos], dim=1
        )
        min_hand_table_clearance = (
            hand_points[..., 2] - canonical_table_heights.unsqueeze(1)
        ).min(dim=1).values
        self.episode_min_hand_table_clearance = torch.minimum(
            self.episode_min_hand_table_clearance,
            min_hand_table_clearance,
        )

        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.has_hit_table[:],
            reward_info,
        ) = self.reward_function(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            successes=self.successes,
            current_successes=self.current_successes,
            has_hit_table=self.has_hit_table,
            max_episode_length=self.max_episode_length,
            table_heights=canonical_table_heights,
            object_pos=object_pos,
            palm_pos=palm_pos,
            fingertip_pos=fingertip_pos,
            num_fingers=self.num_fingers,
            object_init_states=object_init_states,
            end_effector_pose=self.rigid_body_states.view(-1, 13)[
                self.eef_idx, 0:7
            ],
            hand_qpos=self.robot_dof_pos[:, self.active_hand_dof_indices],
        )

        reward_info["world_tilt_deg"] = torch.full(
            (self.num_envs,),
            0.0,
            device=self.device,
        )
        reward_info["traj_rot_deg"] = self.env_traj_angles.clone()
        reward_info["min_hand_table_clearance"] = min_hand_table_clearance
        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["has_hit_table"] = self.has_hit_table

        if torch.all(self.progress_buf == self.max_episode_length - 1):
            self.latest_angle_metrics = {}
            for angle_id in torch.unique(self.env_angle_ids):
                mask = self.env_angle_ids == angle_id
                angle = self.world_tilt_angles[int(angle_id.item())]
                self.latest_angle_metrics[angle] = {
                    "success": self.successes[mask].float().mean().item(),
                    "hit": self.has_hit_table[mask].float().mean().item(),
                    "min_clearance": (
                        self.episode_min_hand_table_clearance[
                            mask
                        ].mean().item()
                    ),
                }

    def compute_required_observations(self, obs_buf, obs_type, num_obs):
        if "worldtilt" not in obs_type:
            return super().compute_required_observations(
                obs_buf, obs_type, num_obs
            )

        base_obs_type = "+".join(
            part for part in obs_type.split("+") if part != "worldtilt"
        )
        base_num_obs = num_obs - 3

        # ActorCritic assumes objpcl occupies the final points_per_object*3
        # values. Keep that invariant by inserting worldtilt immediately before
        # the point cloud rather than appending it after the point cloud.
        if "objpcl" in base_obs_type:
            pcl_dim = self.points_per_object * 3
            state_dim = base_num_obs - pcl_dim
            temporary = torch.empty(
                (self.num_envs, base_num_obs),
                dtype=obs_buf.dtype,
                device=obs_buf.device,
            )
            super().compute_required_observations(
                temporary, base_obs_type, base_num_obs
            )
            if self.world_tilt_pointcloud_frame == "object":
                temporary[:, state_dim:] = self.obj_pcl_buf.reshape(
                    self.num_envs, -1
                )
            elif self.world_tilt_pointcloud_frame == "table":
                world_pcl = temporary[:, state_dim:].reshape(
                    self.num_envs, self.points_per_object, 3
                )
                temporary[:, state_dim:] = (
                    self._world_points_to_table_frame(world_pcl).reshape(
                        self.num_envs, -1
                    )
                )
            obs_buf[:, :state_dim] = temporary[:, :state_dim]
            obs_buf[:, state_dim:state_dim + 3] = self.env_traj_normal
            obs_buf[:, state_dim + 3:] = temporary[:, state_dim:]
        else:
            super().compute_required_observations(
                obs_buf[:, :-3], base_obs_type, base_num_obs
            )
            obs_buf[:, -3:] = self.env_traj_normal

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        canonical_center = self.table_start_pos.clone()
        self.canonical_table_top = canonical_center.clone()
        # table_heights denotes the actual support surface. In the realistic
        # scene this is the top of the mat, not the top of the box below it.
        self.canonical_table_top[:, 2] = self.table_heights
        self.episode_min_hand_table_clearance = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self.identity_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.table_normal = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.world_tilt_quat = self.identity_quat.clone()
        self.world_tilt_quat_inv = self.identity_quat.clone()
        self.env_angle_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_traj_angles = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.env_traj_quat = self.identity_quat.view(1, 4).repeat(
            self.num_envs, 1
        )
        self.env_traj_quat_inv = self.identity_quat.view(1, 4).repeat(
            self.num_envs, 1
        )
        self.env_traj_normal = self.table_normal.view(1, 3).repeat(
            self.num_envs, 1
        )
