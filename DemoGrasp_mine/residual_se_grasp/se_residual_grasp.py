"""Two-DOF object-centered residual grasp environment.

This environment keeps the residual PPO stack from ``residual_tilt_grasp`` but replaces
the old single-axis trajectory tilt with a small SE(3)-like guide family:

* yaw moves the hand around a bounded object-centered horizontal arc;
* tilt reuses the old single-axis multi-angle training rotation;
* pitch moves the hand on the corresponding spherical shell and rotates the
  wrist with it, so the palm-facing relation is preserved.

The policy observes the sampled guide through sin/cos angle features. Training
can use discrete angle pairs while later inference can write continuous values
through the same geometric parameterization.
"""

import torch

from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

from residual_tilt_grasp.residual_tilted_grasp import ResidualTiltedGrasp
from residual_tilt_grasp.tilted_hand_only_grasp import (
    _quat_to_base_angles_xyz,
    _quat_to_base_angles_zyx,
)


def _normalize(vector, eps=1e-8):
    return vector / torch.linalg.vector_norm(
        vector, dim=-1, keepdim=True
    ).clamp_min(eps)


class SEResidualGrasp(ResidualTiltedGrasp):
    """Residual grasp task with discrete two-angle guide sampling."""

    GUIDE_OBS_DIM = 4

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.se_guide_mode = env_cfg.get("seGuideMode", "tilt_pitch")
        if self.se_guide_mode not in (
            "yaw_pitch", "tilt_pitch", "legacy_tilt"
        ):
            raise ValueError(
                "seGuideMode must be 'yaw_pitch', 'tilt_pitch', "
                "or 'legacy_tilt'"
            )
        yaw_angles = [
            float(angle)
            for angle in env_cfg.get(
                "seYawAngles", [-30.0, -15.0, 0.0, 15.0, 30.0]
            )
        ]
        tilt_angles = [
            float(angle)
            for angle in env_cfg.get(
                "seTiltAngles", [0.0, 15.0, 30.0, 45.0]
            )
        ]
        pitch_angles = [
            float(angle)
            for angle in env_cfg.get(
                "sePitchAngles", [-20.0, -10.0, 0.0, 10.0, 20.0]
            )
        ]
        explicit_pairs = env_cfg.get("seYawPitchPairs", None)
        if explicit_pairs is None:
            if self.se_guide_mode == "tilt_pitch":
                pairs = [
                    (tilt, pitch)
                    for tilt in tilt_angles
                    for pitch in pitch_angles
                ]
            else:
                pairs = [
                    (yaw, pitch) for yaw in yaw_angles for pitch in pitch_angles
                ]
        else:
            pairs = [(float(pair[0]), float(pair[1])) for pair in explicit_pairs]
        if not pairs:
            raise ValueError("seYawPitchPairs must not be empty")

        self.se_angle_pairs_cfg = pairs
        self.se_tilt_axis_cfg = env_cfg.get("seTiltAxis", [1.0, 0.0, 0.0])
        if self.se_guide_mode == "legacy_tilt":
            env_cfg["worldTiltAngles"] = [
                float(angle)
                for angle in env_cfg.get(
                    "seLegacyTiltAngles", [0.0, 15.0, 30.0, 45.0]
                )
            ]
            env_cfg["worldTiltAxis"] = env_cfg.get(
                "seLegacyTiltAxis", [0.0, 1.0, 0.0]
            )
        else:
            env_cfg["worldTiltAngles"] = [
                float(index) for index in range(len(pairs))
            ]
            env_cfg["worldTiltAxis"] = [0.0, 0.0, 1.0]
        env_cfg["worldTiltSampling"] = env_cfg.get(
            "seSampling", env_cfg.get("worldTiltSampling", "random")
        )
        env_cfg["residualTiltModulation"] = env_cfg.get(
            "residualTiltModulation", "none"
        )

        super().init_configs(cfg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.se_angle_pairs = torch.tensor(
            self.se_angle_pairs_cfg, dtype=torch.float32, device=self.device
        )
        self.env_se_yaw = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.env_se_pitch = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.env_se_guide_obs = torch.zeros(
            (self.num_envs, self.GUIDE_OBS_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        tilt_axis = torch.tensor(
            self.se_tilt_axis_cfg, dtype=torch.float32, device=self.device
        )
        tilt_axis_norm = torch.linalg.vector_norm(tilt_axis)
        if tilt_axis_norm < 1e-8:
            raise ValueError("seTiltAxis must be non-zero")
        self.se_tilt_axis = tilt_axis / tilt_axis_norm

    def _guide_quat_from_yaw_pitch(self, env_ids, yaw_deg, pitch_deg):
        count = len(env_ids)
        base_start = self.base_tracking_reference["wrist_initobj_pos"][
            env_ids, 0
        ]
        base_dir = _normalize(base_start)

        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).view(1, 3).expand(count, -1)
        yaw_quat = quat_from_angle_axis(torch.deg2rad(yaw_deg), z_axis)
        yaw_dir = _normalize(quat_apply(yaw_quat, base_dir))

        side_axis = torch.cross(z_axis, yaw_dir, dim=-1)
        side_small = torch.linalg.vector_norm(side_axis, dim=-1) < 1e-6
        if torch.any(side_small):
            side_axis[side_small] = torch.tensor(
                [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            )
        side_axis = _normalize(side_axis)
        pitch_quat = quat_from_angle_axis(torch.deg2rad(pitch_deg), side_axis)
        return quat_mul(pitch_quat, yaw_quat)

    def _guide_quat_from_tilt_pitch(self, env_ids, tilt_deg, pitch_deg):
        count = len(env_ids)
        base_start = self.base_tracking_reference["wrist_initobj_pos"][
            env_ids, 0
        ]
        base_dir = _normalize(base_start)

        tilt_axis = self.se_tilt_axis.view(1, 3).expand(count, -1)
        tilt_quat = quat_from_angle_axis(torch.deg2rad(tilt_deg), tilt_axis)
        tilted_dir = _normalize(quat_apply(tilt_quat, base_dir))

        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).view(1, 3).expand(count, -1)
        side_axis = torch.cross(z_axis, tilted_dir, dim=-1)
        side_small = torch.linalg.vector_norm(side_axis, dim=-1) < 1e-6
        if torch.any(side_small):
            side_axis[side_small] = self.se_tilt_axis
        side_axis = _normalize(side_axis)
        pitch_quat = quat_from_angle_axis(torch.deg2rad(pitch_deg), side_axis)
        return quat_mul(pitch_quat, tilt_quat)

    def _set_env_trajectory_rotations(self, env_ids, angle_ids):
        if self.se_guide_mode == "legacy_tilt":
            super()._set_env_trajectory_rotations(env_ids, angle_ids)
            angles = self.env_traj_angles[env_ids]
            angle_rad = torch.deg2rad(angles)
            self.env_se_yaw[env_ids] = angles
            self.env_se_pitch[env_ids] = 0.0
            self.env_se_guide_obs[env_ids] = torch.stack(
                [
                    torch.sin(angle_rad),
                    torch.cos(angle_rad),
                    torch.zeros_like(angle_rad),
                    torch.ones_like(angle_rad),
                ],
                dim=-1,
            )
            return

        self._ensure_base_tracking_reference()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        angle_ids = angle_ids.to(device=self.device, dtype=torch.long)
        pairs = self.se_angle_pairs[angle_ids]
        primary_deg = pairs[:, 0]
        pitch_deg = pairs[:, 1]
        if self.se_guide_mode == "tilt_pitch":
            guide_quat = self._guide_quat_from_tilt_pitch(
                env_ids, primary_deg, pitch_deg
            )
        else:
            guide_quat = self._guide_quat_from_yaw_pitch(
                env_ids, primary_deg, pitch_deg
            )

        self.env_angle_ids[env_ids] = angle_ids
        self.env_se_yaw[env_ids] = primary_deg
        self.env_se_pitch[env_ids] = pitch_deg
        self.env_traj_angles[env_ids] = torch.linalg.vector_norm(pairs, dim=-1)
        self.env_traj_quat[env_ids] = guide_quat
        self.env_traj_quat_inv[env_ids] = quat_conjugate(guide_quat)

        table_z = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device
        ).expand(len(env_ids), -1)
        self.env_traj_normal[env_ids] = quat_apply(guide_quat, table_z)

        primary_rad = torch.deg2rad(primary_deg)
        pitch_rad = torch.deg2rad(pitch_deg)
        self.env_se_guide_obs[env_ids] = torch.stack(
            [
                torch.sin(primary_rad),
                torch.cos(primary_rad),
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

    def _prepare_reset_robot_pose(self, env_ids, object_init_pose):
        if self.se_guide_mode == "legacy_tilt":
            return super()._prepare_reset_robot_pose(env_ids, object_init_pose)

        if not (self._is_sr_shadow_hand() or self._is_new_sr_hand_simple()):
            return None
        self._ensure_base_tracking_reference()
        if self._is_sr_shadow_hand() and not self._sr_reference_prepared:
            return None
        if self._is_new_sr_hand_simple() and not self._new_sr_reference_prepared:
            return None

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        robot_pose = self.robot_dof_default_pos.unsqueeze(0).repeat(
            len(env_ids), 1
        )
        base_rot_indices = self.arm_dof_indices[3:6]
        base_xyz_indices = self.arm_dof_indices[0:3]

        frame0_quat = self.tracking_reference["wrist_quat"][env_ids, 0]
        if self._is_sr_shadow_hand():
            frame0_angles = _quat_to_base_angles_xyz(frame0_quat)
            offset = quat_apply(
                frame0_quat,
                self._sr_palm_offset().view(1, 3).expand(len(env_ids), 3),
            )
        else:
            frame0_angles = _quat_to_base_angles_zyx(frame0_quat)
            offset = torch.zeros(
                (len(env_ids), 3),
                dtype=torch.float32,
                device=self.device,
            )

        robot_pose[:, base_rot_indices] = frame0_angles
        robot_pose[:, base_xyz_indices] = (
            object_init_pose[:, 0:3]
            + self.tracking_reference["wrist_initobj_pos"][env_ids, 0]
            - offset
        )
        return robot_pose

    def build_reference_dof_pose(self, env_ids, frame_id=0):
        """Return robot DOF pose for a transformed reference frame.

        This is for visualization/debugging. It places the virtual base at the
        requested wrist reference frame instead of freezing the reset pose.
        """
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        frame_id = int(frame_id)
        frame_id = max(0, min(frame_id, self.T_ref - 1))
        robot_pose = self.robot_dof_default_pos.unsqueeze(0).repeat(
            len(env_ids), 1
        )

        frame_quat = self.tracking_reference["wrist_quat"][
            env_ids, frame_id
        ]
        if self._is_sr_shadow_hand():
            frame_angles = _quat_to_base_angles_xyz(frame_quat)
            offset = quat_apply(
                frame_quat,
                self._sr_palm_offset().view(1, 3).expand(len(env_ids), 3),
            )
        else:
            frame_angles = _quat_to_base_angles_zyx(frame_quat)
            offset = torch.zeros(
                (len(env_ids), 3),
                dtype=torch.float32,
                device=self.device,
            )

        robot_pose[:, self.arm_dof_indices[0:3]] = (
            self.object_init_states[env_ids, 0:3]
            + self.tracking_reference["wrist_initobj_pos"][env_ids, frame_id]
            - offset
        )
        robot_pose[:, self.arm_dof_indices[3:6]] = frame_angles

        hand_qpos = self.tracking_reference["hand_qpos"][
            env_ids, frame_id
        ]
        if hand_qpos.shape[-1] == len(self.active_hand_dof_indices):
            robot_pose[:, self.active_hand_dof_indices] = hand_qpos
        return robot_pose

    def get_baseline_observation(self, residual_obs=None):
        if residual_obs is None:
            residual_obs = self.obs_dict["obs"]
        if getattr(self, "baseline_uses_residual_observation", False):
            return residual_obs
        if "seguide" not in self.obs_type.split("+"):
            return super().get_baseline_observation(residual_obs)

        pcl_dim = self.points_per_object * 3 if "objpcl" in self.obs_type else 0
        state_dim = residual_obs.shape[-1] - pcl_dim - self.GUIDE_OBS_DIM
        state_obs = residual_obs[:, :state_dim]
        pcl_obs = residual_obs[:, state_dim + self.GUIDE_OBS_DIM:]

        # The frozen tilted baseline was trained with a 3-D worldtilt/table
        # normal before the point cloud. The residual policy receives the new
        # 4-D sin/cos guide, so convert that slot back to the baseline format.
        if "objpcl" in self.obs_type.split("+"):
            return torch.cat([state_obs, self.env_traj_normal, pcl_obs], dim=-1)
        return torch.cat([state_obs, self.env_traj_normal], dim=-1)

    def compute_required_observations(self, obs_buf, obs_type, num_obs):
        if "seguide" not in obs_type.split("+"):
            return super().compute_required_observations(
                obs_buf, obs_type, num_obs
            )

        base_obs_type = "+".join(
            part for part in obs_type.split("+") if part != "seguide"
        )
        base_num_obs = num_obs - self.GUIDE_OBS_DIM
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
            obs_buf[:, :state_dim] = temporary[:, :state_dim]
            obs_buf[:, state_dim:state_dim + self.GUIDE_OBS_DIM] = (
                self.env_se_guide_obs
            )
            obs_buf[:, state_dim + self.GUIDE_OBS_DIM:] = temporary[
                :, state_dim:
            ]
        else:
            super().compute_required_observations(
                obs_buf[:, :-self.GUIDE_OBS_DIM],
                base_obs_type,
                base_num_obs,
            )
            obs_buf[:, -self.GUIDE_OBS_DIM:] = self.env_se_guide_obs

    def compute_reward(self):
        super().compute_reward()
        self.extras["se_yaw_deg"] = self.env_se_yaw.clone()
        self.extras["se_pitch_deg"] = self.env_se_pitch.clone()
        self.extras["se_guide_norm_deg"] = self.env_traj_angles.clone()
