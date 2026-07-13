"""Visualize SE wrist guide poses and reference trajectories.

This script is intentionally separate from PPO training. It builds the same
``SEResidualGrasp`` environment, forces a deterministic angle grid, and either
shows one transformed reference frame or plays the transformed reference
trajectory frame by frame.
"""

import os
import sys
import math
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from isaacgym import gymapi  # noqa: F401
from isaacgym import gymtorch
import torch
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import tasks  # noqa: F401
from residual_se_grasp.se_residual_grasp import SEResidualGrasp
from residual_se_grasp.train_se_reslearn import configure_se_training


TASK_NAME = "se_residual_grasp"


def configure_play(cfg):
    with open_dict(cfg):
        if "train" not in cfg or cfg.get("train", None) is None:
            cfg.train = OmegaConf.load(
                os.path.join(PROJECT_ROOT, "tasks", "train", "PPOOneStep.yaml")
            )
        cfg.se_view_only = True
        cfg.headless = bool(cfg.get("headless", False))
        cfg.force_render = bool(cfg.get("force_render", True))
        cfg.se_sampling = cfg.get("se_sampling", "cycle")
        cfg.num_envs = int(cfg.get("num_envs", 20))
    configure_se_training(cfg)
    with open_dict(cfg):
        play_mode = str(cfg.get("play_mode", "static"))
        dynamic_object = bool(
            cfg.get(
                "play_dynamic_object",
                play_mode == "play" or bool(cfg.get("play_eval_success", False)),
            )
        )
        if dynamic_object:
            cfg.task.env.fixObjectBase = False
            cfg.task.env.disableObjectGravity = False


def force_angle_grid(env, cfg):
    env_ids = torch.arange(env.num_envs, device=env.device)
    if cfg.task.env.seGuideMode == "legacy_tilt":
        guide_count = len(cfg.task.env.seLegacyTiltAngles)
    else:
        guide_count = int(env.se_angle_pairs.shape[0])
    angle_ids = (torch.arange(env.num_envs, device=env.device) % guide_count).long()
    env._set_env_trajectory_rotations(env_ids, angle_ids)
    return env_ids


def center_preview_objects(env, cfg):
    if hasattr(env, "canonical_table_top"):
        object_pos = env.canonical_table_top.clone()
    else:
        object_pos = env.table_start_pos.clone()
        object_pos[:, 2] = env.table_heights
    object_pos = object_pos[:env.num_envs]
    object_pos[:, 2] = object_pos[:, 2] + float(
        cfg.get("se_preview_object_height", 0.08)
    )
    env.root_state_tensor[env.object_indices, 0:3] = object_pos
    env.root_state_tensor[env.object_indices, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=env.root_state_tensor.dtype,
        device=env.device,
    ).view(1, 4)
    env.root_state_tensor[env.object_indices, 7:13] = 0.0
    env.object_init_states[:, 0:3] = object_pos
    env.object_init_states[:, 3:7] = env.root_state_tensor[
        env.object_indices, 3:7
    ]
    object_indices = env.object_indices.to(torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_state_tensor),
        gymtorch.unwrap_tensor(object_indices),
        env.num_envs,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    return env.root_state_tensor[env.object_indices].clone()


def apply_dof_pose(env, dof_pose):
    env.robot_dof_pos[:, :] = dof_pose
    env.robot_dof_vel[:, :] = 0.0
    env.prev_targets[:, :env.num_robot_dofs] = dof_pose
    env.cur_targets[:, :env.num_robot_dofs] = dof_pose
    robot_indices = env.robot_indices.to(torch.int32)
    env.gym.set_dof_position_target_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.prev_targets),
        gymtorch.unwrap_tensor(robot_indices),
        env.num_envs,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.robot_dof_state),
        gymtorch.unwrap_tensor(robot_indices),
        env.num_envs,
    )
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def set_position_target(env, target_pose):
    env.gym.set_dof_position_target_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.cur_targets),
    )


def maybe_lock_object(env, locked_object_state):
    if locked_object_state is None:
        return
    env.root_state_tensor[env.object_indices] = locked_object_state
    object_indices = env.object_indices.to(torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_state_tensor),
        gymtorch.unwrap_tensor(object_indices),
        env.num_envs,
    )


def step_to_target(env, target_pose, locked_object_state=None, render=True):
    env.cur_targets[:, :env.num_robot_dofs] = target_pose
    env.gym.set_dof_position_target_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.cur_targets),
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    maybe_lock_object(env, locked_object_state)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    if render:
        env.render()


def unwrap_angles(angles, time_dim=0):
    if angles.shape[time_dim] < 2:
        return angles
    moved = angles.movedim(time_dim, 0)
    delta = moved[1:] - moved[:-1]
    wrapped_delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    wrapped_delta = torch.where(
        (wrapped_delta == -math.pi) & (delta > 0), -wrapped_delta, wrapped_delta
    )
    result = moved.clone()
    result[1:] = moved[0] + torch.cumsum(wrapped_delta, dim=0)
    return result.movedim(0, time_dim)


def build_pose_sequence(env, env_ids, start_frame, end_frame):
    poses = [
        env.build_reference_dof_pose(env_ids, frame_id=frame)
        for frame in range(start_frame, end_frame + 1)
    ]
    sequence = torch.stack(poses, dim=0)
    rot_indices = torch.tensor(
        env.arm_dof_indices[3:6], dtype=torch.long, device=env.device
    )
    sequence[:, :, rot_indices] = unwrap_angles(
        sequence[:, :, rot_indices], time_dim=0
    )
    return sequence


def report_success(env, cfg, episode):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.compute_observations()
    env.compute_reward()
    success = env.successes.detach()
    mean_success = success.float().mean().item()
    print(
        f"Episode {episode} success={mean_success:.3f} "
        f"({int(success.sum().item())}/{success.numel()})",
        flush=True,
    )


def print_preview_summary(env, cfg):
    count = min(10, env.num_envs)
    label = "tilt" if cfg.task.env.seGuideMode == "tilt_pitch" else "yaw"
    frame0_rel = env.tracking_reference["wrist_initobj_pos"][
        :count, 0
    ].detach().cpu()
    base_qpos = env.robot_dof_pos[
        :count, env.arm_dof_indices
    ].detach().cpu()
    print(
        "SE trajectory preview: "
        f"mode={cfg.task.env.seGuideMode}, "
        f"{label}={env.env_se_yaw[:count].detach().cpu().tolist()}, "
        f"pitch={env.env_se_pitch[:count].detach().cpu().tolist()}, "
        f"frame0_rel_xyz={frame0_rel.tolist()}, "
        f"base_qpos={base_qpos.tolist()}",
        flush=True,
    )


def run_static(env, cfg, env_ids, locked_object_state):
    frame = int(cfg.get("frame", cfg.get("se_view_reference_frame", 0)))
    pose = env.build_reference_dof_pose(env_ids, frame_id=frame)
    apply_dof_pose(env, pose)
    print_preview_summary(env, cfg)
    steps = int(cfg.get("steps", cfg.get("se_view_steps", 600)))
    print(f"Static reference frame={frame}, steps={steps}", flush=True)
    for _ in range(steps):
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            break
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        maybe_lock_object(env, locked_object_state)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.render()


def run_play(env, cfg, env_ids, locked_object_state):
    start_frame = int(cfg.get("start_frame", 0))
    end_frame = int(cfg.get("end_frame", env.T_ref - 1))
    start_frame = max(0, min(start_frame, env.T_ref - 1))
    end_frame = max(start_frame, min(end_frame, env.T_ref - 1))
    interpolation_steps = int(cfg.get("interpolation_steps", 20))
    frame_hold = int(cfg.get("frame_hold", cfg.get("frame_repeat", 1)))
    episodes = int(cfg.get("episodes", 0))
    eval_success = bool(cfg.get("play_eval_success", False))
    episode = 0
    pose_sequence = build_pose_sequence(env, env_ids, start_frame, end_frame)
    print_preview_summary(env, cfg)
    print(
        f"Playing frames {start_frame}..{end_frame}, "
        f"interpolation_steps={interpolation_steps}, "
        f"frame_hold={frame_hold}, episodes={episodes}",
        flush=True,
    )
    current_pose = pose_sequence[0]
    apply_dof_pose(env, current_pose)
    while True:
        for frame_offset in range(pose_sequence.shape[0]):
            target_pose = pose_sequence[frame_offset]
            for step in range(1, interpolation_steps + 1):
                alpha = step / max(interpolation_steps, 1)
                pose = current_pose + alpha * (target_pose - current_pose)
                if (
                    env.viewer is not None
                    and env.gym.query_viewer_has_closed(env.viewer)
                ):
                    return
                step_to_target(env, pose, locked_object_state)
            current_pose = target_pose
            for _ in range(frame_hold):
                if (
                    env.viewer is not None
                    and env.gym.query_viewer_has_closed(env.viewer)
                ):
                    return
                step_to_target(env, current_pose, locked_object_state)
        episode += 1
        if eval_success:
            report_success(env, cfg, episode)
        if episodes > 0 and episode >= episodes:
            return
        current_pose = pose_sequence[0]
        apply_dof_pose(env, current_pose)


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    set_np_formatting()
    configure_play(cfg)
    isaacgym_task_map[TASK_NAME] = SEResidualGrasp
    rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(
        cfg.seed,
        torch_deterministic=cfg.torch_deterministic,
        rank=rank,
    )
    print(
        "SE trajectory viewer: "
        f"mode={cfg.task.env.seGuideMode}, "
        f"tilt={list(cfg.task.env.seTiltAngles)}, "
        f"yaw={list(cfg.task.env.seYawAngles)}, "
        f"pitch={list(cfg.task.env.sePitchAngles)}, "
        f"num_envs={cfg.task.env.numEnvs}",
        flush=True,
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
    env.reset()
    env_ids = force_angle_grid(env, cfg)
    locked_object_state = center_preview_objects(env, cfg)
    if not bool(cfg.get("play_lock_object", False)):
        locked_object_state = None
    mode = str(cfg.get("play_mode", "static"))
    if mode == "static":
        run_static(env, cfg, env_ids, locked_object_state)
    elif mode == "play":
        run_play(env, cfg, env_ids, locked_object_state)
    else:
        raise ValueError("play_mode must be 'static' or 'play'")


if __name__ == "__main__":
    main()
