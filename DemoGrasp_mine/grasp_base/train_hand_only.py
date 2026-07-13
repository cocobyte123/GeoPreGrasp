"""Train a hand-only Shadow Hand policy without modifying the original tasks."""

import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gym
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# Isaac Gym must be imported before modules that import torch-based bindings.
from isaacgym import gymapi  # noqa: F401
import torch
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import tasks  # noqa: F401 - registers the original tasks
from grasp_base.hand_only_grasp import HandOnlyGrasp


TASK_NAME = "hand_only_grasp"


def configure_hand_only(cfg):
    """Apply only the invariants required by the isolated hand-only task."""
    with open_dict(cfg):
        cfg.task.name = TASK_NAME
        cfg.task_name = TASK_NAME
        cfg.task.hand_config = cfg.hand

        cfg.task.env.armController = "qpos"
        cfg.task.env.trackingReferenceFile = "tasks/grasp_ref_shadow.pkl"
        cfg.task.env.trackingReferenceLiftTimestep = 11
        cfg.task.env.randomizeTrackingReference = True
        cfg.task.env.randomizeGraspPose = True
        cfg.task.env.resetDofPosRandomInterval = 0

        # Six virtual wrist joints + 18 active hand joints.
        cfg.task.env.numActions = 24
        cfg.hand.numActions = 24
        cfg.hand.num_obs_dict.lastact = 24


def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test

    if not is_testing:
        run_name = cfg.get(
            "run_name",
            f"{TASK_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as file:
            json.dump(OmegaConf.to_container(cfg), file, indent=4)
    else:
        log_dir = None

    if train_param.name != "ppo_onestep":
        raise ValueError("Hand-only training currently supports PPOOneStep only")

    from algo import ppo_onestep

    runner = ppo_onestep.PPO(
        vec_env=env,
        actor_critic_class=ppo_onestep.ActorCritic,
        train_param=train_param,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=env.num_active_hand_dofs,
    )

    if cfg.checkpoint:
        if is_testing:
            runner.test(cfg.checkpoint)
        else:
            runner.load(cfg.checkpoint)
    return runner


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    set_np_formatting()
    configure_hand_only(cfg)
    isaacgym_task_map[TASK_NAME] = HandOnlyGrasp

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
            f"videos/{TASK_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            step_trigger=lambda step: step % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )

    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    runner = build_runner(cfg, env)
    runner.run()


if __name__ == "__main__":
    main()
