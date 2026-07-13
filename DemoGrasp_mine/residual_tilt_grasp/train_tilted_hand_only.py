"""Train Shadow Hand grasping across one or more tilted-world angles."""

import json
import os
import sys
from datetime import datetime


def _set_cli_override(args, key, value):
    prefix = f"{key}="
    updated = list(args)
    for index, argument in enumerate(updated):
        if argument.startswith(prefix):
            updated[index] = f"{prefix}{value}"
            return updated
    updated.append(f"{prefix}{value}")
    return updated


def _reexec_viewer_with_physical_gpu():
    """Avoid CUDA/Vulkan ordinal mismatch caused by CUDA_VISIBLE_DEVICES."""
    marker = "_ROT_RL_VIEWER_REEXEC"
    if os.getenv(marker) == "1":
        return

    headless = next(
        (
            argument.split("=", 1)[1].lower()
            for argument in sys.argv[1:]
            if argument.startswith("headless=")
        ),
        "false",
    )
    if headless in ("true", "1", "yes"):
        return

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    entries = [entry.strip() for entry in visible_devices.split(",")]
    if len(entries) != 1 or not entries[0].isdigit():
        return

    physical_gpu = int(entries[0])
    if physical_gpu == 0:
        return

    args = _set_cli_override(
        sys.argv[1:], "sim_device", f"cuda:{physical_gpu}"
    )
    args = _set_cli_override(args, "rl_device", f"cuda:{physical_gpu}")
    args = _set_cli_override(args, "graphics_device_id", str(physical_gpu))
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment[marker] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    print(
        "Restarting viewer process with consistent physical GPU ordinals: "
        f"CUDA/Vulkan GPU {physical_gpu}",
        flush=True,
    )
    os.execvpe(
        sys.executable,
        [sys.executable, sys.argv[0], *args],
        environment,
    )


_reexec_viewer_with_physical_gpu()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gym
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from isaacgym import gymapi  # noqa: F401
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import tasks  # noqa: F401
from residual_tilt_grasp.tilted_hand_only_grasp import TiltedHandOnlyGrasp


TASK_NAME = "tilted_hand_only_grasp"


def _format_angle_for_run_name(angle):
    value = float(angle)
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _default_run_name(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hand_name = str(cfg.hand.name)
    angles = cfg.get("tilt_angles", [0.0, 15.0, 30.0, 45.0])
    angle_text = "-".join(_format_angle_for_run_name(angle) for angle in angles)
    return f"{timestamp}_{hand_name}_angles_{angle_text}"


def configure_tilted_training(cfg):
    with open_dict(cfg):
        cfg.task.name = TASK_NAME
        cfg.task_name = TASK_NAME

        initial_base_dof_pos = cfg.get("initial_base_dof_pos", None)
        initial_hand_dof_pos = cfg.get("initial_hand_dof_pos", None)
        if initial_base_dof_pos is not None:
            if len(initial_base_dof_pos) != int(cfg.hand.num_arm_dofs):
                raise ValueError(
                    "initial_base_dof_pos must contain "
                    f"{cfg.hand.num_arm_dofs} values"
                )
            default_dof_pos = list(cfg.hand.default_dof_pos)
            default_dof_pos[:cfg.hand.num_arm_dofs] = list(initial_base_dof_pos)
            cfg.hand.default_dof_pos = default_dof_pos
        if initial_hand_dof_pos is not None:
            expected = len(cfg.hand.default_dof_pos) - int(cfg.hand.num_arm_dofs)
            if len(initial_hand_dof_pos) != expected:
                raise ValueError(
                    f"initial_hand_dof_pos must contain {expected} values"
                )
            default_dof_pos = list(cfg.hand.default_dof_pos)
            default_dof_pos[cfg.hand.num_arm_dofs:] = list(initial_hand_dof_pos)
            cfg.hand.default_dof_pos = default_dof_pos

        cfg.task.env.armController = "qpos"
        # Allow the hand config or CLI to override the reference trajectory.
        default_reference_files = {
            "shadow_simple": "tasks/grasp_ref_shadow.pkl",
            "sr_shadow_hand": "tasks/grasp_ref_sr_shadow.pkl",
            "new_sr_hand_simple": "tasks/grasp_ref_shadow.pkl",
        }
        default_reference_file = default_reference_files.get(
            str(cfg.hand.name),
            cfg.task.env.get(
                "trackingReferenceFile", "tasks/grasp_ref_sr_shadow.pkl"
            ),
        )
        cfg.task.env.trackingReferenceFile = cfg.get(
            "tracking_reference_file",
            default_reference_file,
        )
        cfg.task.env.trackingReferenceLiftTimestep = 11
        cfg.task.env.randomizeTrackingReference = True
        cfg.task.env.randomizeGraspPose = True
        cfg.task.env.resetDofPosRandomInterval = 0
        # Otherwise Grasp.reset_idx ignores the configured initial hand pose
        # and samples every hand joint across its complete range.
        cfg.task.env.resetHandDofPosFullRange = cfg.get(
            "initial_hand_randomization_full_range", False
        )
        cfg.task.env.coarseGraspHandDofPos = cfg.get(
            "coarse_grasp_hand_dof_pos", None
        )
        cfg.task.env.graspDeltaScale = cfg.get(
            "grasp_delta_scale",
            cfg.task.env.randomizeGraspPoseRange,
        )

        # Read numActions from the hand config (30 for sr_shadow_hand) instead
        # of hardcoding, so hand replacement is seamless.
        hand_actions = int(cfg.hand.get("numActions", 24))
        cfg.task.env.numActions = hand_actions
        cfg.hand.numActions = hand_actions
        cfg.hand.num_obs_dict.lastact = hand_actions
        include_world_tilt = bool(cfg.get("tilt_include_observation", True))
        if include_world_tilt:
            cfg.hand.num_obs_dict.worldtilt = 3

        cfg.task.env.worldTiltAngles = cfg.get(
            "tilt_angles", [0.0, 15.0, 30.0, 45.0]
        )
        cfg.task.env.worldTiltAxis = cfg.get(
            "tilt_axis", [0.0, 1.0, 0.0]
        )
        cfg.task.env.worldTiltSampling = cfg.get(
            "tilt_sampling", "random"
        )
        cfg.task.env.worldTiltLogInterval = cfg.get(
            "tilt_log_interval", 100
        )
        cfg.task.env.lockStructuralWrist = cfg.get(
            "weld_wrist",
            cfg.get(
                "lock_structural_wrist",
                str(cfg.hand.name) == "new_sr_hand_simple",
            ),
        )
        cfg.task.env.policyControlsStructuralWrist = cfg.get(
            "policy_controls_wrist",
            cfg.get(
                "policy_controls_structural_wrist",
                str(cfg.hand.name) != "new_sr_hand_simple",
            ),
        )
        cfg.task.env.worldTiltPointCloudFrame = cfg.get(
            "tilt_pointcloud_frame", "table"
        )

        # The original hand-only checkpoint was trained with 40-step episodes.
        # Keep that baseline unless a dedicated tilted-task override is given.
        cfg.task.env.episodeLength = cfg.get("tilt_episode_length", 40)

        # Angles now rotate per-env trajectories rather than shared gravity, so
        # different angles can coexist in one vectorized rollout. Keep PPO's
        # configured nsteps instead of collecting one rollout per angle.
        if len(cfg.task.env.worldTiltAngles) > 1:
            cfg.train.params.sampler = cfg.train.params.get(
                "sampler", "random"
            )

        # Every angle, including 0 degrees, must use the same physical scene.
        # TiltedHandOnlyGrasp builds its own rotatable table geometry.
        cfg.task.env.render.enable = False
        cfg.task.env.render.appearance_realistic = False

        if (
            include_world_tilt
            and "worldtilt" not in cfg.task.env.observationType.split("+")
        ):
            cfg.task.env.observationType += "+worldtilt"
        elif (
            not include_world_tilt
            and "worldtilt" in cfg.task.env.observationType.split("+")
        ):
            cfg.task.env.observationType = "+".join(
                part
                for part in cfg.task.env.observationType.split("+")
                if part != "worldtilt"
            )

        # Assign after all hand-specific changes because OmegaConf materializes
        # this interpolation as a separate task configuration.
        cfg.task.hand_config = cfg.hand


def log_viewer_device(cfg):
    if bool(cfg.headless):
        return
    print(
        f"Viewer devices: CUDA_VISIBLE_DEVICES="
        f"{os.getenv('CUDA_VISIBLE_DEVICES', '') or '<unset>'}, "
        f"sim={cfg.sim_device}, rl={cfg.rl_device}, "
        f"Vulkan graphics_device_id={cfg.graphics_device_id}",
        flush=True,
    )


def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test

    if not is_testing:
        run_name = cfg.get(
            "run_name",
            _default_run_name(cfg),
        )
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as file:
            json.dump(OmegaConf.to_container(cfg), file, indent=4)
    else:
        log_dir = None

    if train_param.name != "ppo_onestep":
        raise ValueError("Tilted hand-only training supports PPOOneStep only")

    from algo import ppo_onestep
    from residual_tilt_grasp.tilted_ppo import TiltedPPO

    runner = TiltedPPO(
        vec_env=env,
        actor_critic_class=ppo_onestep.ActorCritic,
        train_param=train_param,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=(
            getattr(env, "policy_action_dim", None)
            or env.num_active_hand_dofs
        ),
    )

    if cfg.checkpoint:
        if is_testing:
            runner.load_pretrained(cfg.checkpoint, is_testing=True)
        elif cfg.get("resume", False):
            runner.load_resume(cfg.checkpoint)
        else:
            runner.load_pretrained(
                cfg.checkpoint,
                action_std=cfg.get("pretrained_action_std", 0.15),
            )
            runner.configure_pretrained_finetuning(
                learning_rate=cfg.get(
                    "pretrained_learning_rate", 3e-5
                ),
                epochs=cfg.get("pretrained_epochs", 2),
                freeze_backbone=cfg.get(
                    "pretrained_freeze_backbone", True
                ),
            )
            eval_rounds = int(cfg.get("pretrained_eval_rounds", 1))
            if eval_rounds > 0:
                runner.evaluate_pretrained(eval_rounds)
    return runner


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    set_np_formatting()
    configure_tilted_training(cfg)
    log_viewer_device(cfg)
    print(
        "Tilt training schedule: "
        f"angles={list(cfg.task.env.worldTiltAngles)}, "
        f"sampling={cfg.task.env.worldTiltSampling}, "
        f"rollouts_per_update={cfg.train.params.nsteps}, "
        f"sampler={cfg.train.params.get('sampler', 'sequential')}"
    )
    isaacgym_task_map[TASK_NAME] = TiltedHandOnlyGrasp

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

    runner = build_runner(cfg, env)
    runner.run()


if __name__ == "__main__":
    main()
