"""Train bounded wrist/finger residuals over a frozen Shadow Hand policy."""

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
    marker = "_SROT_RL_RESIDUAL_VIEWER_REEXEC"
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
        "Restarting residual viewer process with consistent physical GPU "
        f"ordinals: CUDA/Vulkan GPU {physical_gpu}",
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
import torch
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import tasks  # noqa: F401
from residual_tilt_grasp.residual_tilted_grasp import ResidualTiltedGrasp


TASK_NAME = "residual_tilted_hand_only_grasp"
DEFAULT_NEW_SR_BASELINE = (
    "runs_ppo/2026-06-18_05-41-34_new_sr_hand_simple_angles_0/model_0.pt"
)


def _format_angle_for_run_name(angle):
    value = float(angle)
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _default_run_name(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hand_name = str(cfg.hand.name)
    angles = cfg.get("tilt_angles", [0.0, 15.0, 30.0, 45.0])
    angle_text = "-".join(_format_angle_for_run_name(angle) for angle in angles)
    return f"{timestamp}_{hand_name}_residual_angles_{angle_text}"


def _as_list(value, expected_size, name):
    values = list(value)
    if len(values) != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    return values


def configure_residual_training(cfg):
    with open_dict(cfg):
        base_nminibatches = int(cfg.train.params.nminibatches)
        cfg.task.name = TASK_NAME
        cfg.task_name = TASK_NAME

        requested_envs = int(cfg.get("num_envs", cfg.task.env.numEnvs))
        safe_env_limit = int(cfg.get("residual_physx_safe_num_envs", 4096))
        allow_large_envs = bool(cfg.get("allow_large_physx_envs", False))
        if (
            str(cfg.hand.name) == "sr_shadow_hand"
            and requested_envs > safe_env_limit
            and not allow_large_envs
        ):
            raise ValueError(
                "sr residual training is likely to overflow GPU PhysX "
                f"memory at num_envs={requested_envs}. Use "
                f"num_envs<={safe_env_limit} or explicitly set "
                "+allow_large_physx_envs=True if you want to risk it. "
                "Suggested start: num_envs=3200."
            )

        cfg.task.env.armController = "qpos"
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

        cfg.task.env.episodeLength = cfg.get("tilt_episode_length", 40)

        if len(cfg.task.env.worldTiltAngles) > 1:
            cfg.train.params.sampler = cfg.train.params.get(
                "sampler", "random"
            )

        # Keep the PPO minibatch size close to the configured baseline.
        # Trajectory rotations are sampled per env, so nsteps no longer grows
        # with the number of angles.
        default_nminibatches = base_nminibatches
        cfg.train.params.nminibatches = int(
            cfg.get("residual_nminibatches", default_nminibatches)
        )
        if cfg.train.params.nminibatches <= 0:
            raise ValueError("residual_nminibatches must be positive")

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

        residual_mode = str(cfg.get("residual_mode", "hybrid"))
        if residual_mode not in ("wrist", "finger", "hybrid"):
            raise ValueError(
                "residual_mode must be wrist, finger, or hybrid"
            )
        cfg.task.env.residualMode = residual_mode
        cfg.task.env.residualFrame = cfg.get("residual_frame", "table")
        cfg.task.env.residualScope = cfg.get(
            "residual_scope", "full_trajectory"
        )
        cfg.task.env.residualPoseComposition = cfg.get(
            "residual_pose_composition", "full_se3"
        )
        cfg.task.env.residualTranslationScale = _as_list(
            cfg.get(
                "residual_translation_scale", [0.02, 0.02, 0.02]
            ),
            3,
            "residual_translation_scale",
        )
        cfg.task.env.residualRotationScaleDeg = _as_list(
            cfg.get(
                "residual_rotation_scale_deg", [10.0, 10.0, 10.0]
            ),
            3,
            "residual_rotation_scale_deg",
        )
        cfg.task.env.fingerResidualAlpha = float(
            cfg.get("finger_residual_alpha", 0.3)
        )
        cfg.task.env.fingerResidualMinRatio = float(
            cfg.get("finger_residual_min_ratio", 0.02)
        )
        cfg.task.env.fingerResidualMaxRatio = float(
            cfg.get("finger_residual_max_ratio", 0.3)
        )
        cfg.task.env.residualTiltModulation = cfg.get(
            "tilt_modulation", "normalized_sin"
        )
        cfg.task.env.residualTiltModulationMaxAngle = float(
            cfg.get(
                "tilt_modulation_max_angle",
                max(
                    abs(float(angle))
                    for angle in cfg.task.env.worldTiltAngles
                ),
            )
        )
        cfg.task.env.wristTiltModulation = bool(
            cfg.get("wrist_tilt_modulation", True)
        )
        cfg.task.env.fingerTiltModulation = bool(
            cfg.get("finger_tilt_modulation", True)
        )

        cfg.train.params.cliprange = float(
            cfg.get("residual_cliprange", 0.1)
        )
        cfg.train.params.init_noise_std = float(
            cfg.get("residual_action_std", 0.3)
        )
        cfg.train.params.optim_stepsize = float(
            cfg.get("residual_learning_rate", 3e-4)
        )
        cfg.train.params.noptepochs = int(
            cfg.get("residual_epochs", 2)
        )
        cfg.train.params.schedule = cfg.get(
            "residual_lr_schedule", "fixed"
        )
        cfg.train.params.ent_coef = float(
            cfg.get("residual_entropy_coef", 1e-3)
        )

        if (
            str(cfg.hand.name) == "new_sr_hand_simple"
            and not cfg.get("baseline_checkpoint", "")
            and not cfg.get("checkpoint", "")
        ):
            cfg.baseline_checkpoint = DEFAULT_NEW_SR_BASELINE

        cfg.task.hand_config = cfg.hand


def _load_checkpoint_payload(path):
    return torch.load(path, map_location="cpu")


def _is_residual_checkpoint(payload):
    return (
        isinstance(payload, dict)
        and "model_state_dict" in payload
        and "residual_metadata" in payload
    )


def _resolve_saved_baseline(residual_checkpoint, saved_path):
    saved_path = str(saved_path or "")
    if not saved_path:
        return ""

    candidates = [saved_path]
    if not os.path.isabs(saved_path):
        candidates.append(
            os.path.join(
                os.path.dirname(os.path.abspath(residual_checkpoint)),
                saved_path,
            )
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def get_checkpoint_paths(cfg):
    baseline_checkpoint = str(cfg.get("baseline_checkpoint", "") or "")
    residual_checkpoint = str(cfg.get("residual_checkpoint", "") or "")
    checkpoint = str(cfg.checkpoint or "")

    if baseline_checkpoint:
        if not os.path.isfile(baseline_checkpoint):
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {baseline_checkpoint}"
            )
        payload = _load_checkpoint_payload(baseline_checkpoint)
        if _is_residual_checkpoint(payload):
            if (
                residual_checkpoint
                and os.path.abspath(residual_checkpoint)
                != os.path.abspath(baseline_checkpoint)
            ):
                raise ValueError(
                    "baseline_checkpoint points to a residual checkpoint, "
                    "but residual_checkpoint specifies a different file"
                )
            residual_checkpoint = baseline_checkpoint
            baseline_checkpoint = _resolve_saved_baseline(
                residual_checkpoint, payload.get("baseline_checkpoint")
            )

    if checkpoint and not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if checkpoint:
        if residual_checkpoint and not baseline_checkpoint:
            baseline_checkpoint = checkpoint
        elif not residual_checkpoint:
            payload = _load_checkpoint_payload(checkpoint)
            if _is_residual_checkpoint(payload):
                residual_checkpoint = checkpoint
                if not baseline_checkpoint:
                    baseline_checkpoint = _resolve_saved_baseline(
                        checkpoint, payload.get("baseline_checkpoint")
                    )
            elif not baseline_checkpoint:
                baseline_checkpoint = checkpoint

    if residual_checkpoint and not os.path.isfile(residual_checkpoint):
        raise FileNotFoundError(
            f"Residual checkpoint not found: {residual_checkpoint}"
        )

    if residual_checkpoint and not baseline_checkpoint:
        payload = _load_checkpoint_payload(residual_checkpoint)
        if _is_residual_checkpoint(payload):
            baseline_checkpoint = _resolve_saved_baseline(
                residual_checkpoint, payload.get("baseline_checkpoint")
            )

    if not baseline_checkpoint:
        raise ValueError(
            "A frozen baseline is required. For a legacy hand-only weight, "
            "set checkpoint=path/model.pt or "
            "+baseline_checkpoint=path/model.pt. For a residual checkpoint "
            "whose recorded baseline has moved, also set "
            "+baseline_checkpoint=path/model.pt."
        )
    if not os.path.isfile(baseline_checkpoint):
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_checkpoint}"
        )
    return baseline_checkpoint, residual_checkpoint


def residual_metadata(cfg, env):
    return {
        "mode": env.residual_mode,
        "action_dim": env.residual_action_dim,
        "frame": env.residual_frame,
        "scope": env.residual_scope,
        "pose_composition": env.residual_pose_composition,
        "translation_scale": list(
            cfg.task.env.residualTranslationScale
        ),
        "rotation_scale_deg": list(
            cfg.task.env.residualRotationScaleDeg
        ),
        "finger_alpha": cfg.task.env.fingerResidualAlpha,
        "finger_min_ratio": cfg.task.env.fingerResidualMinRatio,
        "finger_max_ratio": cfg.task.env.fingerResidualMaxRatio,
        "tilt_modulation": cfg.task.env.residualTiltModulation,
        "tilt_modulation_max_angle": (
            cfg.task.env.residualTiltModulationMaxAngle
        ),
        "wrist_tilt_modulation": cfg.task.env.wristTiltModulation,
        "finger_tilt_modulation": cfg.task.env.fingerTiltModulation,
        "tilt_angles": list(cfg.task.env.worldTiltAngles),
        "pointcloud_frame": cfg.task.env.worldTiltPointCloudFrame,
    }


def log_viewer_device(cfg):
    if bool(cfg.headless):
        return
    print(
        f"Viewer devices: DISPLAY={os.getenv('DISPLAY', '') or '<unset>'}, "
        f"CUDA_VISIBLE_DEVICES="
        f"{os.getenv('CUDA_VISIBLE_DEVICES', '') or '<unset>'}, "
        f"sim={cfg.sim_device}, rl={cfg.rl_device}, "
        f"Vulkan graphics_device_id={cfg.graphics_device_id}, "
        f"force_render={cfg.force_render}",
        flush=True,
    )


def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test
    baseline_checkpoint, residual_checkpoint = get_checkpoint_paths(cfg)

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
        raise ValueError("Residual training supports PPOOneStep only")

    from residual_tilt_grasp.residual_actor_critic import ResidualActorCritic
    from residual_tilt_grasp.residual_ppo import ResidualPPO

    runner = ResidualPPO(
        vec_env=env,
        actor_critic_class=ResidualActorCritic,
        train_param=train_param,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=env.residual_action_dim,
        baseline_checkpoint=baseline_checkpoint,
        residual_metadata=residual_metadata(cfg, env),
        per_angle_advantage=bool(
            cfg.get("per_angle_advantage", True)
        ),
        advantage_eps=float(cfg.get("advantage_eps", 1e-8)),
        advantage_min_samples=int(
            cfg.get("advantage_min_samples", 32)
        ),
        zero_inactive_angle_advantage=bool(
            cfg.get("zero_inactive_angle_advantage", True)
        ),
        freeze_residual_backbone=bool(
            cfg.get("residual_freeze_backbone", True)
        ),
        initialize_residual_features=bool(
            cfg.get("residual_initialize_from_baseline", True)
        ),
    )

    if residual_checkpoint:
        runner.load_residual(
            residual_checkpoint,
            resume=bool(cfg.get("resume", False) and not is_testing),
        )
        if is_testing:
            runner.actor_critic.eval()
    elif is_testing:
        raise ValueError(
            "test=True requires +residual_checkpoint=path/model.pt"
        )
    elif not residual_checkpoint:
        runner.evaluate_initial_policy(
            rounds=int(cfg.get("residual_initial_eval_rounds", 1))
        )
    return runner


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    set_np_formatting()
    configure_residual_training(cfg)
    log_viewer_device(cfg)
    print(
        "Residual tilt training: "
        f"angles={list(cfg.task.env.worldTiltAngles)}, "
        f"sampling={cfg.task.env.worldTiltSampling}, "
        f"rollouts_per_update={cfg.train.params.nsteps}, "
        f"minibatches={cfg.train.params.nminibatches}, "
        "samples_per_minibatch="
        f"{cfg.task.env.numEnvs * cfg.train.params.nsteps // cfg.train.params.nminibatches}, "
        f"mode={cfg.task.env.residualMode}, "
        f"modulation={cfg.task.env.residualTiltModulation}, "
        f"lr={cfg.train.params.optim_stepsize:g}, "
        f"std={cfg.train.params.init_noise_std:g}"
    )
    isaacgym_task_map[TASK_NAME] = ResidualTiltedGrasp

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
