"""Train residual PPO on a two-DOF SE-style pregrasp guide family."""

import json
import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone


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
from residual_se_grasp.se_residual_grasp import SEResidualGrasp
from residual_tilt_grasp.train_tilted_hand_only_reslearn import (
    configure_residual_training,
    get_checkpoint_paths,
    log_viewer_device,
)


TASK_NAME = "se_residual_grasp"
CHINA_TZ = timezone(timedelta(hours=8))


def _dataset_name_from_cfg(cfg):
    asset_cfg = cfg.task.env.asset
    if bool(asset_cfg.get("multiObject", True)):
        asset_path = str(asset_cfg.get("multiObjectList", "") or "")
    else:
        asset_path = str(asset_cfg.get("objectAssetFile", "") or "")
    dataset = asset_path.split("/", 1)[0].strip()
    return dataset or "dataset"


def _default_run_name(cfg):
    timestamp = datetime.now(CHINA_TZ).strftime("%Y%m%d_%H%M")
    return f"{_dataset_name_from_cfg(cfg)}_{timestamp}"


def configure_se_training(cfg):
    with open_dict(cfg):
        if "train" not in cfg or cfg.get("train", None) is None:
            cfg.train = OmegaConf.load(
                os.path.join(PROJECT_ROOT, "tasks", "train", "PPOOneStep.yaml")
            )
        if cfg.get("se_baseline_checkpoint", "") and not cfg.get(
            "baseline_checkpoint", ""
        ):
            cfg.baseline_checkpoint = cfg.se_baseline_checkpoint
        cfg.train.params.is_vision = bool(cfg.get("se_is_vision", True))
        cfg.tilt_include_observation = False
        cfg.tilt_modulation = cfg.get("se_residual_modulation", "none")
        cfg.wrist_tilt_modulation = bool(
            cfg.get("se_wrist_modulation", False)
        )
        cfg.finger_tilt_modulation = bool(
            cfg.get("se_finger_modulation", False)
        )

    configure_residual_training(cfg)

    with open_dict(cfg):
        if bool(cfg.get("se_view_only", False)):
            cfg.task.env.useObjectVhacd = False
            cfg.task.env.enablePointCloud = False
            cfg.task.env.fixObjectBase = True
            cfg.task.env.disableObjectGravity = True
            cfg.task.env.resetRandomRot = "fixed"
            preview_center = cfg.get(
                "se_preview_object_xy",
                [0.51, -0.075],
            )
            preview_height = float(cfg.get("se_preview_object_height", 0.08))
            cfg.task.env.resetPositionRange = [
                [float(preview_center[0]), float(preview_center[0])],
                [float(preview_center[1]), float(preview_center[1])],
                [preview_height, preview_height],
            ]
            cfg.task.env.asset.multiObject = False
            cfg.task.env.asset.useDistractorObjects = False
            cfg.task.env.asset.objectAssetFile = cfg.get(
                "se_preview_object_asset",
                "union_ycb_unidex/urdf/065-d_cups.urdf",
            )
            cfg.task.env.render.enable = False
            cfg.task.env.render.appearance_realistic = False

        cfg.task.name = TASK_NAME
        cfg.task_name = TASK_NAME
        cfg.task.env.seYawAngles = cfg.get(
            "se_yaw_angles", [-30.0, -15.0, 0.0, 15.0, 30.0]
        )
        cfg.task.env.seTiltAngles = cfg.get(
            "se_tilt_angles", cfg.get("se_yaw_angles", [0.0, 15.0, 30.0, 45.0])
        )
        cfg.task.env.seTiltAxis = cfg.get("se_tilt_axis", [1.0, 0.0, 0.0])
        cfg.task.env.sePitchAngles = cfg.get(
            "se_pitch_angles", [-20.0, -10.0, 0.0, 10.0, 20.0]
        )
        cfg.task.env.seGuideMode = cfg.get("se_guide_mode", "tilt_pitch")
        cfg.task.env.seLegacyTiltAngles = cfg.get(
            "se_legacy_tilt_angles", [0.0, 15.0, 30.0, 45.0]
        )
        cfg.task.env.seLegacyTiltAxis = cfg.get(
            "se_legacy_tilt_axis", [0.0, 1.0, 0.0]
        )
        explicit_pairs = cfg.get("se_yaw_pitch_pairs", None)
        if explicit_pairs is not None:
            cfg.task.env.seYawPitchPairs = cfg.se_yaw_pitch_pairs
            guide_count = len(cfg.task.env.seYawPitchPairs)
        elif cfg.task.env.seGuideMode == "legacy_tilt":
            guide_count = len(cfg.task.env.seLegacyTiltAngles)
        elif cfg.task.env.seGuideMode == "tilt_pitch":
            guide_count = (
                len(cfg.task.env.seTiltAngles) * len(cfg.task.env.sePitchAngles)
            )
        else:
            guide_count = (
                len(cfg.task.env.seYawAngles) * len(cfg.task.env.sePitchAngles)
            )
        cfg.task.env.seSampling = cfg.get("se_sampling", "random")
        cfg.task.env.worldTiltSampling = cfg.task.env.seSampling
        cfg.task.env.worldTiltAngles = [
            float(index) for index in range(guide_count)
        ]
        cfg.task.env.worldTiltAxis = [0.0, 0.0, 1.0]
        cfg.task.env.enablePointCloud = bool(
            cfg.get("se_enable_pointcloud", True)
        )
        cfg.task.env.observationType = cfg.get(
            "se_observation_type",
            "eefpose+objinitpose+objpcl",
        )
        cfg.task.env.residualTiltModulation = cfg.get(
            "se_residual_modulation", "none"
        )
        cfg.task.env.wristTiltModulation = bool(
            cfg.get("se_wrist_modulation", False)
        )
        cfg.task.env.fingerTiltModulation = bool(
            cfg.get("se_finger_modulation", False)
        )

        cfg.hand.num_obs_dict.seguide = SEResidualGrasp.GUIDE_OBS_DIM
        obs_parts = cfg.task.env.observationType.split("+")
        obs_parts = [part for part in obs_parts if part != "worldtilt"]
        if "seguide" not in obs_parts:
            obs_parts.append("seguide")
        cfg.task.env.observationType = "+".join(obs_parts)

        if not cfg.get("run_name", ""):
            cfg.run_name = _default_run_name(cfg)

        cfg.task.hand_config = cfg.hand


def se_residual_metadata(cfg, env):
    return {
        "mode": env.residual_mode,
        "action_dim": env.residual_action_dim,
        "frame": env.residual_frame,
        "scope": env.residual_scope,
        "pose_composition": env.residual_pose_composition,
        "translation_scale": list(cfg.task.env.residualTranslationScale),
        "rotation_scale_deg": list(cfg.task.env.residualRotationScaleDeg),
        "finger_alpha": cfg.task.env.fingerResidualAlpha,
        "finger_min_ratio": cfg.task.env.fingerResidualMinRatio,
        "finger_max_ratio": cfg.task.env.fingerResidualMaxRatio,
        "guide": "se_wrist",
        "guide_obs": "sin_primary,cos_primary,sin_pitch,cos_pitch",
        "guide_mode": cfg.task.env.seGuideMode,
        "se_yaw_angles": list(cfg.task.env.seYawAngles),
        "se_tilt_angles": list(cfg.task.env.seTiltAngles),
        "se_tilt_axis": list(cfg.task.env.seTiltAxis),
        "se_pitch_angles": list(cfg.task.env.sePitchAngles),
        "se_legacy_tilt_angles": list(cfg.task.env.seLegacyTiltAngles),
        "se_legacy_tilt_axis": list(cfg.task.env.seLegacyTiltAxis),
        "se_sampling": cfg.task.env.seSampling,
        "residual_modulation": cfg.task.env.residualTiltModulation,
        "wrist_modulation": cfg.task.env.wristTiltModulation,
        "finger_modulation": cfg.task.env.fingerTiltModulation,
    }


def _checkpoint_path(cfg):
    return str(cfg.get("residual_checkpoint", "") or cfg.checkpoint or "")


def _embedded_baseline_export_path(payload):
    saved_hash = str(payload.get("baseline_checkpoint_hash", "") or "")
    digest = saved_hash[:16]
    if not digest:
        state = payload.get("baseline_model_state_dict", {})
        hasher = hashlib.sha256()
        for key in sorted(state.keys()):
            tensor = state[key]
            if torch.is_tensor(tensor):
                hasher.update(key.encode("utf-8"))
                hasher.update(tensor.detach().cpu().numpy().tobytes())
        digest = hasher.hexdigest()[:16]
    root = os.path.join(PROJECT_ROOT, "runs_ppo", "_embedded_baselines")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"baseline_{digest}.pt")


def _prepare_embedded_baseline_if_needed(cfg):
    baseline_checkpoint = str(
        cfg.get("baseline_checkpoint", "")
        or cfg.get("se_baseline_checkpoint", "")
        or ""
    )
    if baseline_checkpoint and os.path.isfile(baseline_checkpoint):
        return None

    checkpoint = _checkpoint_path(cfg)
    if not checkpoint or not os.path.isfile(checkpoint):
        return None

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    baseline_state = payload.get("baseline_model_state_dict")
    if not isinstance(baseline_state, dict):
        return None

    export_path = _embedded_baseline_export_path(payload)
    if not os.path.isfile(export_path):
        torch.save(
            {
                "model_state_dict": baseline_state,
                "embedded_from_residual_checkpoint": os.path.abspath(checkpoint),
                "baseline_checkpoint": payload.get("baseline_checkpoint", ""),
                "embedded_original_baseline_hash": payload.get(
                    "baseline_checkpoint_hash", ""
                ),
            },
            export_path,
        )
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.baseline_checkpoint = export_path
        cfg.embedded_baseline_checkpoint_hash = str(
            payload.get("baseline_checkpoint_hash", "") or ""
        )
    print(
        "Using embedded frozen baseline from residual checkpoint: "
        f"{export_path} "
        f"(original_sha256={cfg.embedded_baseline_checkpoint_hash[:12]})",
        flush=True,
    )
    return export_path


def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test
    _prepare_embedded_baseline_if_needed(cfg)
    baseline_checkpoint, residual_checkpoint = get_checkpoint_paths(cfg)

    if not is_testing:
        run_name = cfg.get("run_name", _default_run_name(cfg))
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as file:
            json.dump(OmegaConf.to_container(cfg), file, indent=4)
    else:
        log_dir = None

    if train_param.name != "ppo_onestep":
        raise ValueError("SE residual training supports PPOOneStep only")

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
        raise ValueError("test=True requires +residual_checkpoint=path/model.pt")
    elif not residual_checkpoint:
        runner.evaluate_initial_policy(
            rounds=int(cfg.get("residual_initial_eval_rounds", 1))
        )
    return runner


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    set_np_formatting()
    configure_se_training(cfg)
    log_viewer_device(cfg)
    sanity_eval = bool(cfg.get("se_sanity_eval", False))
    if sanity_eval:
        with open_dict(cfg):
            cfg.residual_initial_eval_rounds = int(
                cfg.get("se_sanity_rounds", 3)
            )
    print(
        "SE residual training: "
        f"guide_mode={cfg.task.env.seGuideMode}, "
        f"yaw={list(cfg.task.env.seYawAngles)}, "
        f"tilt={list(cfg.task.env.seTiltAngles)}, "
        f"pitch={list(cfg.task.env.sePitchAngles)}, "
        f"sampling={cfg.task.env.seSampling}, "
        f"obs={cfg.task.env.observationType}, "
        f"rollouts_per_update={cfg.train.params.nsteps}, "
        f"minibatches={cfg.train.params.nminibatches}, "
        f"mode={cfg.task.env.residualMode}, "
        f"modulation={cfg.task.env.residualTiltModulation}, "
        f"baseline={cfg.get('baseline_checkpoint', '') or cfg.get('se_baseline_checkpoint', '<default>')}",
        flush=True,
    )
    if sanity_eval:
        print(
            "SE sanity eval only: "
            f"rounds={cfg.residual_initial_eval_rounds}, "
            f"obs={cfg.task.env.observationType}, "
            f"vision={cfg.train.params.is_vision}, "
            f"pointcloud={cfg.task.env.enablePointCloud}",
            flush=True,
        )
    isaacgym_task_map[TASK_NAME] = SEResidualGrasp

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
    if sanity_eval:
        print("SE sanity eval finished; skip PPO training.", flush=True)
        return
    runner.run()


if __name__ == "__main__":
    main()
