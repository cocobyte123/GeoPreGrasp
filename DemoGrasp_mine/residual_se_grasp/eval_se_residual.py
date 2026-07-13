"""Standalone evaluation entry for the original SE residual PPO model.

This script intentionally does not use yaw-field or pregrasp dataset logic.  It
is a thin test-only wrapper around ``train_se_reslearn.py`` so it can be used as
the clean baseline for later yaw/field comparisons.
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
        "--tilt_angles": "+se_tilt_angles=",
        "--pitch_angles": "+se_pitch_angles=",
        "--se_sampling": "+se_sampling=",
    }
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        matched = False
        for cli_name, hydra_name in aliases.items():
            if arg.startswith(cli_name + "="):
                value = arg.split("=", 1)[1]
                if cli_name in ("--tilt_angles", "--pitch_angles"):
                    value = _convert_csv_arg_to_hydra_list(value)
                converted.append(hydra_name + value)
                matched = True
                break
            if arg == cli_name and index + 1 < len(argv):
                value = argv[index + 1]
                if cli_name in ("--tilt_angles", "--pitch_angles"):
                    value = _convert_csv_arg_to_hydra_list(value)
                converted.append(hydra_name + value)
                skip_next = True
                matched = True
                break
        if not matched:
            converted.append(arg)
    sys.argv[:] = converted


def hydra_main():
    _add_project_paths()

    import gym
    import hydra
    import isaacgymenvs
    from isaacgym import gymapi  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
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

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        with open_dict(cfg):
            cfg.test = True
            if "train" in cfg and cfg.train is not None:
                cfg.train.params.test = True
            if cfg.get("checkpoint", "") and not cfg.get("residual_checkpoint", ""):
                cfg.residual_checkpoint = cfg.checkpoint

        configure_se_training(cfg)
        with open_dict(cfg):
            cfg.test = True
            cfg.train.params.test = True

        log_viewer_device(cfg)
        print(
            "Standalone SE residual eval: "
            f"guide_mode={cfg.task.env.seGuideMode}, "
            f"tilt={list(cfg.task.env.seTiltAngles)}, "
            f"pitch={list(cfg.task.env.sePitchAngles)}, "
            f"sampling={cfg.task.env.seSampling}, "
            f"obs={cfg.task.env.observationType}, "
            f"num_envs={cfg.task.env.numEnvs}, "
            f"multi_object={cfg.task.env.asset.multiObject}, "
            f"multi_object_list={cfg.task.env.asset.multiObjectList}, "
            f"reset_random_rot={cfg.task.env.resetRandomRot}, "
            f"reset_position_range={cfg.task.env.resetPositionRange}, "
            f"checkpoint={cfg.get('residual_checkpoint', '')}",
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
                f"videos/{TASK_NAME}_standalone_eval",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner = build_runner(cfg, env)
        runner.run()

    _main()


if __name__ == "__main__":
    _normalize_cli_aliases(sys.argv)
    hydra_main()
