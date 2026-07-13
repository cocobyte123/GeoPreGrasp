'''
DemoGrasp: Play back the demonstration trajectory (no RL policy).

Usage examples:
  # Inspire hand (default), 16 envs:
  python play_traj.py num_envs=16 task.env.asset.multiObjectList="union_ycb_unidex/example.yaml"

  # Shadow hand, 16 envs:
  python play_traj.py num_envs=16 hand=shadow_simple \
      task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl \
      task.env.asset.multiObjectList="union_ycb_unidex/example.yaml"

  # UR5 + Allegro hand, 8 envs, headless:
  python play_traj.py num_envs=8 headless=True hand=ur5_allegro \
      task.env.trackingReferenceFile=tasks/grasp_ref_allegro.pkl \
      task.env.asset.multiObjectList="union_ycb_unidex/example.yaml"

  # Show all available hands:
  python play_traj.py --help
'''

import hydra
from omegaconf import DictConfig, OmegaConf

# NOTE: isaacgym MUST be imported before torch.
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
import tasks
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import torch

@hydra.main(version_base="1.3", config_path="./tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    set_np_formatting()
    set_seed(cfg.seed)

    # Disable all randomization for clean demo replay
    OmegaConf.update(cfg, "task.env.randomizeTrackingReference", False, merge=False)
    OmegaConf.update(cfg, "task.env.randomizeGraspPose", False, merge=False)
    OmegaConf.update(cfg, "task.env.enablePointCloud", False, merge=False)
    OmegaConf.update(cfg, "task.env.limitControlError", False, merge=False)

    # Create the IsaacGym environment
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

    # Initial reset
    env.reset_idx(torch.arange(env.num_envs))

    print(f"\n{'='*60}")
    print(f"  Hand:          {env.hand_name}")
    print(f"  Arm controller: {env.arm_controller}")
    print(f"  Num envs:       {env.num_envs}")
    print(f"  Episode length: {env.max_episode_length}")
    print(f"  Ref file:       {cfg.task.env.trackingReferenceFile}")
    print(f"{'='*60}\n")

    # Main demo replay loop
    step_count = 0
    while True:
        # Get action from the tracking reference (no policy involved)
        action = env.compute_reference_actions()

        # Step the physics simulation
        obs, reward, reset, extras = env.step(action)
        step_count += 1

        # Print success rate at the end of each episode
        if step_count % env.max_episode_length == 0:
            success_rate = env.current_successes.mean().item()
            print(f"  Episode {step_count // env.max_episode_length:4d}  |  "
                  f"success rate: {success_rate:.3f}")

        # Reset environments that are done
        env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            env.reset_idx(env_ids)


if __name__ == "__main__":
    main()
