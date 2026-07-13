#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/DemoGrasp_mine
source demograsp_new_coco_20260627_222202/bin/activate

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m build_Pregrasp_data.yaw_pregrasp.collect.collect_yaw_pregrasp_rollouts \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  headless=true force_render=false num_envs="${NUM_ENVS:-1080}" \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_min10.yaml" \
  checkpoint=runs_ppo/2026-06-22_07-59-10_new_sr_hand_simple_se_tilt_pitch_0-15-30_pitch_0-15-30/model_15000.pt \
  --yaw_angles=0,30,60,90,120,150,180,210,240,270,300,330 \
  --tilt_angles=0,15,30 \
  --pitch_angles=0,15,30 \
  +pregrasp_yaw_frame=pca_short \
  +pregrasp_yaw_sampling=cycle \
  +pregrasp_yaw_canonical_obs=True \
  +pregrasp_yaw_canonical_obs_scope=scene \
  +pregrasp_yaw_direct_base_qpos=True \
  +yaw_pregrasp_collect_max_batches="${BATCHES:-100}" \
  +yaw_pregrasp_collect_overwrite_root=True \
  +yaw_pregrasp_collect_rollout_root=/mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/pca_short_yaw12_tilt3_pitch3_scene_min10 \
  +pregrasp_eval_log_interval=0
