#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/DemoGrasp_mine
source demograsp_new_coco_20260627_222202/bin/activate

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 pregrasp_eval/eval_yaw_field_SE_residual.py \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  test=True headless=true force_render=true \
  num_envs="${NUM_ENVS:-3200}" \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  checkpoint=runs_ppo/2026-06-22_07-59-10_new_sr_hand_simple_se_tilt_pitch_0-15-30_pitch_0-15-30/model_15000.pt \
  --yaw_angles=0,30,60,90,120,150,180,210,240,270,300,330 \
  --tilt_angles=0 \
  --pitch_angles=0 \
  +pregrasp_yaw_field_root=/mnt/AAAI2027_grasp/PregraspPrior/data/fields/pca_short_yaw12_tilt0_pitch0_scene \
  +pregrasp_yaw_frame=pca_short \
  +pregrasp_yaw_sampling=field_top1 \
  +pregrasp_yaw_canonical_obs=True \
  +pregrasp_yaw_canonical_obs_scope=scene \
  +pregrasp_yaw_direct_base_qpos=True \
  +pregrasp_eval_log_interval=0

