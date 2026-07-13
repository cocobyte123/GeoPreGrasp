#!/usr/bin/env bash
set -euo pipefail

python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep \
  hand=shadow_simple \
  num_envs=7000 \
  headless=True \
  '+tilt_angles=[0,15,30,45]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=random' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.episodeLength=40 \
  task.env.enablePointCloud=True \
  task.env.randomizeGraspPoseRange=1 \
  train.params.is_vision=True
