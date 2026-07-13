#!/usr/bin/env bash
set -euo pipefail

python -u grasp_base/train_hand_only.py \
  train=PPOOneStep \
  hand=shadow_simple \
  num_envs=7000 \
  headless=True \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.episodeLength=40 \
  task.env.enablePointCloud=True \
  task.env.randomizeGraspPoseRange=1 \
  train.params.is_vision=True

