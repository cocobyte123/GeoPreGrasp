#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/PregraspPrior
source /mnt/AAAI2027_grasp/DemoGrasp_mine/demograsp_new_coco_20260627_222202/bin/activate

PYTHONPATH=/mnt/AAAI2027_grasp/PregraspPrior/src \
python3 -m oc_pregrasp.train.train_scorer \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/pca_short_yaw12_tilt3_pitch3_scene_min10 \
  --output-dir /mnt/AAAI2027_grasp/PregraspPrior/data/models/scorer_mlp_geom_pca_short_yaw12_tilt3_pitch3_scene_min10 \
  --feature-mode pca_short \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --lr "${LR:-0.001}" \
  --device "${DEVICE:-cuda}" \
  ${RESUME_CHECKPOINT:+--resume-checkpoint "$RESUME_CHECKPOINT"}
