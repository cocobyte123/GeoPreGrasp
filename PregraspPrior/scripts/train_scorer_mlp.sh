#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/PregraspPrior

PYTHONPATH=/mnt/AAAI2027_grasp/PregraspPrior/src \
python3 -m oc_pregrasp.train.train_scorer \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/pca_short_yaw12_tilt0_pitch0_scene \
  --output-dir /mnt/AAAI2027_grasp/PregraspPrior/data/models/scorer_mlp_pca_short_yaw12_tilt0_pitch0_scene \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --lr "${LR:-0.001}" \
  --device "${DEVICE:-cuda}" \
  ${RESUME_CHECKPOINT:+--resume-checkpoint "$RESUME_CHECKPOINT"}
