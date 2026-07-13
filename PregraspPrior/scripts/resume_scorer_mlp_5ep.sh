#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/PregraspPrior

EPOCHS="${EPOCHS:-5}" \
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/AAAI2027_grasp/PregraspPrior/data/models/scorer_mlp_pca_short_yaw12_tilt0_pitch0_scene/best.pt}" \
bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/train_scorer_mlp.sh

