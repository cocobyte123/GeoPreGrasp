#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/PregraspPrior
source /mnt/AAAI2027_grasp/DemoGrasp_mine/demograsp_new_coco_20260627_222202/bin/activate

PYTHONPATH=/mnt/AAAI2027_grasp/PregraspPrior/src \
python3 -m oc_pregrasp.eval.eval_scorer_offline \
  --checkpoint /mnt/AAAI2027_grasp/PregraspPrior/data/models/scorer_mlp_geom_pca_short_yaw12_tilt3_pitch3_scene_min10/best.pt \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/pca_short_yaw12_tilt3_pitch3_scene_min10 \
  --output-json /mnt/AAAI2027_grasp/PregraspPrior/data/evals/scorer_mlp_geom_pca_short_yaw12_tilt3_pitch3_scene_min10_offline.json \
  --batch-size "${BATCH_SIZE:-1024}" \
  --device "${DEVICE:-cuda}"
