#!/usr/bin/env bash
set -euo pipefail

cd /mnt/AAAI2027_grasp/DemoGrasp_mine
source demograsp_new_coco_20260627_222202/bin/activate

python3 -m build_Pregrasp_data.yaw_pregrasp.aggregate.build_yaw_field_store \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/pca_short_yaw12_tilt3_pitch3_scene \
  --field-root /mnt/AAAI2027_grasp/PregraspPrior/data/fields/pca_short_yaw12_tilt3_pitch3_scene

