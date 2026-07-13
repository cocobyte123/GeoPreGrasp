# Yaw Pregrasp Dataset

This directory contains the IsaacGym-facing yaw-expanded pregrasp dataset
pipeline.  The current data root lives outside this repo at
`/mnt/AAAI2027_grasp/PregraspPrior`.

The old M-point field mapped a sampled spatial bin directly to a real hand
initialization.  This pipeline keeps the PPO policy in its trained canonical
input domain and treats yaw as an external geometric expansion:

```text
reference_yaw: pca_short/object/absolute frame selected by pregrasp_yaw_frame
real execution: (reference_yaw + relative yaw) * tilt_pitch hand/reference transform
policy input: scene canonicalized by -real_yaw for the stable object-centric variant
label: success probability for (relative yaw, tilt, pitch)
```

## Dataset Target

Each rollout sample records:

```text
object_id
object_pose_world
yaw_deg                 # candidate yaw relative to the selected reference
world_yaw_deg           # executed world yaw / real_yaw
object_yaw_deg          # z-yaw from the object quaternion
reference_yaw_deg       # selected reference yaw: absolute/object/pca_long/pca_short
tilt_deg
pitch_deg
success
hit_table
min_clearance
real_palm_pose_world
canonical_palm_pose_world
```

Aggregation produces per-object fields:

```text
success_rate[yaw, tilt, pitch]
success_count[yaw, tilt, pitch]
trial_count[yaw, tilt, pitch]
```

## Planned Tools

```text
collect/collect_yaw_pregrasp_rollouts.py
aggregate/build_yaw_field_store.py
```

The collector should reuse the same geometry as
`pregrasp_eval/eval_yaw_SE_residual.py` and
`pregrasp_eval/eval_yaw_field_SE_residual.py`:

```text
pregrasp_yaw_frame=pca_short  # absolute, object, pca_long, or pca_short
pregrasp_yaw_canonical_obs=True
pregrasp_yaw_canonical_obs_scope=scene
pregrasp_yaw_direct_base_qpos=True
```

For the current object-centric labels, `yaw_deg` is relative to the selected
geometry frame and `world_yaw_deg = reference_yaw_deg + yaw_deg`.

## Minimal Collection Check

Run from `DemoGrasp_mine` after activating the IsaacGym Python 3.8 env:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m build_Pregrasp_data.yaw_pregrasp.collect.collect_yaw_pregrasp_rollouts \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  headless=true force_render=false num_envs=108 \
  task.env.asset.multiObject=false \
  +task.env.asset.objectAssetFile=union_ycb_unidex/urdf/065-f_cups.urdf \
  task.env.resetRandomRot=fixed \
  task.env.resetPositionRange='[[0.5,0.5],[-0.1,-0.1],[0.11,0.11]]' \
  checkpoint=runs_ppo/2026-06-22_07-59-10_new_sr_hand_simple_se_tilt_pitch_0-15-30_pitch_0-15-30/model_15000.pt \
  --yaw_angles=0,30,60,90,120,150,180,210,240,270,300,330 \
  --tilt_angles=0,15,30 \
  --pitch_angles=0,15,30 \
  +pregrasp_yaw_frame=pca_short \
  +pregrasp_yaw_sampling=cycle \
  +pregrasp_yaw_canonical_obs=True \
  +pregrasp_yaw_canonical_obs_scope=scene \
  +yaw_pregrasp_collect_max_batches=1 \
  +yaw_pregrasp_collect_overwrite_root=True \
  +yaw_pregrasp_collect_rollout_root=/mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/yaw_pregrasp_debug \
  +pregrasp_eval_log_interval=0
```

Aggregate the collected labels:

```bash
python3 -m build_Pregrasp_data.yaw_pregrasp.aggregate.build_yaw_field_store \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/yaw_pregrasp_debug \
  --field-root /mnt/AAAI2027_grasp/PregraspPrior/data/fields/yaw_pregrasp_debug
```

Evaluate by selecting yaw/tilt/pitch from the aggregated field:

```bash
CUDA_VISIBLE_DEVICES=0 python3 pregrasp_eval/eval_yaw_field_SE_residual.py \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  test=True headless=false force_render=true \
  num_envs=20 \
  task.env.asset.multiObject=false \
  +task.env.asset.objectAssetFile=union_ycb_unidex/urdf/065-f_cups.urdf \
  task.env.resetRandomRot=fixed \
  task.env.resetPositionRange='[[0.5,0.5],[-0.1,-0.1],[0.11,0.11]]' \
  checkpoint=runs_ppo/2026-06-22_07-59-10_new_sr_hand_simple_se_tilt_pitch_0-15-30_pitch_0-15-30/model_15000.pt \
  --yaw_angles=0,30,60,90,120,150,180,210,240,270,300,330 \
  --tilt_angles=0,15,30 \
  --pitch_angles=0,15,30 \
  +pregrasp_yaw_field_root=/mnt/AAAI2027_grasp/PregraspPrior/data/fields/yaw_pregrasp_debug \
  +pregrasp_yaw_frame=pca_short \
  +pregrasp_yaw_sampling=field_top1 \
  +pregrasp_yaw_canonical_obs=True \
  +pregrasp_yaw_canonical_obs_scope=scene \
  +pregrasp_yaw_direct_base_qpos=True \
  +pregrasp_eval_log_interval=0
```
