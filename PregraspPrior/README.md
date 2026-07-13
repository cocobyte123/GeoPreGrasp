# PregraspPrior

This directory is the pregrasp-prior workspace.  `DemoGrasp_mine` still runs
IsaacGym collection/evaluation, while this directory owns the data roots,
configs, scripts, and learning code.

## Current Definition

The working pregrasp label is:

```text
P(success | object geometry, object pose, relative yaw, tilt, pitch)
```

The current stable setting is:

```text
reference_yaw = pca_short(object point cloud projected to table xy)
real_yaw = reference_yaw + relative_yaw
PPO observation frame = scene canonicalized by -real_yaw
```

In practice this means the real hand can rotate around the object, but PPO sees
a normalized scene close to its trained input distribution.

## Data Roots

```text
data/rollouts/   raw rollout labels, one batch npz per object
data/fields/     aggregated per-object yaw/tilt/pitch success fields
data/models/     learned pregrasp prior checkpoints
data/evals/      evaluation summaries
```

The legacy standalone field workspace is no longer the default path for the
yaw-pregrasp collector and aggregator.

## First Loop

Collect labels:

```bash
BATCHES=50 CUDA_VISIBLE_DEVICES=0 bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/collect_pca_short_yaw12_tilt0_pitch0_scene.sh
```

Aggregate fields:

```bash
bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/aggregate_pca_short_yaw12_tilt0_pitch0_scene.sh
```

Evaluate field top-1:

```bash
CUDA_VISIBLE_DEVICES=0 bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/eval_field_top1_pca_short_yaw12_tilt0_pitch0_scene.sh
```

Train the first learnable scorer baseline:

```bash
cd /mnt/AAAI2027_grasp/DemoGrasp_mine
source demograsp_new_coco_20260627_222202/bin/activate
bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/train_scorer_mlp.sh
```

Evaluate the trained scorer offline against rollout labels:

```bash
bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/eval_scorer_mlp_offline.sh
```

Evaluate the trained scorer inside IsaacGym:

```bash
CUDA_VISIBLE_DEVICES=0 bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/eval_scorer_mlp_gym_pca_short_yaw12_tilt0_pitch0_scene.sh
```

Compare base yaw0 against the learned scorer in one IsaacGym run:

```bash
CUDA_VISIBLE_DEVICES=0 bash /mnt/AAAI2027_grasp/PregraspPrior/scripts/compare_base_vs_scorer_gym_pca_short_yaw12_tilt0_pitch0_scene.sh
```

## Expanded Candidate Grid

For the larger 108-candidate grid:

```text
yaw:   0,30,...,330
tilt:  0,15,30
pitch: 0,15,30
```

use the `*_yaw12_tilt3_pitch3_*` scripts.  This is the setting that should
better expose whether the learned scorer can choose object-specific pregrasp
poses beyond the simple yaw-only case.

## Learning Baselines

The first model is a compact PointNet scorer:

```text
object point cloud -> PointNet max pool
candidate yaw/tilt/pitch -> sin/cos angle feature
object pose -> pose feature
concat -> MLP -> success logit
```

This is not the final matching-flow method; it is the clean baseline to prove
that the collected labels are learnable and useful.
