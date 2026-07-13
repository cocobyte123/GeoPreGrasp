# collect

This directory contains rollout collectors for object-level PregraspPrior data.
The random M/T/P collector does not read any legacy DGA bridge scenes.
It uses normal DemoGrasp random object resets, samples object-frame
PregraspPrior bins, executes the SE residual PPO policy, and writes rollout
`.npz` shards.

Example smoke command:

```bash
cd /mnt/AAAI2027_grasp/DemoGrasp_mine

python -m build_Pregrasp_data.collect.collect_random_pregrasp_rollouts \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  headless=true num_envs=64 \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  +residual_checkpoint=/path/to/model.pt \
  +pregrasp_collect_max_batches=1 \
  +pregrasp_collect_mtp_mode=cycle \
  +pregrasp_collect_rollout_root=/mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/ppo_random_mtp_smoke
```

Aggregate exported rollouts into a PregraspPrior field store:

```bash
cd /mnt/AAAI2027_grasp/DemoGrasp_mine

python -m build_Pregrasp_data.aggregate.build_field_store_from_rollouts \
  --pregrasp-root /mnt/AAAI2027_grasp/PregraspPrior \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/ppo_random_mtp_smoke \
  --output-root /mnt/AAAI2027_grasp/PregraspPrior/data/fields/ppo_random_mtp_smoke \
  --overwrite
```
