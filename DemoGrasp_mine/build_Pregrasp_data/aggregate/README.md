# aggregate

这里放 PPO rollout 到 PregraspPrior field store 的后处理脚本。

当前脚本：

```bash
python -m build_Pregrasp_data.aggregate.inspect_rollout_batch <batch.npz>

python -m build_Pregrasp_data.aggregate.build_field_store_from_rollouts \
  --rollout-root /mnt/AAAI2027_grasp/PregraspPrior/data/rollouts/ppo_random_mtp \
  --output-root /mnt/AAAI2027_grasp/PregraspPrior/data/fields/ppo_random_mtp \
  --overwrite
```
