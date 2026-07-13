# Hand-Only Shadow Hand RL

This directory contains an isolated training experiment. It does not modify the
original task registry or training entry point.

The six virtual wrist joints follow `tasks/grasp_ref_shadow.pkl`. PPO outputs
only 18 values, one adjustment for each active Shadow Hand joint. Physics,
objects, observations, reward, reset behavior, and PPO implementation are
inherited from the original `Grasp` task.

## Train

```bash
bash grasp_base/train.sh
```

For a smaller validation run:

```bash
python -u grasp_base/train_hand_only.py \
  train=PPOOneStep hand=shadow_simple num_envs=16 headless=True \
  task.env.asset.multiObjectList="union_ycb_unidex/example.yaml" \
  task.env.observationType="eefpose+objinitpose" \
  task.env.enablePointCloud=False \
  train.params.is_vision=False \
  train.params.max_iterations=10
```

## Test A Checkpoint

```bash
python -u grasp_base/train_hand_only.py \
  train=PPOOneStep hand=shadow_simple num_envs=16 test=True headless=False \
  checkpoint=/path/to/model.pt \
  task.env.asset.multiObjectList="union_ycb_unidex/example.yaml" \
  task.env.observationType="eefpose+objinitpose" \
  task.env.enablePointCloud=False \
  train.params.is_vision=False
```

Useful experiment overrides:

- `task.env.randomizeGraspPoseRange=0.3`: reduce policy adjustment range.
- `task.env.observationType=...`: use any observation combination supported by
  the original `Grasp` environment.
- `task.env.enablePointCloud=True` with `train.params.is_vision=True`: enable
  the original point-cloud policy path.
