# SE residual grasp training

This experiment replaces the old single-axis tilted trajectory family with a
two-angle object-centered guide.

The sampled discrete training pair is:

- `se_tilt_angles`: old multi-angle single-axis wrist/reference tilt.
- `se_pitch_angles`: elevation/pitch on the object-centered shell.

The PPO observation receives:

```text
seguide = [sin(tilt), cos(tilt), sin(pitch), cos(pitch)]
```

IsaacGym static pose preview:

```bash
python residual_se_grasp/play_traj.py \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  headless=false force_render=true \
  num_envs=20 +play_mode=static \
  +se_guide_mode=tilt_pitch \
  +se_tilt_angles=[0,15,30,45] \
  +se_tilt_axis=[1,0,0] \
  +se_pitch_angles=[-20,-10,0,10,20] \
  +frame=0
```

IsaacGym trajectory playback:

```bash
python residual_se_grasp/play_traj.py \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple \
  headless=false force_render=true \
  num_envs=20 +play_mode=play \
  +se_guide_mode=tilt_pitch \
  +se_tilt_angles=[0,15,30,45] \
  +se_tilt_axis=[1,0,0] \
  +se_pitch_angles=[-20,-10,0,10,20] \
  +start_frame=0 +end_frame=18 \
  +interpolation_steps=20 +frame_hold=1
```

Add success statistics:

```bash
+play_eval_success=True +episodes=3
```

Object motion behavior:

```text
play_mode=static: object is fixed for pose inspection.
play_mode=play: object starts from the same table-center pose, then moves dynamically.
+play_lock_object=True: force-lock the object every step for debugging only.
```

Swap the preview/test object:

```bash
+se_preview_object_asset=union_ycb_unidex/urdf/unidex_core_bottle-b13f6dc78d904e5c30612f5c0ef21eb8_006.urdf
```

Other quick candidates:

```text
union_ycb_unidex/urdf/065-d_cups.urdf
union_ycb_unidex/urdf/unidex_core_mug-642eb7c42ebedabd223d193f5a188983_006.urdf
union_ycb_unidex/urdf/unidex_core_bowl-8d1f575e9223b28b8183a4a81361b94_008.urdf
union_ycb_unidex/urdf/unidex_core_bottle-b13f6dc78d904e5c30612f5c0ef21eb8_006.urdf
```

Start training after the reset distribution looks correct:

```bash
python residual_se_grasp/train_se_reslearn.py \
  task=grasp train=PPOOneStep hand=new_sr_hand_simple headless=true \
  num_envs=3200 +se_sampling=random \
  +se_guide_mode=tilt_pitch \
  +se_tilt_angles=[0,15,30,45] \
  +se_tilt_axis=[1,0,0] \
  +se_pitch_angles=[-20,-10,0,10,20]
```

Use a specific frozen baseline:

```bash
+se_baseline_checkpoint=/path/to/model_0.pt
```

This is an alias for the existing:

```bash
+baseline_checkpoint=/path/to/model_0.pt
```

Training defaults align with the frozen baseline checkpoint:

```text
train.params.is_vision = True
task.env.enablePointCloud = True
task.env.observationType = eefpose+objinitpose+objpcl+seguide
```
