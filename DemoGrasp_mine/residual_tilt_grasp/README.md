# residual_tilt_grasp：sr_shadow_hand 倾斜抓取实验

`residual_tilt_grasp` 是独立的 sr Shadow Hand 实验目录。这个目录后续默认面向 `sr_shadow_hand`，不再以 `shadow_simple` 为主要目标。

与 `shadow_simple` 相比，`sr_shadow_hand` 的手部动作维度不同：

```text
shadow_simple: 18 hand DOFs
sr_shadow_hand: 24 hand DOFs
```

因此旧的 18 维 checkpoint 不能完整作为 sr 策略直接推理使用；当前普通训练脚本只把旧模型的观测层、PointNet/backbone、critic 等可兼容部分作为预训练初始化，24 维动作头会重新初始化。

## 目录状态

本目录已经改为自引用包路径，Python 代码中的模块导入应使用：

```python
residual_tilt_grasp.*
```

不应再依赖复制来源目录。若运行日志中出现来源目录下的 `.py` 文件，说明命令跑错目录，或者外部脚本还在引用旧路径。

## 核心文件

`rotate_play.py`

用于播放模板轨迹并可视化不同桌面倾角。当前默认 `--hand sr_shadow_hand` 相关配置已经包含 forearm-to-palm offset、sr wrist/palm frame 对齐等修正。默认 CPU pipeline；只有显式加 `--gpu_pipeline` 才启用 GPU pipeline。

`tilted_hand_only_grasp.py`

倾斜桌面训练环境。基于原始 hand-only grasp 环境，加入桌面/物体/重力倾斜、桌面坐标系 reward、`worldtilt` observation 等逻辑。

`train_tilted_hand_only.py`

普通 tilted PPO 训练入口。适合从零训练 sr，或用旧 `shadow_simple` checkpoint 初始化可兼容网络层后微调 sr。注意 18 维动作头会跳过并重新初始化为 24 维。

`tilted_ppo.py`

普通 tilted PPO 的 checkpoint 兼容层。支持：

- 旧 observation 扩展到包含 `worldtilt`
- 18 维旧动作头跳过，当前 sr 24 维动作头重新初始化
- 普通 checkpoint resume

`train_tilted_hand_only_reslearn.py`

残差学习入口。冻结 baseline policy，只训练 residual policy。对 sr 使用时需要特别注意：frozen baseline 必须能输出 sr 的 24 维动作，不能直接用 18 维 `shadow_simple` baseline。

`residual_tilted_grasp.py`

残差环境逻辑，负责把 residual action 解码为 wrist / finger residual，并叠加到 frozen baseline 和参考轨迹上。

`residual_ppo.py`

残差 PPO runner，管理 frozen baseline、residual policy、per-angle advantage、residual checkpoint 元数据等。

`residual_actor_critic.py`

残差策略网络定义，默认 residual 输出从接近 0 开始，避免一开始破坏 baseline。

`assets/tilted_table.urdf`

倾斜桌面 asset。

`说明.txt`

个人常用命令记录，可作为临时命令草稿；正式说明以本 README 为准。

## 轨迹播放

可视化 sr 模板轨迹：
CUDA_VISIBLE_DEVICES=0 python residual_tilt_grasp/play_tra_rot.py \
  --hand sr_shadow_hand \
  --num_envs 16 \
  --traj_rot 0 15 20 30 \
  --rot_axis 1 0 0


## 普通 tilted PPO 训练

### 从零训练 sr

```bash
CUDA_VISIBLE_DEVICES=7 python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=3200 headless=True \
  '+tilt_angles=[0]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=random' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

### 用旧 18 维 hand checkpoint 初始化 sr

这会加载兼容层，并重新初始化 sr 的 24 维动作头：

```bash
CUDA_VISIBLE_DEVICES=7 python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=3200 headless=True \
  checkpoint=runs_ppo/tilted_hand_only_grasp_2026-06-17_13-19-55/model_300.pt \
  '+tilt_angles=[0]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=random' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

预期日志会类似：

```text
Expanded actor_mean.0.weight ... with zero world-tilt columns
Expanded critic.0.weight ... with zero world-tilt columns
Reinitialized action head entries for current hand: ...
```

### 指定初始姿势和粗抓取姿势

sr 的 24 维手关节顺序：

```text
WRJ2, WRJ1,
FFJ4, FFJ3, FFJ2, FFJ1,
MFJ4, MFJ3, MFJ2, MFJ1,
RFJ4, RFJ3, RFJ2, RFJ1,
LFJ5, LFJ4, LFJ3, LFJ2, LFJ1,
THJ5, THJ4, THJ3, THJ2, THJ1
```

示例：

```bash
CUDA_VISIBLE_DEVICES=3 python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=7000 headless=True \
  '+initial_base_dof_pos=[0.5,-0.1,0.4,0,1.57,0]' \
  '+initial_hand_dof_pos=[0,0,-0.1,0.3,0,0,0,-0.2,0.3,0.3,0,0,0.3,0.3,0,-0.1,0.3,0,0,0,1.2,0,-0.2,0]' \
  '+coarse_grasp_hand_dof_pos=[0,0,-0.3,0.42,0.89,0.89,0.27,-0.34,0.45,0.45,0.72,-0.14,0.61,0.61,0.78,-0.03,0.67,0.72,0.72,0.48,1.08,-0.12,0.19,0.31]' \
  '+grasp_delta_scale=0.1' \
  '+tilt_angles=[0]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=random' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

### 多角度训练

```bash
CUDA_VISIBLE_DEVICES=7 python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=7000 headless=True \
  checkpoint=runs_ppo/hand_only_grasp_2026-06-05_14-52-57/model_4000.pt \
  '+tilt_angles=[0,15,30,45]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=random' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

`tilt_sampling=random` 表示每个 rollout 从角度列表随机采样；调试时可以用：

```bash
'+tilt_sampling=cycle'
```

## 测试 / 可视化训练模型

固定 30 度测试：

```bash
CUDA_VISIBLE_DEVICES=7 python -u residual_tilt_grasp/train_tilted_hand_only.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=64 \
  test=True headless=False \
  checkpoint=runs_ppo/tilted_hand_only_grasp_xxx/model_xxx.pt \
  '+tilt_angles=[15]' \
  '+tilt_axis=[0,1,0]' \
  '+tilt_sampling=cycle' \
  task.env.asset.multiObjectList="union_ycb_unidex/example.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

如果使用 `headless=False` 且 `CUDA_VISIBLE_DEVICES` 指向非 0 GPU，脚本会自动重启并同步 CUDA / Vulkan GPU 编号，避免 viewer segfault。

## 残差学习注意事项

残差学习命令入口：

```bash
CUDA_VISIBLE_DEVICES=6 python -u residual_tilt_grasp/train_tilted_hand_only_reslearn.py \
  train=PPOOneStep hand=sr_shadow_hand num_envs=3200 headless=True \
  '+baseline_checkpoint=runs_ppo/tilted_hand_only_grasp_2026-06-17_13-19-55/model_200.pt' \
  '+tilt_angles=[0,15,30]' \
  '+tilt_axis=[1,0,0]' \
  '+tilt_sampling=random' \
  '+residual_mode=hybrid' \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.enablePointCloud=True \
  train.params.is_vision=True
```

重要限制：

- residual baseline 必须能输出 sr 的 24 维 hand action。
- 不要直接把 `shadow_simple` 的 18 维 checkpoint 当 frozen baseline 给 sr residual 使用。
- 如果只有旧 18 维 checkpoint，应先通过普通 tilted PPO 训练出一个 sr baseline，再做 residual learning。

## sr hand 对齐问题

sr 的 base 接到 `forearm`，再经过 `WRJ2/WRJ1` 到 `palm`；`shadow_simple` 的 base/palm frame 定义不同。因此 sr 相关 reset、reaching、tracking 必须明确 frame：

- 位置上需要考虑 `palm_offset_from_forearm=(0.0, -0.01, 0.247)`。
- 姿态上不要把 `wrist_quat_offset` 当成通用 reset 修正。
- 默认初始化更适合从 reference frame 0 / canonical palm pose 生成，而不是直接复用 `shadow_simple` 的 hard-coded default pose。

详细记录见：

```text
residual_tilt_grasp/6.17sr的手和hand_simple手的区别.md
```
