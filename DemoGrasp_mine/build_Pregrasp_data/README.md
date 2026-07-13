# build_Pregrasp_data

这个目录负责把 PPO 验证过程中的 rollout 结果制作成 PregraspPrior 的训练数据集。

核心目标是：

```text
固定物体 PPO 验证 batch
  -> 记录每个 env 的初始手势、物体桌面位姿、最终成功/失败
  -> 把初始掌心位姿映射到 M/T/P 球壳网格
  -> 统计 attempt_count 和 success_count
  -> 生成 target_field[M,T,P]
  -> 保存为 PregraspPrior 可训练的 field store
```

这里的数据不是传统抓取数据集标签，而是 **PPO 结果监督的预抓取概率标签**。

也就是说，`target_field[M,T,P]` 的含义是：

> 在当前 PPO 策略、当前物体、当前桌面场景分布下，某个初始预抓取位置和姿态最终抓取成功的概率。

## 路径约定

本文里：

```text
<DemoGrasp>      = /mnt/AAAI2027_grasp/DemoGrasp_mine
<PregraspPrior> = /mnt/AAAI2027_grasp/PregraspPrior
```

`build_Pregrasp_data` 位于 DemoGrasp 侧：

```text
<DemoGrasp>/build_Pregrasp_data/
```

最终生成的数据集放到 PregraspPrior 侧：

```text
<PregraspPrior>/data/
```

## 目录职责

这个目录负责：

- 定义 PPO rollout 导出的原始数据格式。
- 把 PPO rollout 记录聚合成 PregraspPrior 的 field store。
- 同时保留成功样本和失败样本，形成成功率标签。
- 保持输出格式和 `oc_pregrasp.field.field_store` 兼容。

这个目录不负责：

- 不训练 PPO。
- 不修改 PPO 策略网络输入。
- 不直接运行 IsaacGym 环境。
- 不训练最终的 PregraspPrior 神经网络。

## 数据构建思路

PPO 验证时，一个 batch 固定一个物体，例如：

```text
object_name = 002_master_chef_can
num_envs = 3200
```

每个 env 都有：

- 同一个物体类别。
- 一个桌面上的物体初始位姿。
- 一个初始预抓取手势。
- rollout 后的成功/失败结果。

这个 batch 跑完后，就可以导出为一个固定数据 shard，不需要继续维护这些环境。

对每个 env 样本：

1. 读取初始掌心位置 `initial_palm_center_world`。
2. 读取初始掌心朝向 `initial_palm_toward_world`。
3. 读取物体位姿 `object_pose_world`。
4. 把掌心位置和方向从 world frame 转到 object frame。
5. 根据掌心位置方向确定球壳点 `M`。
6. 根据手背方向相对 `M` 的局部角度确定 `T/P`。
7. 根据最终是否成功累计统计量：

```text
attempt_count[M,T,P] += 1
success_count[M,T,P] += success
```

最后得到：

```text
target_field[M,T,P] = success_count[M,T,P] / attempt_count[M,T,P]
confidence[M,T,P] = normalized attempt_count[M,T,P]
```

失败样本不能丢。失败样本决定某个 bin 的成功率低，而不是简单地没有出现过。

## 推荐目录结构

建议后续组织为：

```text
build_Pregrasp_data/
  README.md

  collect/
    README.md
    collect_random_pregrasp_rollouts.py

  aggregate/
    README.md
    build_field_store_from_rollouts.py

  schemas/
    rollout_record_schema.md
    field_dataset_schema.md
```

其中：

- `collect/`：面向 PregraspPrior 数据集制作的 PPO rollout 采集入口。当前主入口随机初始化物体位姿，并采样 object-frame M/T/P bin。
- `export/`：PPO 侧导出 rollout 原始记录。当前保留 bridge/eval 场景导出包装器。
- `aggregate/`：PregraspPrior 侧把 rollout 聚合成热图标签。
- `schemas/`：记录数据格式，避免后续跨项目忘记字段含义。

生成的数据不要放在 `build_Pregrasp_data/` 下面，也不要放在 DemoGrasp 根目录的 `data/` 下面；最终数据集统一写到 PregraspPrior 项目的 `data/` 下面。

推荐：

```text
<PregraspPrior>/data/rollouts/ppo_random_mtp/
  <object_name>/
    manifest.json
    batch_000000.npz
    batch_000001.npz

<PregraspPrior>/data/fields/ppo_random_mtp/
  manifest.json
  fields/
    <object_key>.npz
```

含义：

- `PregraspPrior/data/rollouts/ppo_random_mtp/`：PPO 原始 rollout 日志。
- `PregraspPrior/data/fields/ppo_random_mtp`：聚合后的 PregraspPrior 训练标签。

## PPO Rollout 原始记录格式

每个 batch 文件建议只对应一个物体。

示例：

```text
<PregraspPrior>/data/rollouts/ppo_random_mtp/002_master_chef_can/batch_000000.npz
```

必需字段：

```text
schema: str = "PregraspPPORollout/v1"
object_name: str
sample_id: str

object_pose_world: float32[B,7]
  # 每个 env 的物体位姿，格式为 xyz + quat_xyzw

table_height: float32[B]

initial_qpos33: float32[B,33]
  # PPO 初始化时使用的手姿态

initial_palm_center_world: float32[B,3]
initial_palm_toward_world: float32[B,3]

success: bool[B]
```

建议额外保存：

```text
object_point_cloud_world: float32[B,N,3] or float32[N,3]
object_point_cloud_center_world: float32[B,3]

candidate_index: int64[B]
field_mtp_index: int64[B,3]
candidate_score: float32[B]

episode_reward: float32[B]
failure_code: int64[B]
```

说明：

- 如果 `field_mtp_index` 已知，后处理可以直接累计到对应 bin。
- 如果 `field_mtp_index` 不存在，就根据掌心位置和掌心朝向重新映射。
- `episode_reward` 和 `failure_code` 第一版不是必须，但后续分析失败原因会有用。

## 输出 Field Store 格式

聚合后的结果应该兼容现有 field store：

```text
<PregraspPrior>/data/fields/ppo_random_mtp/
  manifest.json
  fields/<object_key>.npz
```

每个 `.npz` 对应一个物体，建议包含：

```text
schema: str
object_key: str
object_name: str
aliases: object array

points: float32[N,3]

target_field: float32[M,T,P]
confidence: float32[M,T,P]

attempt_count: float32[M,T,P]
success_count: float32[M,T,P]

sphere_dirs: float32[M,3]
sphere_faces: int64[F,3]
tilt_angles: float32[T]
pitch_angles: float32[P]

template_kind: str
subdivision: int64

rollout_count: int64
env_count: int64
success_total: int64
attempt_total: int64
```

其中：

- `target_field`：训练标签，表示成功率。
- `confidence`：该 bin 的数据置信度，通常来自 `attempt_count`。
- `attempt_count`：该 bin 被尝试的次数。
- `success_count`：该 bin 成功的次数。

保留 `attempt_count/success_count` 很重要。这样后续可以重新归一化、过滤低样本 bin、或者换损失权重，而不需要重新跑 PPO。

## M/T/P 映射规则

对每个 env：

```text
object_R_world = quat_to_matrix(object_pose_world.quat_xyzw)
object_t_world = object_pose_world.xyz

palm_center_obj =
    object_R_world.T @ (initial_palm_center_world - object_t_world)

palm_toward_obj =
    object_R_world.T @ initial_palm_toward_world
```

然后：

```text
position_direction = normalize(palm_center_obj)
hand_back_direction = -normalize(palm_toward_obj)
```

含义：

- `position_direction` 决定球壳上的位置 bin `M`。
- `hand_back_direction` 决定该位置下的局部姿态 bin `T/P`。

第一版建议使用硬分配：

```text
m = nearest_sphere_dir(position_direction)
t, p = nearest_local_angle_bins(hand_back_direction relative to sphere_dirs[m])

attempt_count[m,t,p] += 1
success_count[m,t,p] += success
```

后续可以升级为软分配：

- `M` 对最近点和一环邻居做角度加权。
- `T/P` 对角度 bin 做 Gaussian 权重。

第一版先硬分配，便于检查数据是否正确。

## 负样本处理

失败样本必须参与统计。

正确做法：

```text
attempt_count[m,t,p] += 1
success_count[m,t,p] += 0 or 1
target_field = success_count / attempt_count
```

错误做法：

```text
只累计成功样本
```

只累计成功样本会让热图变成“成功样本出现频率”，而不是“该初始手势的成功概率”。这会混淆没有尝试过的区域和尝试后失败的区域。

## Batch 采集策略

推荐采集方式：

```text
for object_name in object_list:
    创建 3200 个 env，全部使用同一个 object_name
    每个 env 随机一个桌面物体位姿和一个初始手势
    运行一次 PPO 验证 rollout
    导出 batch_XXXXXX.npz
    重置后换下一个物体
```

一个 batch 文件就是：

```text
一个物体 + B 个 env 样本
```

如果一个物体需要更多数据，就继续追加：

```text
<PregraspPrior>/data/rollouts/ppo_random_mtp/002_master_chef_can/
  batch_000000.npz
  batch_000001.npz
  batch_000002.npz
```

聚合脚本会把同一物体下的所有 batch 合并。

## 拟定脚本

### `collect/collect_random_pregrasp_rollouts.py`

PPO 侧采集工具。它随机或循环采样 object-frame M/T/P bin，执行 SE
residual policy，并导出可聚合的 `batch_*.npz` rollout。

### `aggregate/build_field_store_from_rollouts.py`

PregraspPrior 侧聚合工具。

输入：

```bash
--rollout-root <PregraspPrior>/data/rollouts/ppo_random_mtp
--output-root <PregraspPrior>/data/fields/ppo_random_mtp
--sphere-template geodesic
--subdivision 3
--tilt-angles 0,15,30
--pitch-angles 0,15,30
```

输出：

```text
<PregraspPrior>/data/fields/ppo_random_mtp/manifest.json
<PregraspPrior>/data/fields/ppo_random_mtp/fields/*.npz
```

## 实施顺序

建议按这个顺序来：

1. 在 PPO 验证代码中加一个最小 rollout exporter。
2. 先只导出一个物体、一个 batch、3200 个 env。
3. 写一个 inspector，检查：
   - batch 样本数是否为 3200。
   - 成功率是多少。
   - 掌心位置是否合理。
   - candidate 分布是否合理。
4. 写聚合脚本，先用硬分配生成 `attempt_count/success_count`。
5. 保存为 `<PregraspPrior>/data/fields/ppo_random_mtp`。
6. 用 `visualize_field_store.py` 看热图。
7. 用 `sample_pregrasp_candidates.py` 从这个新 field store 采样。
8. 再送回 PPO 做闭环冒烟测试。

## 暂定第一版选择

第一版先做：

- object-level 聚合，不区分具体 object pose。
- hard nearest-bin assignment。
- 失败样本全部保留。
- `target_field = success_count / attempt_count`。
- `confidence` 使用归一化后的 `attempt_count`。
- 一个 batch 一个物体。

暂时不做：

- 不做 object-pose-conditioned field。
- 不对失败类型加权。
- 不做 soft label 扩散。
- 不过滤低样本 bin，先在可视化里观察。

## 后续待确认问题

- PPO 侧成功标志到底用哪个变量最稳定？
- 是否需要保存每个 env 的完整 object point cloud，还是保存 object-level 点云即可？
- `M/T/P` 是 PPO 采样时直接保存，还是后处理重新映射？
- 低样本 bin 是否在训练时 mask 掉？
- 多个 PPO checkpoint 生成的数据是否混合，还是按策略版本分开保存？
