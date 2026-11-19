# data_generator

导入名为 `tech_core_data_generator`。该包使用 Blender Cycles 与 MuJoCo 渲染科技核心，生成带完整真值的图像，并写出符合数据集契约的目录。同时持有整个仓库的**几何契约**和**数据集契约**。

## 职责

| 负责 | 不负责 |
|---|---|
| 科技核心网格、灯效状态（9 种状态 × 红/蓝）、相机内参与视角采样 | 图像空间增强、负样本（见 [data_enhancement](../data_enhancement/Readme.md)） |
| 渲染时的外观域随机化：光源、世界光、曝光、材质、发光强度、色调映射 | YOLO / 热图等模型专用标签（见 [model_train](../model_train/Readme.md)） |
| 真值：RGB、目标掩膜、`camera_from_core`、16 点像素坐标、可见性 | 位姿解算与评价 |
| 按来源视角划分 train / val / test；用独立种子生成 MuJoCo 开发集与验收集 | 装配动力学、能量单元、marker 运动 |
| 几何契约 `keypoints.yaml` 和几何指纹 | |

两种渲染器的分工：Blender 生成全部训练图和 Blender 验证 / 测试图；MuJoCo 只生成跨渲染器的开发集和验收集，**不进入梯度训练**，并且拒绝外观随机化参数。

## 几何契约

`tech_core_data_generator/keypoints.yaml` 是 16 个关键点的唯一来源，profile 为 `shell5_rule_lengths_model_reference_v2`：

| 序号 | 分组 | 点 |
|---|---|---|
| 0–4 | shell | 上部折线 3 点、两侧 0.074 m 竖段的下端 |
| 5–14 | marker | 静止初态 marker 的 10 个角点 |
| 15 | pillar | 中心轴与可见端盖表面的交点 |

坐标单位为 m，位于核心系。目前使用的是模型原点，与规则 E 点的偏差尚未标定，详见 demo 中的[点集说明](../../../ref/rm27_tech_core_estimation_demo/tech_core_keypoint_pos_estimation/EVALUATION.md)。

`geometry_contract()` 返回 profile、点名、点序、分组和坐标；`geometry_sha256()` 返回它的指纹。其他模块只能通过这两个函数读取几何，不得复制坐标。修改点集或坐标会改变指纹，旧数据和旧权重随之失效，需要重新生成和训练。

## 数据集契约

每个数据目录包含 `manifest.yaml`、`images/`、`masks/`。只有生成全部完成后才会写出 `manifest.yaml`；生成过程中以 YAML 多文档（`---` 分隔）逐条追加 `records.partial.yaml`，中断后的部分结果不能当作数据集使用。

顶层字段：`schema_version`、`geometry`（几何契约及其指纹）、`generator`（包版本、命令行参数、种子、Blender 版本）、`records`。

| record 字段 | 含义 |
|---|---|
| `id`、`split`、`source_view` | 记录 ID；数据划分；来源视角 ID。同一来源视角的所有记录必须属于同一划分 |
| `renderer` | `blender` 或 `mujoco` |
| `image`、`mask` | 相对路径；RGB 为 8 位 PNG，掩膜为目标轮廓 |
| `width`、`height`、`camera_matrix`、`distortion` | 图像尺寸与内参（pixel）；仿真无畸变 |
| `camera_from_core` | 4×4 矩阵，表示核心系在相机光学系中的位姿，平移单位 m |
| `pixels`、`visibility` | 16×2 原图像素坐标（相机背后的点为 `null`）；可见性 0 = 出界，1 = 遮挡，2 = 可见 |
| `bbox_xyxy` | 由掩膜和有效点得到的包围框；负样本为 `null` |
| `light_state`、`team`、`time` | 灯效状态、阵营、灯效时刻（s） |
| `appearance` | 渲染外观参数；`null` 表示默认外观 |
| `augmentation`、`source`、`negative` | 由 data_enhancement 填写：增强参数、原记录 ID、是否为负样本；原始渲染为 `null` / `null` / `false` |

data_enhancement 输出的目录沿用同一 schema。

## 使用

```bash
cd src/train
# Blender 训练 / 验证 / 测试，带外观域随机化
uv run python -m tech_core_data_generator blender --out ../../output/exp/raw \
  --train 1024 --val 96 --test 64 --width 320 --height 240 --samples 8 --seed 2026111804
# MuJoCo 开发集或验收集（每次使用新种子）
uv run python -m tech_core_data_generator mujoco --out ../../output/exp/dev \
  --regular 160 --stress 64 --seed 2026111802 --exclude ../../output/exp/raw
# 单帧预览，叠加关键点
uv run python -m tech_core_data_generator preview --backend blender --state ready --team red --out ../../output/preview
```

`blender --appearance random` 为默认策略；`--appearance default` 使用固定的参考场景外观。

`--exclude` 指定已有数据目录，新生成的相机位姿不得与其重复，保证验收集是真正的留出集。默认相机为 320×240、垂直视场 1.047 rad；视角以 E 点为中心按球坐标采样，范围如下：

| 分布 | 距离 r | 方位角 | 仰角 |
|---|---|---|---|
| 常规 | 0.45–1.0 m | ±0.96 rad | −0.35–0.61 rad |
| 边界 | 0.2–1.0 m | ±1.57 rad | ±1.05 rad |
| 训练（Blender） | 85%：0.4–1.0 m；15%：0.25–1.0 m | 85%：±1.13 rad；15%：±1.40 rad | 85%：−0.52–0.79 rad；15%：±0.96 rad |

这是实验采用的采样分布，并不覆盖全部比赛情形。

## 迁移来源

| 本包模块 | demo 来源 |
|---|---|
| `sim/`（几何、相机、灯效、帧、Blender / MuJoCo 后端）、`models/` | `tech_core_sim/`、`models/tech_core/` |
| `keypoints.yaml` | `tech_core_sim/keypoints.json`，内容不变，转写为 YAML；几何指纹随之重新计算 |
| `appearance.py` | `tech_core_sim/appearance.py`、`tech_core_dataset/domain_data.py:sample_appearance` |
| `dataset.py`（划分、manifest、几何契约） | `tech_core_dataset/data.py` 中不含增强与标签的部分、`domain_data.py`、`tech_core_yolo_litenet/holdout.py` |

网格的来源与许可证随 `models/` 一同迁移（`SOURCE.md`、`LICENSE-HKUST.txt`）。

## 测试

`uv run pytest data_generator`：覆盖灯效状态与频率边界、关键点坐标与指纹、视角采样的可复现性、可见性与遮挡判定、manifest 的划分约束，以及两种后端的轮廓和深度对齐。未安装 Blender 时 Blender 用例显式 skip，验收时需报告 skip 数量。
