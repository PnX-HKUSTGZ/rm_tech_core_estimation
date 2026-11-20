# data_enhancement

导入名为 `tech_core_data_enhancement`。该包对 data_generator 输出的训练划分做图像空间增强，生成纯背景负样本，并提供 model_train 训练时使用的在线增强函数。离线和在线两种用法共用同一套增强实现。

## 职责

| 负责 | 不负责 |
|---|---|
| 几何增强：缩放、平移，同时更新关键点像素、包围框和 K | 渲染时的光照与材质随机化（见 [data_generator](../data_generator/Readme.md)） |
| 光度增强：亮度 / 对比度、Gamma、色调、渐变照明 | 改变真值位姿或几何契约 |
| 退化增强：噪声、模糊、JPEG 压缩、局部遮挡 | 训练 ROI 抖动、热图生成（见 [model_train](../model_train/Readme.md)） |
| 背景替换（程序纹理）、纯背景负样本 | |

不做镜像翻转，因为 16 点的语义没有定义左右镜像下的点名置换。

## 规则

- 只增强 `split == "train"` 的记录。val / test / MuJoCo 数据原样保留，作为评价基准。
- 增强副本继承原记录的 `source_view` 和 `split`，因此同源图像不会跨越划分。
- 遮挡点和从原图外移入画面的投影点都标为 `visibility = 1`。移出画面或坐标为 `null` 的点标为 `0`。`camera_from_core` 保持不变。
- 每条输出记录在 `augmentation` 中写明所用操作和参数，在 `source` 中记录原记录 ID，以便复现。
- 负样本标记为 `negative: true`，`pixels` / `bbox_xyxy` 为 `null`。负样本只用于检测器训练和误检统计，不参与关键点训练。

## 使用

离线增强，输出新的数据目录：

```bash
cd src/train
uv run python -m tech_core_data_enhancement --data ../../output/exp/raw --out ../../output/exp/train \
  --copies 1 --negatives 0.05 --policy standard --seed 20261118
```

`--copies N` 表示每条训练记录保留原图，并额外生成 N 张增强图；`--negatives` 为负样本数量占目标图数量的比例；`--policy` 取 `standard`（几何 + 光度 + 退化）或 `none`（只复制原图并加入负样本）。

在线增强，由 model_train 的 DataLoader 调用：

```python
from tech_core_data_enhancement import photometric

rgb = photometric(rgb, rng)  # uint8 H×W×3 RGB → 同尺寸；不改变几何
```

## 调整增强策略

修改增强策略时，固定检测器、开发集和训练预算，只改变增强这一个因素做对照；数据量、视角范围与增强同时改变时，收益无法归因于某一种增强。

## 测试

`uv run pytest data_enhancement`：覆盖仿射变换后关键点与 K 的一致性（重投影误差 < 1e-6 pixel）、遮挡和出界时可见性的改写、来源分组不跨划分、负样本字段，以及相同种子的可复现性。
