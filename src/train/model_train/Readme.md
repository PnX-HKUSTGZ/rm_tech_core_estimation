# model_train

导入名为 `tech_core_model_train`。该包训练关键点识别模型，在仿真数据上以 PnP 位姿精度评价模型，并导出部署端使用的 ONNX 模型和 `model_meta.yaml`。同时持有**模型交付契约**。

## 职责

| 负责 | 不负责 |
|---|---|
| 把 manifest 转换为 YOLO detect / pose 标签和 LiteHRNet 裁剪样本 | 渲染与增强（见 [data_generator](../data_generator/Readme.md)、[data_enhancement](../data_enhancement/Readme.md)） |
| 训练 YOLO11 检测器、LiteHRNet-18/30、YOLO11-pose（对照） | 部署端推理与实车位姿解算（见 [src/deploy](../../deploy/Readme.md)） |
| Python 参考 PnP；端到端评价、轮次与门限选择、冻结后验收 | |
| ONNX 导出、PyTorch 与 ONNX 一致性校验，以及供部署端测试使用的金标准用例 | |

## 两条识别路线

| 路线 | 流程 | 定位 |
|---|---|---|
| 两阶段（部署方案） | RGB → YOLO11n 检测 → 取最高分框，扩为 1.25 倍正方形 → 裁剪并缩放到 256×256 → LiteHRNet-18 → 16 张 128×128 热图 + 16 个画内 logit → 峰值与局部对数二次亚像素细化 → 逆变换回原图 | 唯一导出到部署端的路线 |
| YOLO11-pose 直接 16 点 | RGB → YOLO11-pose | 对照基线，不导出 |

LiteHRNet 采用 HKUST ENTERPRIZE 的 LiteHRNet 配置（MMPose 实现，带画内可见性分支），输出由原来的 12 点改为 16 点。训练时使用 Blender 真值框加尺度 / 平移抖动；推理时始终使用检测器预测的框。

## 使用

```bash
cd src/train
uv run python -m tech_core_model_train detect    --data ../../output/exp/train --out ../../output/exp/detect --size n --epochs 20 --imgsz 320
uv run python -m tech_core_model_train keypoints --data ../../output/exp/train --out ../../output/exp/lite18 --variant 18 --epochs 40 --batch 16
uv run python -m tech_core_model_train finetune  --data ../../output/exp/train --initial ../../output/exp/lite18/best.pt \
  --dev-data ../../output/exp/dev --detector ../../output/exp/detect --out ../../output/exp/finetune --learning-rate 1e-4
uv run python -m tech_core_model_train pose      --data ../../output/exp/train --out ../../output/exp/pose --size n   # 对照
uv run python -m tech_core_model_train evaluate  --data ../../output/exp/accept --detector ../../output/exp/detect \
  --keypoints ../../output/exp/finetune --out ../../output/exp/evaluation
uv run python -m tech_core_model_train export    --detector ../../output/exp/detect --keypoints ../../output/exp/finetune \
  --evaluation ../../output/exp/evaluation --out ../../output/exp/release --verify
```

每个训练命令都在输出目录写 `experiment.yaml`，记录几何指纹、数据 manifest 的 SHA-256、超参数、种子、依赖版本和所选轮次。加载数据或权重时，若几何指纹不一致则拒绝运行。

## 评价协议

- **达标判据**：同一帧上，核心系原点的平移误差 ≤ 0.01 m，且旋转测地误差 ≤ 0.0873 rad。漏检、PnP 未解算的帧都计入分母；纯背景上的误检另外统计。
- **推理不使用真值**：真值框、可见性和位姿只用于监督和外部计分。
- **选择与验收分离**：轮次和关键点门限只能在 Blender val 或 MuJoCo 开发集上选择，先写出 `selection.yaml`，再运行测试推理。最终验收集要在模型、门限冻结之后用新种子生成（见 data_generator `--exclude`）；验收集一旦参与调参，就降级为开发集。
- **报告内容**：检测召回 / 精确率（IoU ≥ 0.5）、PCK@5px、解算率、达标率（附 Wilson 区间）、已解算帧的平移 / 旋转误差中位数与 P95、单帧端到端耗时，以及逐帧失败分类（无框 / 有框未解算 / 已解算超差）。Blender 与 MuJoCo 的结果分开报告。

`evaluate` 自动读取微调目录的 `selection.yaml`。对于初训模型和 `--pose` 对照模型，在 Blender 数据上评价时默认用 val 选门限；独立验收时传入此前生成的 `--selection` 或预先确定的 `--keypoint-threshold`。逐帧预测与分域汇总分别保存在 `frames.yaml` 和 `summary.yaml`。

参考 PnP（`pnp.py`）按几何契约的点序对齐，以 RANSAC + AP3P + SQPnP + IPPE 生成候选，LM 内点重拟合，Huber 细化；默认阈值 3 pixel，最少 4 个非退化内点。它既是评价基准，也是部署端 C++ 实现的数值参照。

## 模型交付契约

`export` 在发布目录中写出以下文件，部署端只读取这个目录：

| 文件 | 内容 |
|---|---|
| `detector.onnx` | YOLO11n 检测，输入 1×3×320×320，输出未经 NMS 的候选框；单目标只取最高分候选，部署端无需 NMS |
| `keypoints.onnx` | LiteHRNet-18，输入 1×3×256×256，输出热图 1×16×128×128 与 logit 1×16 |
| `model_meta.yaml` | 见下表 |
| `golden/` | 若干验收图像，以及 PyTorch 端的框、16 点像素、置信度和参考 PnP 位姿 |

| `model_meta.yaml` 字段 | 含义 |
|---|---|
| `schema_version` | 契约版本号 |
| `geometry` | 完整几何契约（点名、分组、核心系坐标 m）及其 SHA-256 |
| `detector` | 输入尺寸、通道顺序 RGB、letterbox 参数、归一化方式、框置信度门限 0.25 |
| `keypoints` | 裁剪尺寸 256、扩框系数 1.25、热图尺寸 128、均值 / 标准差、冻结的关键点门限（如 0.25）、置信度计算方式（sigmoid(logit) × 热图峰值） |
| `provenance` | 训练所用 `experiment.yaml` 的 SHA-256、验收结果摘要、导出时间 |

门限随模型冻结，部署端不提供覆盖参数。`export --verify` 用 ONNX Runtime 在 `golden/` 上复算，要求关键点像素与 PyTorch 的最大偏差 < 0.5 pixel，否则导出失败。金标准图像中没有任何检出时校验失败，此时需要更长的训练。部署端用 `golden/` 做 C++ 推理和 PnP 的回归测试。

## 依赖

Ultralytics 8.3.228、PyTorch 2.7 + CUDA 12.8、mmengine 0.10.7、mmcv-lite 2.2.0、OpenCV、onnx / onnxruntime。

## 测试

`uv run pytest model_train`：覆盖标签列数与可见性编码、裁剪与逆变换的往返误差、热图解码、几何指纹不一致时的拒绝、PnP 的平面 / 非平面 / 畸变 / 离群点 / 少于 4 点用例、评价分母包含失败帧，以及在小模型上执行导出与 `--verify`。
