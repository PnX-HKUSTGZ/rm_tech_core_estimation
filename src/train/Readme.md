# rm_tech_core_estimation/src/train

负责数据生成、数据增强和模型训练，最终交付部署端使用的 ONNX 模型。三个子目录是同一个 uv 工作区中的三个 Python 包。

| 包 | 导入名 | 职责 |
|---|---|---|
| [data_generator](./data_generator/Readme.md) | `tech_core_data_generator` | 在 Blender / MuJoCo 中渲染并生成真值；持有几何契约和数据集契约 |
| [data_enhancement](./data_enhancement/Readme.md) | `tech_core_data_enhancement` | 离线图像增强、生成纯背景负样本，并提供训练时使用的在线增强函数 |
| [model_train](./model_train/Readme.md) | `tech_core_model_train` | 训练 YOLO11 检测器、LiteHRNet、YOLO11-pose 对照模型，完成 PnP 位姿评价和 ONNX 导出 |

依赖方向：`data_enhancement → data_generator`，`model_train → data_generator, data_enhancement`。数据包之间只通过 `manifest.yaml` 目录交换数据，不互相调用内部函数。

## 环境（uv）

工作区根目录为 `src/train/`。`pyproject.toml` 声明成员包、Python 3.12，以及 PyTorch 的 CUDA 索引；`uv.lock` 纳入版本管理。

```bash
cd src/train
uv sync                                  # 安装全部成员及开发依赖，生成 .venv/
uv run pytest                            # 运行三个包的测试
uv run python -m tech_core_data_generator --help
```

```toml
# src/train/pyproject.toml（要点）
[tool.uv.workspace]
members = ["data_generator", "data_enhancement", "model_train"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

外部工具不由 uv 管理：

- **Blender ≥ 4.2**（已验证 4.5.3 LTS），用于生成训练数据。通过 `BLENDER_EXECUTABLE` 环境变量或 `PATH` 指定。
- **MuJoCo 无显示渲染**：在没有显示器的 Linux 上，启动 Python 之前设置 `MUJOCO_GL=egl`（需要 EGL 驱动）；纯软件环境可改用 `osmesa`。

`src/train/` 下放置 `COLCON_IGNORE`，防止在仓库根目录执行 colcon 时扫描到这些 Python 包。所有命令的输出写入仓库根目录 `output/<实验名>/`，并且要求输出目录为空，避免覆盖已有实验。

## 端到端流程

```bash
uv run python -m tech_core_data_generator blender --out ../../output/exp/raw ...
uv run python -m tech_core_data_generator mujoco  --out ../../output/exp/dev ...
uv run python -m tech_core_data_enhancement       --data ../../output/exp/raw --out ../../output/exp/train ...
uv run python -m tech_core_model_train detect     --data ../../output/exp/train ...
uv run python -m tech_core_model_train keypoints  --data ../../output/exp/train ...
uv run python -m tech_core_model_train evaluate   --data ../../output/exp/dev ...
uv run python -m tech_core_model_train export     --out ../../output/exp/release ...
```

各命令的参数见对应包的 Readme。
