# 第三方来源和许可

1. HKUST ENTERPRIZE，RM2026-Engineer-Assembly-Algorithm，MIT。
   固定提交 `7fb7609aa6b5a8b1992d32e14bc17d24d1a15578`。
   [LiteHRNet配置](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm/blob/7fb7609aa6b5a8b1992d32e14bc17d24d1a15578/src/nn/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py)，
   [可见性头](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm/blob/7fb7609aa6b5a8b1992d32e14bc17d24d1a15578/src/nn/hrnet/custom/visibility_head.py)。
   原文件以 `.txt` 保留于vendor供比对，许可证 `src/tech_core_model_train/litehrnet/vendor/LICENSE-HKUST.txt`。
   同一提交的 [LiteHRNet-30配置](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm/blob/7fb7609aa6b5a8b1992d32e14bc17d24d1a15578/src/nn/hrnet/configs/td-hm_litehrnet30_exchange12_v11.0.py) 同样保存为 `src/tech_core_model_train/litehrnet/vendor/hkust_litehrnet30_config.txt`。
   本包将输出头适配为16点。参考 PnP 使用该项目的[位姿解算](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm/blob/7fb7609aa6b5a8b1992d32e14bc17d24d1a15578/runtime/core/arm_exchange_core/perception/pose.py)，以及 demo 0.3.0 的 AP3P 候选与 Huber 细化。

2. OpenMMLab MMPose v1.3.2（上游骨干再改编自 HRNet/Lite-HRNet），Apache-2.0。
   [骨干源码](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/models/backbones/litehrnet.py)。
   `src/tech_core_model_train/litehrnet/vendor/litehrnet.py` 保留原算法及版权头，仅移除MODELS注册，替换BaseBackbone/channel_shuffle为本地兼容导入；compat.py采用相同reshape/transpose运算。BaseBackbone在此仅需mmengine BaseModule的默认行为。
   上游原始下载字节SHA256：`7be2543d8a9a3e7bf93e9a7abc3a37defa0e3ca29990ffdac841d525f90534ad`。
   适配文件SHA256（UTF-8无BOM、LF、单个结尾换行）：`c487e3ad0e191deb231438bfe084940eb3e7fb9abb86c6ddd811b73c607781f0`。
   完整许可证见 `src/tech_core_model_train/litehrnet/vendor/LICENSE-MMPose.txt`。该第三方文件保留上游docstring格式；本项目新增公共接口使用Doxygen风格。

3. COCO预训练骨干取OpenMMLab公开模型索引：
   [litehrnet_coco.yml](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/topdown_heatmap/coco/litehrnet_coco.yml)。
   下载URL、实际内容SHA256写入每次实验的 `experiment.yaml`，原人体关键点头不用于本任务。

4. Ultralytics 8.3.228为外部AGPL-3.0依赖，YOLO11 COCO权重来自 `ultralytics/assets` v8.3.0；未把其源代码或权重复制进Git。mmcv-lite 2.2.0、mmengine 0.10.7为Apache-2.0依赖。
