# rm_tech_core_estimation/src/deploy

实车部署代码，使用 C++17 编写，以 colcon（ament_cmake）构建。部署端读取 [model_train](../train/model_train/Readme.md) 导出的发布目录（ONNX + `model_meta.yaml`），对相机图像进行推理，并输出科技核心在世界系中的位姿。

| 包 | 职责 | ROS 依赖 |
|---|---|---|
| [tech_core_estimation_model](./tech_core_estimation_model/Readme.md) | 模型加载与推理：BGR 图像 → 检测框 + 16 点像素坐标与置信度；后端为 OpenVINO | 无，仅用 ament 构建 |
| [tech_core_estimation](./tech_core_estimation/Readme.md) | 位姿解算：16 点观测 + 相机标定 → `camera_from_core`；世界系动态追踪尚在规划 | 无，仅用 ament 构建 |
| [tech_core_estimation_ros](./tech_core_estimation_ros/Readme.md) | ROS 2 封装：订阅图像与相机标定，查询 TF，发布世界系下的核心位姿 | rclcpp 等 |

前两个包不依赖 ROS 运行时，可以单独链接到其他程序；两者之间也互不依赖，只由 ROS 节点把模型输出传给位姿解算。

## 约定

- 单位遵守 [AGENTS.md](../../AGENTS.md)：长度 m，角度 rad，时间 s。图像坐标、包围框、重投影误差和相机内参使用 pixel。
- 坐标系使用名称：相机光学系、核心系、世界系。变换命名为 `<目标系>_from_<源系>`。
- 公共头文件不通过异常传递错误：加载和初始化通过返回值报告失败原因，单帧没有结果属于正常输出。
- 16 点的顺序和核心系坐标只来自 `model_meta.yaml`，部署端不另外保存一份坐标。
- 契约与配置文件统一使用 YAML，C++ 端以 yaml-cpp 读取。

## 构建

```bash
source /opt/ros/<distro>/setup.bash
rosdep install --from-paths src/deploy --ignore-src -y
colcon build --base-paths src/deploy --cmake-args -DTECH_CORE_GOLDEN_DIR=<发布目录>  # 金标准测试需要发布目录
colcon test --base-paths src/deploy && colcon test-result --verbose
```

OpenVINO 不在 rosdep 中，因此不写入 `package.xml`，安装方式见 [tech_core_estimation_model](./tech_core_estimation_model/Readme.md)。
