# tech_core_estimation_ros

把 [tech_core_estimation_model](../tech_core_estimation_model/Readme.md) 和 [tech_core_estimation](../tech_core_estimation/Readme.md) 封装为 ROS 2 节点。节点订阅相机图像和标定，输出科技核心在世界系中的位姿。节点只做消息转换和转发，不重复实现模型或解算逻辑。

## 数据流

```text
Image + CameraInfo ──► TechCoreModel::infer ──► PnPEstimator::estimate ──► camera_from_core
                                                                              │
TF: world_from_camera（取图像时间戳）──────────────────────────────────────────┤
                                                                              ▼
                                         world_from_core = world_from_camera · camera_from_core
```

1. 启动时加载模型，用 `model_meta.yaml` 中的几何契约构造 `KeypointModel`。
2. 收到第一帧 CameraInfo 时创建 `PnPEstimator`；之后若 K 或 D 发生变化，则重新创建。
3. 每帧依次执行推理和解算；按图像时间戳查询 TF `world_frame ← header.frame_id`（相机光学系），合成得到世界系位姿后发布。
4. 未检出、解算失败或 TF 超时时不发布位姿，只计入诊断统计。

图像直接以原始（带畸变）形式输入模型，畸变由解算阶段处理。模型是在无畸变的仿真图像上训练的，真实镜头畸变对识别的影响尚未验证。

## 话题与 TF

| 名称 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `image` | `sensor_msgs/Image` | 订阅 | `bgr8` 或 `rgb8`，通过 `image_transport::CameraSubscriber` 与 `camera_info` 同步 |
| `camera_info` | `sensor_msgs/CameraInfo` | 订阅 | 原始针孔标定，焦距与主点单位 pixel，D 为 OpenCV 畸变系数；`frame_id` 为相机光学系 |
| `~/core_pose` | `geometry_msgs/PoseStamped` | 发布 | 核心系在世界系中的位姿，平移单位 m，时间戳与图像相同 |
| `~/debug_image` | `sensor_msgs/Image` | 发布 | 检测框、关键点与重投影模型叠加图；仅在有订阅者时生成 |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 发布 | 帧率、各类失败原因的计数、推理与解算耗时（s） |
| TF `world_frame → core_frame` | — | 发布 | 仅当 `publish_tf` 为真时发布 |

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_dir` | 无（必填） | model_train 导出的发布目录 |
| `backend` | `openvino` | 推理后端，目前仅支持 `openvino` |
| `device` | `CPU` | OpenVINO 设备名 |
| `world_frame` | `base_link` | 世界系的 frame 名，即机械臂基座系 |
| `core_frame` | `tech_core` | 核心系的 frame 名 |
| `publish_tf` | `true` | 是否发布 TF |
| `tf_timeout` | `0.05` | TF 查询超时，单位 s |
| `pnp.reprojection_threshold` 等 | 同 `PnPConfig` | 直接传入 `PnPConfig`，由 `create` 校验 |

框门限和关键点门限随模型冻结，不作为节点参数。参数示例见 `config/tech_core_estimation.yaml`。

## 使用

```bash
ros2 launch tech_core_estimation_ros tech_core_estimation.launch.py \
  model_dir:=/path/to/release image:=/camera/image_raw camera_info:=/camera/camera_info
```

节点以 `rclcpp_components` 组件形式实现（`tech_core_estimation_ros::EstimatorNode`），可以与相机驱动放在同一容器中，实现进程内零拷贝传图；同时提供独立可执行文件 `tech_core_estimation_node`。

## 测试

- launch_testing：发布 `golden/` 中的图像、对应 CameraInfo 和静态 TF，检查 `~/core_pose` 与金标准位姿的差异小于 1e-3 m 和 1e-3 rad。
- 消息转换：CameraInfo → `CameraIntrinsics`，`rgb8` / `bgr8` 编码，世界系位姿合成。
- 失败路径：缺少 TF 或无目标时不发布位姿，诊断计数相应增加。

## 依赖

`package.xml`：`rclcpp`、`rclcpp_components`、`sensor_msgs`、`geometry_msgs`、`diagnostic_updater`、`tf2_ros`、`tf2_eigen`、`cv_bridge`、`image_transport`、`tech_core_estimation_model`、`tech_core_estimation`。
