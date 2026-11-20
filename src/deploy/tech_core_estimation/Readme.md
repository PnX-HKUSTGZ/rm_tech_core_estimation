# tech_core_estimation

科技核心位姿解算库。输入为 16 个关键点的像素观测、核心系三维点和相机标定，输出核心系在相机光学系中的位姿 `camera_from_core` 及诊断信息。世界系下的动态追踪尚在规划。

## 职责

| 负责 | 不负责 |
|---|---|
| 单帧 PnP：与 model_train 的 Python 参考 PnP（`tech_core_model_train.pnp`）流程一致的 C++ 实现 | 神经网络推理（见 [tech_core_estimation_model](../tech_core_estimation_model/Readme.md)） |
| 处理相机畸变（标准 OpenCV 0/4/5/8/12/14 个系数） | TF 查询与相机外参（由 [tech_core_estimation_ros](../tech_core_estimation_ros/Readme.md) 完成） |
| 失败分类与诊断：参与点、内点、重投影 RMSE、候选数 | 关键点门限（随模型冻结，由模型库应用） |
| 世界系动态追踪（规划） | |

## 单帧解算流程

与 Python 参考实现逐步对应，参数默认值相同：

1. 只取 `valid` 为真的点；少于 4 点时判为失败。
2. 对全部点做 SQPnP，共面时也尝试 IPPE。至少 5 点时还用 RANSAC 生成候选（共面用去畸变像素的单应性，非共面用 PnP，另加 AP3P 假设）。5 或 6 点时还会依次尝试剔除其中一个点。
3. 对每个候选做 LM 优化，取重投影误差 ≤ 阈值的点为内点，支持集变化时最多重拟合 3 轮。按全部观测的残差平方和排序，至少 5 点时才截断离群点代价。
4. 至少 5 点时，对距当前投影 2 倍阈值以内的观测做 Huber 加权细化。若细化破坏共识，则保留细化前的候选。
5. 所有参与点深度须为正，且至少有 4 个非共线内点。恰有 4 点时，每个点都须满足重投影阈值。

不使用时间滤波和视角先验。平面或接近正视的几何可能存在姿态二义性：重投影误差小并不代表位姿正确。

## 接口

```cpp
namespace tech_core_estimation {

/** @brief 已知的关键点三维模型。 */
struct KeypointModel {
  std::vector<Eigen::Vector3d> points;  ///< 核心系坐标，单位 m；顺序与观测一致。
};

/** @brief 单帧关键点观测，顺序与 KeypointModel 一致。 */
struct Observation {
  std::vector<Eigen::Vector2d> pixels;  ///< 原图像素坐标，单位 pixel，u 向右、v 向下。
  std::vector<bool> valid;              ///< 是否参与解算。
};

/** @brief 相机标定。 */
struct CameraIntrinsics {
  Eigen::Matrix3d camera_matrix;  ///< 内参矩阵，单位 pixel，零 skew。
  std::vector<double> distortion; ///< OpenCV 畸变系数，个数为 0/4/5/8/12/14。
};

/** @brief 解算参数。 */
struct PnPConfig {
  double reprojection_threshold = 3.0;  ///< 内点阈值，单位 pixel。
  int ransac_iterations = 300;          ///< RANSAC 迭代次数。
  double ransac_confidence = 0.999;     ///< RANSAC 置信度。
  int local_optimization_steps = 3;     ///< 内点重拟合轮数上限。
  bool huber_refinement = true;         ///< 是否做 Huber 加权细化。
};

/** @brief 解算失败原因。 */
enum class PoseFailure { kNone, kTooFewPoints, kDegenerate, kNoConsensus, kNoPositiveDepth };

/** @brief 单帧解算结果。 */
struct PoseEstimate {
  PoseFailure failure;                  ///< kNone 表示成功；否则其余字段无意义。
  Eigen::Isometry3d camera_from_core;   ///< 核心系在相机光学系中的位姿，平移单位 m。
  std::vector<bool> inliers;            ///< 逐点是否为最终内点。
  double reprojection_rmse;             ///< 全部参与点的重投影 RMSE，单位 pixel。
  double inlier_rmse;                   ///< 内点重投影 RMSE，单位 pixel。
  bool ransac_applied;                  ///< 是否使用了鲁棒估计（不少于 5 点时启用）。
  int candidate_count;                  ///< 参与排序的候选数。
};

class PnPEstimator {
public:
  /** @brief 创建解算器；模型点少于 4 个、配置越界或标定非法时返回 std::nullopt，并在 error 中写入原因。 */
  static std::optional<PnPEstimator> create(KeypointModel model, CameraIntrinsics camera,
                                            PnPConfig config, std::string & error);

  /** @brief 解算一帧。@pre observation 的长度与模型点数相同。 */
  PoseEstimate estimate(const Observation & observation) const;
};

}  // namespace tech_core_estimation
```

模型与观测按下标对齐。点序的一致性由 ROS 节点在启动时通过 `model_meta.yaml` 的几何指纹保证，因此本库不再按点名匹配。相机标定在 `create` 时校验一次；标定变化时需重新创建解算器。

## 动态追踪（规划）

输入为带时间戳（s）的世界系单帧测量 `world_from_core`，输出平滑后的位姿和有效标记。运动模型（静止、升降、步骤 5 中的旋转）以及测量噪声，需要等到有实测连续序列后再确定，并以抖动、延迟、跟踪丢失和恢复时间进行评价。在此之前，ROS 节点直接发布单帧结果。

## 测试与验收

- **数值一致性**：`test/data/pnp_cases.yaml` 是由 Python 参考实现生成的 40 个固定回归用例（带畸变，覆盖全部点、离群点、丢点、仅 marker 共面点、4/5/6 点和不可解帧）。每帧的成功 / 失败须与参考一致，成功帧的位姿差异小于 1e-3 m 和 1e-3 rad，内点标记相同，重投影 RMSE 差异小于 1e-4 pixel。
- **单元用例**：平面与非平面、带畸变、离群点、丢点、少于 4 点、共线退化、无正深度解。
- **耗时**：16 点单次解算的中位数与 P95（s），不得慢于 Python 参考实现。

## 依赖

`package.xml`：`ament_cmake`、`eigen`、`libopencv-dev`（需要 OpenCV ≥ 4.5.3 以支持 SQPnP）。
