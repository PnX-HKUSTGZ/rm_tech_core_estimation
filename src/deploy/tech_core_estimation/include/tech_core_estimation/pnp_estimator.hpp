/**
 * @file pnp_estimator.hpp
 * @brief 科技核心单帧关键点位姿解算接口。
 */
#pragma once

#include <Eigen/Geometry>

#include <optional>
#include <string>
#include <vector>

namespace tech_core_estimation
{

/** @brief 按观测顺序排列的关键点模型。 */
struct KeypointModel
{
  std::vector<Eigen::Vector3d> points;  ///< 核心系三维坐标，单位 m。
};

/** @brief 一帧原图观测，与模型点按下标对齐。 */
struct Observation
{
  std::vector<Eigen::Vector2d> pixels;  ///< 原图坐标，单位 pixel，u 向右、v 向下。
  std::vector<bool> valid;              ///< 逐点参与标记；为真时坐标须为有限值。
};

/** @brief 相机光学系的 OpenCV 针孔标定。 */
struct CameraIntrinsics
{
  Eigen::Matrix3d camera_matrix;   ///< 焦距和主点单位 pixel，零 skew，末行为 [0, 0, 1]。
  std::vector<double> distortion;  ///< 0、4、5、8、12 或 14 个标准 OpenCV 畸变参数。
};

/** @brief 单帧 PnP 参数。 */
struct PnPConfig
{
  double reprojection_threshold = 3.0;  ///< 内点半径，单位 pixel，须为有限正数。
  int ransac_iterations = 300;          ///< RANSAC 最大迭代次数，至少为 1。
  double ransac_confidence = 0.999;     ///< RANSAC 置信度，取值在开区间 (0, 1)。
  int local_optimization_steps = 3;     ///< 首次 LM 拟合后的内点重拟合轮数，范围 [0, 10]。
  bool huber_refinement = true;         ///< 是否对鲁棒解的邻近观测做 Huber 细化。
};

/** @brief 单帧解算状态。 */
enum class PoseFailure
{
  kNone,             ///< 解算成功。
  kTooFewPoints,     ///< 可用观测少于 4 个。
  kDegenerate,       ///< 参与的三维点或图像点共线。
  kNoConsensus,      ///< 无候选或没有满足内点要求的候选。
  kNoPositiveDepth   ///< 候选均无法使所有参与点处于相机前方。
};

/** @brief 单帧结果，仅 failure 为 kNone 时其余字段有效。 */
struct PoseEstimate
{
  PoseFailure failure = PoseFailure::kNoConsensus;  ///< 解算状态。
  Eigen::Isometry3d camera_from_core = Eigen::Isometry3d::Identity();  ///< 核心系在相机光学系中的位姿，平移单位 m。
  std::vector<bool> inliers;       ///< 与模型对齐的最终内点标记，未参与点为假。
  double reprojection_rmse = 0.0;  ///< 全部参与点的重投影均方根误差，单位 pixel。
  double inlier_rmse = 0.0;        ///< 最终内点重投影均方根误差，单位 pixel。
  bool ransac_applied = false;     ///< 至少 5 个可用点时启用鲁棒估计。
  int candidate_count = 0;         ///< 参与排序的有效拟合次数。
};

/**
 * @brief 由固定模型与标定解算单帧位姿。
 * @see src/deploy/tech_core_estimation/Readme.md（单帧解算流程）
 */
class PnPEstimator
{
public:
  /**
   * @brief 使用有限模型点、合法标定及参数创建解算器。
   * @param[in] model 核心系关键点，单位 m，至少 4 点。
   * @param[in] camera 相机内参，焦距和主点单位 pixel。
   * @param[in] config 单帧参数，重投影阈值单位 pixel。
   * @param[out] error 失败原因；成功时清空。
   * @return 创建的解算器，输入非法时返回 std::nullopt。
   */
  static std::optional<PnPEstimator> create(KeypointModel model, CameraIntrinsics camera,
                                            PnPConfig config, std::string & error);

  /**
   * @brief 根据一帧有效观测解算核心系在相机光学系中的位姿。
   * @param[in] observation 原图观测，坐标单位 pixel。
   * @pre pixels 和 valid 的长度均等于模型点数，顺序与模型一致。
   * @note 至少 4 个非共线观测的重投影误差须小于或等于配置的内点半径。
   * @return 解算状态、位姿（平移单位 m）、内点及重投影诊断（单位 pixel）。
   */
  PoseEstimate estimate(const Observation & observation) const;

private:
  /** @brief 保存已经校验的模型、标定及解算参数。 */
  PnPEstimator(KeypointModel model, CameraIntrinsics camera, PnPConfig config);

  KeypointModel model_;      ///< 核心系模型，单位 m。
  CameraIntrinsics camera_;  ///< 观测使用的相机标定。
  PnPConfig config_;         ///< 单帧解算参数。
};

}  // namespace tech_core_estimation
