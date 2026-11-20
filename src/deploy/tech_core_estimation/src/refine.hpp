#pragma once

#include "candidates.hpp"

namespace tech_core_estimation::detail
{

/** @brief 细化后的候选解，附带在全部选中观测上的残差。 */
struct Fit
{
  Pose pose;                      ///< 核心系在相机光学系中的位姿。
  std::vector<double> residuals;  ///< 欧氏重投影误差，单位 pixel。
  std::vector<int> inliers;       ///< 最终支持观测的下标。
};

/** @brief 对每个种子做 LM 拟合，支持集变化时重拟合，并记录正深度拟合结果用于失败分类。 */
std::vector<Fit> refineSeeds(const Correspondences & observations, const Calibration & camera,
                             const PnPConfig & config, const std::vector<Seed> & seeds,
                             bool & saw_positive_depth);

/** @brief 在邻近观测上最小化 Huber 损失；若失去一致集则保留原拟合。 */
void refineHuber(const Correspondences & observations, const Calibration & camera, double threshold,
                 Fit & best);

}  // namespace tech_core_estimation::detail
