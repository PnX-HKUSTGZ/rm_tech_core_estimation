#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "tech_core_estimation/pnp_estimator.hpp"

namespace tech_core_estimation::detail
{

/** @brief 选中的模型点与图像点，按对应顺序排列。 */
struct Correspondences
{
  std::vector<cv::Point3d> points;  ///< 核心系坐标，单位 m。
  std::vector<cv::Point2d> pixels;  ///< 原图坐标，单位 pixel。

  /** @brief 按选中观测内的下标取出支持集。 */
  Correspondences subset(const std::vector<int> & indices) const;
};

/** @brief 一帧内所有假设共用的 OpenCV 针孔标定。 */
struct Calibration
{
  cv::Matx33d matrix;              ///< 焦距与主点，单位 pixel。
  const std::vector<double> & distortion;  ///< OpenCV 畸变参数。
};

/** @brief 核心系到相机光学系的刚体变换，采用 OpenCV 参数化。 */
struct Pose
{
  cv::Vec3d rotation;     ///< Rodrigues 旋转向量，单位 rad。
  cv::Vec3d translation;  ///< 核心系原点在相机光学系中的位置，单位 m。
};

/** @brief 初始位姿及支持它的观测。 */
struct Seed
{
  Pose pose;                 ///< 核心系在相机光学系中的初始位姿。
  std::vector<int> support;  ///< 选中观测内的下标。
};

/** @brief 检查模型点与图像点各自至少张成二维。 */
bool noncollinear(const Correspondences & observations);

/** @brief 为选中观测生成有序的 RANSAC 与子集 PnP 假设。 */
std::vector<Seed> generateSeeds(const Correspondences & observations, const Calibration & camera,
                                const PnPConfig & config);

}  // namespace tech_core_estimation::detail
