#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "tech_core_estimation/pnp_estimator.hpp"

namespace tech_core_estimation::detail
{

/** @brief Selected model and image points in matching order. */
struct Correspondences
{
  std::vector<cv::Point3d> points;  ///< Core-frame positions in m.
  std::vector<cv::Point2d> pixels;  ///< Raw image coordinates in pixels.

  /** @brief Gather a support set by its indices within the selected observations. */
  Correspondences subset(const std::vector<int> & indices) const;
};

/** @brief OpenCV pinhole calibration for all hypotheses in a frame. */
struct Calibration
{
  cv::Matx33d matrix;              ///< Focal lengths and principal point in pixels.
  const std::vector<double> & distortion;  ///< OpenCV distortion coefficients.
};

/** @brief Core-to-optical rigid transform in OpenCV's parameterization. */
struct Pose
{
  cv::Vec3d rotation;     ///< Rodrigues rotation vector in rad.
  cv::Vec3d translation;  ///< Core origin in the camera optical frame in m.
};

/** @brief Initial pose and the observations supporting it. */
struct Seed
{
  Pose pose;                 ///< Initial core pose in the optical frame.
  std::vector<int> support;  ///< Indices into the selected observations.
};

/** @brief Check that the model points and image points each span at least two dimensions. */
bool noncollinear(const Correspondences & observations);

/** @brief Generate ordered RANSAC and subset PnP hypotheses for the selected observations. */
std::vector<Seed> generateSeeds(const Correspondences & observations, const Calibration & camera,
                                const PnPConfig & config);

}  // namespace tech_core_estimation::detail
