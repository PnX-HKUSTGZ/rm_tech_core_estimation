#pragma once

#include "candidates.hpp"

namespace tech_core_estimation::detail
{

/** @brief Refined candidate with residuals on the full set of selected observations. */
struct Fit
{
  Pose pose;                      ///< Core pose in the camera optical frame.
  std::vector<double> residuals;  ///< Euclidean reprojection errors in pixels.
  std::vector<int> inliers;       ///< Indices of the final supporting observations.
};

/** @brief LM-fit each seed and refit changed support, recording positive-depth fits for failure classification. */
std::vector<Fit> refineSeeds(const Correspondences & observations, const Calibration & camera,
                             const PnPConfig & config, const std::vector<Seed> & seeds,
                             bool & saw_positive_depth);

/** @brief Minimize Huber loss on nearby observations, keeping the original fit if consensus is lost. */
void refineHuber(const Correspondences & observations, const Calibration & camera, double threshold,
                 Fit & best);

}  // namespace tech_core_estimation::detail
