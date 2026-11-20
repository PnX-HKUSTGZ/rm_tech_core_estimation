#include "refine.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace tech_core_estimation::detail
{
namespace
{

/** @brief 对每个选中点做正深度（cheirality）检验，包括未参与拟合的观测。 */
bool positiveDepth(const Correspondences & observations, const Pose & pose)
{
  if (!cv::checkRange(pose.rotation) || !cv::checkRange(pose.translation)) {
    return false;
  }
  cv::Matx33d rotation;
  cv::Rodrigues(pose.rotation, rotation);
  for (const auto & point : observations.points) {
    const double depth = rotation(2, 0) * point.x + rotation(2, 1) * point.y +
                         rotation(2, 2) * point.z + pose.translation[2];
    if (!(depth > 1e-9) || !std::isfinite(depth)) {
      return false;
    }
  }
  return true;
}

/** @brief 计算原图残差，并确定当前位姿的支持集。 */
Fit evaluate(const Correspondences & observations, const Calibration & camera, const Pose & pose,
             double threshold, bool robust)
{
  std::vector<cv::Point2d> projected;
  cv::projectPoints(observations.points, pose.rotation, pose.translation, camera.matrix,
                    camera.distortion, projected);
  Fit fit{pose, {}, {}};
  for (size_t i = 0; i < projected.size(); ++i) {
    const double residual = cv::norm(projected[i] - observations.pixels[i]);
    fit.residuals.push_back(residual);
    if (!robust || residual <= threshold) {
      fit.inliers.push_back(static_cast<int>(i));
    }
  }
  return fit;
}

/** @brief 计算径向 Huber 损失，可选地求其阻尼 Gauss-Newton 步长。 */
double huberCost(const Correspondences & active, const Calibration & camera, const Pose & pose,
                 double threshold, cv::Vec<double, 6> * step = nullptr)
{
  std::vector<cv::Point2d> residuals;
  cv::Mat jacobian;
  cv::projectPoints(active.points, pose.rotation, pose.translation, camera.matrix, camera.distortion,
                    residuals, step ? cv::OutputArray(jacobian) : cv::noArray());
  cv::Mat weighted;
  if (step) {
    weighted = jacobian.colRange(0, 6).clone();
  }
  double cost = 0.0;
  for (size_t i = 0; i < residuals.size(); ++i) {
    residuals[i] -= active.pixels[i];
    const double distance = cv::norm(residuals[i]);
    if (distance <= threshold) {
      cost += distance * distance;
    } else {
      cost += threshold * (2 * distance - threshold);
    }
    if (step) {
      const double weight = std::min(1.0, threshold / std::max(distance, 1e-10));
      weighted.rowRange(2 * i, 2 * i + 2) *= weight;
    }
  }
  if (step && std::isfinite(cost) && cv::checkRange(weighted)) {
    cv::Mat normal = weighted.t() * jacobian.colRange(0, 6);
    const cv::Mat gradient = weighted.t() * cv::Mat(residuals).reshape(1, weighted.rows);
    for (int i = 0; i < 6; ++i) {
      normal.at<double>(i, i) += 1e-6 * std::max(normal.at<double>(i, i), 1.0);
    }
    cv::solve(normal, gradient, *step, cv::DECOMP_SVD);
  }
  return cost;
}

}  // namespace

std::vector<Fit> refineSeeds(const Correspondences & observations, const Calibration & camera,
                             const PnPConfig & config, const std::vector<Seed> & seeds,
                             bool & saw_positive_depth)
{
  const bool robust = observations.points.size() >= 5;
  std::vector<Fit> candidates;
  saw_positive_depth = false;
  for (const auto & seed : seeds) {
    Pose pose = seed.pose;
    if (!positiveDepth(observations, pose)) {
      continue;
    }
    auto support = seed.support;
    auto group = observations.subset(support);
    for (int iteration = 0; iteration <= config.local_optimization_steps; ++iteration) {
      cv::solvePnPRefineLM(group.points, group.pixels, camera.matrix, camera.distortion,
        pose.rotation, pose.translation, {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 1e-10});
      if (!positiveDepth(observations, pose)) {
        break;
      }
      saw_positive_depth = true;
      auto fit = evaluate(observations, camera, pose, config.reprojection_threshold, robust);
      if (fit.inliers.size() < 4) {
        break;
      }
      group = observations.subset(fit.inliers);
      if (!noncollinear(group)) {
        break;
      }
      candidates.push_back(std::move(fit));
      if (!robust || support == candidates.back().inliers) {
        break;
      }
      support = candidates.back().inliers;
    }
  }
  return candidates;
}

void refineHuber(const Correspondences & observations, const Calibration & camera, double threshold,
                 Fit & best)
{
  std::vector<int> nearby;
  for (size_t i = 0; i < best.residuals.size(); ++i) {
    if (best.residuals[i] <= 2 * threshold) {
      nearby.push_back(static_cast<int>(i));
    }
  }
  const auto active = observations.subset(nearby);
  if (!noncollinear(active)) {
    return;
  }
  Pose pose = best.pose;
  for (int iteration = 0; iteration < 8; ++iteration) {
    cv::Vec<double, 6> step{};
    const double cost = huberCost(active, camera, pose, threshold, &step);
    bool improved = false;
    for (double scale : {1.0, 0.5, 0.25, 0.125}) {
      Pose proposal = pose;
      for (int i = 0; i < 3; ++i) {
        proposal.rotation[i] -= scale * step[i];
        proposal.translation[i] -= scale * step[i + 3];
      }
      if (positiveDepth(observations, proposal) && huberCost(active, camera, proposal, threshold) < cost) {
        pose = proposal;
        improved = true;
        break;
      }
    }
    if (!improved) {
      break;
    }
  }
  auto final = evaluate(observations, camera, pose, threshold, true);
  if (final.inliers.size() >= 4 && noncollinear(observations.subset(final.inliers))) {
    best = std::move(final);
  }
}

}  // namespace tech_core_estimation::detail
