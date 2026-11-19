#include "tech_core_estimation/pnp_estimator.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "refine.hpp"

namespace tech_core_estimation
{
namespace
{

/** @brief Validate model metadata, camera calibration and solver controls on creation. */
std::string inputError(const KeypointModel & model, const CameraIntrinsics & camera,
                       const PnPConfig & config)
{
  if (model.points.size() < 4) {
    return "At least four model points are required";
  }
  if (!std::all_of(model.points.begin(), model.points.end(), [](const auto & p) { return p.allFinite(); })) {
    return "Model coordinates must be finite";
  }
  const auto & matrix = camera.camera_matrix;
  if (!matrix.allFinite() || matrix(0, 0) <= 0 || matrix(1, 1) <= 0 ||
      std::abs(matrix(0, 1)) > 1e-12 || std::abs(matrix(1, 0)) > 1e-12 ||
      (matrix.row(2) - Eigen::RowVector3d::UnitZ()).cwiseAbs().maxCoeff() > 1e-12) {
    return "Camera matrix must be finite with positive focal lengths, zero skew and last row [0, 0, 1]";
  }
  const auto count = camera.distortion.size();
  if ((count != 0 && count != 4 && count != 5 && count != 8 && count != 12 && count != 14) ||
      !std::all_of(camera.distortion.begin(), camera.distortion.end(), [](double x) { return std::isfinite(x); })) {
    return "Distortion must have 0, 4, 5, 8, 12 or 14 finite coefficients";
  }
  if (!std::isfinite(config.reprojection_threshold) || config.reprojection_threshold <= 0) {
    return "Reprojection threshold must be finite and positive";
  }
  if (config.ransac_iterations < 1 || !(config.ransac_confidence > 0 && config.ransac_confidence < 1)) {
    return "RANSAC needs at least one iteration and confidence strictly between zero and one";
  }
  if (config.local_optimization_steps < 0 || config.local_optimization_steps > 10) {
    return "Local optimization steps must be in [0, 10]";
  }
  return {};
}

}  // namespace

PnPEstimator::PnPEstimator(KeypointModel model, CameraIntrinsics camera, PnPConfig config)
: model_(std::move(model)), camera_(std::move(camera)), config_(config)
{
}

std::optional<PnPEstimator> PnPEstimator::create(KeypointModel model, CameraIntrinsics camera,
                                                 PnPConfig config, std::string & error)
{
  error = inputError(model, camera, config);
  if (!error.empty()) {
    return std::nullopt;
  }
  return PnPEstimator(std::move(model), std::move(camera), config);
}

PoseEstimate PnPEstimator::estimate(const Observation & observation) const
{
  cv::setRNGSeed(0);
  PoseEstimate result;
  detail::Correspondences selected;
  std::vector<size_t> indices;
  for (size_t i = 0; i < model_.points.size(); ++i) {
    if (observation.valid[i]) {
      const auto & p = model_.points[i];
      selected.points.emplace_back(p.x(), p.y(), p.z());
      selected.pixels.emplace_back(observation.pixels[i].x(), observation.pixels[i].y());
      indices.push_back(i);
    }
  }
  if (indices.size() < 4) {
    result.failure = PoseFailure::kTooFewPoints;
    return result;
  }
  if (!detail::noncollinear(selected)) {
    result.failure = PoseFailure::kDegenerate;
    return result;
  }
  detail::Calibration camera{{}, camera_.distortion};
  cv::eigen2cv(camera_.camera_matrix, camera.matrix);
  const auto seeds = detail::generateSeeds(selected, camera, config_);
  bool saw_positive_depth;
  auto candidates = detail::refineSeeds(selected, camera, config_, seeds, saw_positive_depth);
  const bool robust = indices.size() >= 5;
  const double threshold = config_.reprojection_threshold;
  double best_score = std::numeric_limits<double>::infinity();
  detail::Fit * best = nullptr;
  for (auto & candidate : candidates) {
    double score = 0.0;
    for (double residual : candidate.residuals) {
      const double squared = residual * residual;
      score += robust ? std::min(squared, threshold * threshold) : squared;
    }
    if (std::isfinite(score)) {
      ++result.candidate_count;
      if (score < best_score) {
        best_score = score;
        best = &candidate;
      }
    }
  }
  if (!best) {
    result.failure = seeds.empty() || saw_positive_depth ? PoseFailure::kNoConsensus : PoseFailure::kNoPositiveDepth;
    return result;
  }
  if (robust && config_.huber_refinement) {
    detail::refineHuber(selected, camera, threshold, *best);
  }
  cv::Matx33d rotation;
  cv::Rodrigues(best->pose.rotation, rotation);
  result.camera_from_core.linear() = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.val);
  result.camera_from_core.translation() = Eigen::Map<Eigen::Vector3d>(best->pose.translation.val);
  result.inliers.resize(model_.points.size(), false);
  double inlier_squared = 0.0;
  for (int i : best->inliers) {
    result.inliers[indices[i]] = true;
    inlier_squared += best->residuals[i] * best->residuals[i];
  }
  const auto & residuals = best->residuals;
  result.reprojection_rmse = std::sqrt(std::inner_product(residuals.begin(), residuals.end(),
                                                         residuals.begin(), 0.0) / residuals.size());
  result.inlier_rmse = std::sqrt(inlier_squared / best->inliers.size());
  result.ransac_applied = robust;
  result.failure = PoseFailure::kNone;
  return result;
}

}  // namespace tech_core_estimation
