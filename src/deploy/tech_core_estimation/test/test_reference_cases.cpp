#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "tech_core_estimation/pnp_estimator.hpp"

namespace tech_core_estimation
{
namespace
{

TEST(ReferenceCases, ReplaysPythonEstimatesAndDiagnostics)
{
  const auto data = YAML::LoadFile(PNP_CASES_FILE);
  KeypointModel model;
  for (const auto & p : data["points"]) {
    model.points.emplace_back(p[0].as<double>(), p[1].as<double>(), p[2].as<double>());
  }
  CameraIntrinsics camera;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      camera.camera_matrix(row, col) = data["camera_matrix"][row][col].as<double>();
    }
  }
  camera.distortion = data["distortion"].as<std::vector<double>>();
  std::string error;
  const auto estimator = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(estimator) << error;
  ASSERT_EQ(data["cases"].size(), 40u);
  for (const auto & frame : data["cases"]) {
    SCOPED_TRACE(frame["id"].as<int>());
    Observation observation;
    for (const auto & p : frame["pixels"]) {
      observation.pixels.emplace_back(p[0].as<double>(), p[1].as<double>());
    }
    observation.valid = frame["valid"].as<std::vector<bool>>();
    const auto estimate = estimator->estimate(observation);
    const bool success = frame["success"].as<bool>();
    EXPECT_EQ(estimate.failure == PoseFailure::kNone, success);
    if (!success || estimate.failure != PoseFailure::kNone) {
      continue;
    }
    Eigen::Isometry3d reference;
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        reference.matrix()(row, col) = frame["camera_from_core"][row][col].as<double>();
      }
    }
    const Eigen::Isometry3d delta = reference.inverse() * estimate.camera_from_core;
    EXPECT_LT(delta.translation().norm(), 1e-3);
    EXPECT_LT(Eigen::AngleAxisd(delta.linear()).angle(), 1e-3);
    EXPECT_EQ(estimate.inliers, frame["inliers"].as<std::vector<bool>>());
    EXPECT_NEAR(estimate.reprojection_rmse, frame["reprojection_rmse"].as<double>(), 1e-4);
    EXPECT_NEAR(estimate.inlier_rmse, frame["inlier_rmse"].as<double>(), 1e-4);
  }
}

}  // namespace
}  // namespace tech_core_estimation
