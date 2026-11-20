#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "tech_core_estimation/pnp_estimator.hpp"

namespace tech_core_estimation
{
namespace
{

/** @brief 单帧测试共用的合成标定与模型。 */
class PnPTest : public ::testing::Test
{
protected:
  /** @brief 把核心放在带径向畸变的相机前方。 */
  PnPTest()
  {
    camera.camera_matrix << 600., 0., 320., 0., 600., 240., 0., 0., 1.;
    camera.distortion = {-0.1, 0., 0., 0.};
    const Eigen::Vector3d rotation(0.12, -0.1, 0.21);
    camera_from_core.linear() = Eigen::AngleAxisd(rotation.norm(), rotation.normalized()).toRotationMatrix();
    camera_from_core.translation() = Eigen::Vector3d(0.01, 0.01, 0.7);
  }

  /** @brief 把夹具模型投影到畸变图像，单位 pixel。 */
  Observation project() const
  {
    Observation observation;
    for (const auto & point : model.points) {
      Eigen::Vector2d xy = (camera_from_core * point).hnormalized();
      if (!camera.distortion.empty()) {
        xy *= 1 + camera.distortion[0] * xy.squaredNorm();
      }
      observation.pixels.push_back((camera.camera_matrix * xy.homogeneous()).hnormalized());
      observation.valid.push_back(true);
    }
    return observation;
  }

  /** @brief 核心系关键点，单位 m。 */
  KeypointModel model{{{-0.1, -0.1, -0.1}, {0.1, -0.1, -0.1}, {-0.1, 0.1, -0.1}, {0.1, 0.1, -0.1},
                       {0., 0., 0.15}, {0.06, -0.01, 0.12}, {-0.04, 0.08, 0.1}, {0.02, 0.05, 0.12}}};
  CameraIntrinsics camera;  ///< 相机标定，单位 pixel。
  Eigen::Isometry3d camera_from_core = Eigen::Isometry3d::Identity();  ///< 真值位姿，平移单位 m。
  std::string error;  ///< 创建失败原因。
};

TEST_F(PnPTest, PlanarDistortedConsensus)
{
  for (auto & point : model.points) {
    point.z() = -0.1;
  }
  auto obs = project();
  obs.pixels[6] += Eigen::Vector2d(50., -60.);
  const auto solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  const auto result = solver->estimate(obs);
  ASSERT_EQ(result.failure, PoseFailure::kNone);
  EXPECT_TRUE(result.camera_from_core.isApprox(camera_from_core, 1e-7));
  auto expected = obs.valid;
  expected[6] = false;
  EXPECT_EQ(result.inliers, expected);
  EXPECT_TRUE(result.ransac_applied);
  EXPECT_GT(result.candidate_count, 0);
  EXPECT_LT(result.inlier_rmse, 1e-7);
  EXPECT_NEAR(result.reprojection_rmse, std::sqrt(6100.0 / 8), 1e-7);
}

TEST_F(PnPTest, FourUsableNonplanarPoints)
{
  auto obs = project();
  obs.valid = {true, true, true, false, true, false, false, false};
  const auto solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  const auto result = solver->estimate(obs);
  ASSERT_EQ(result.failure, PoseFailure::kNone);
  EXPECT_TRUE(result.camera_from_core.isApprox(camera_from_core, 1e-7));
  EXPECT_FALSE(result.ransac_applied);
  EXPECT_EQ(result.inliers, obs.valid);
  obs.valid[1] = false;
  EXPECT_EQ(solver->estimate(obs).failure, PoseFailure::kTooFewPoints);
}

TEST_F(PnPTest, RejectsDegenerateCorrespondences)
{
  auto obs = project();
  auto solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  for (auto & p : obs.pixels) {
    p.y() = 1.0;
  }
  EXPECT_EQ(solver->estimate(obs).failure, PoseFailure::kDegenerate);
  obs = project();
  for (auto & p : model.points) {
    p.y() = p.z() = 0.0;
  }
  solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  EXPECT_EQ(solver->estimate(obs).failure, PoseFailure::kDegenerate);
}

TEST_F(PnPTest, ReportsNoPositiveDepth)
{
  camera.distortion.clear();
  camera_from_core.translation().z() = 0.09;
  auto obs = project();
  obs.valid = {true, true, true, false, true, false, false, false};
  const auto solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  EXPECT_EQ(solver->estimate(obs).failure, PoseFailure::kNoPositiveDepth);
}

TEST_F(PnPTest, OtherHypothesesSurviveCollapsedImageCoordinates)
{
  camera.distortion.clear();
  auto obs = project();
  for (size_t i = 0; i < model.points.size(); ++i) {
    obs.pixels[i] = Eigen::Vector2d(320., 240.) + 0.5 * model.points[i].head<2>();
  }
  const auto solver = PnPEstimator::create(model, camera, {}, error);
  ASSERT_TRUE(solver) << error;
  const auto full = solver->estimate(obs);
  ASSERT_EQ(full.failure, PoseFailure::kNone);
  EXPECT_EQ(full.inliers, obs.valid);
  obs.valid = {true, true, true, false, true, false, false, false};
  EXPECT_EQ(solver->estimate(obs).failure, PoseFailure::kNoConsensus);
}

TEST_F(PnPTest, AcceptsSupportedDistortionModels)
{
  for (int n : {0, 4, 5, 8, 12, 14}) {
    camera.distortion.assign(n, 0.0);
    const auto solver = PnPEstimator::create(model, camera, {}, error);
    ASSERT_TRUE(solver) << error;
    const auto result = solver->estimate(project());
    ASSERT_EQ(result.failure, PoseFailure::kNone);
    EXPECT_TRUE(result.camera_from_core.isApprox(camera_from_core, 1e-7));
  }
}

TEST_F(PnPTest, RejectsInvalidModelAndCalibration)
{
  const auto invalid = [&](const KeypointModel & m, const CameraIntrinsics & c) {
    EXPECT_FALSE(PnPEstimator::create(m, c, {}, error));
    EXPECT_FALSE(error.empty());
  };
  invalid({}, camera);
  auto bad = camera;
  bad.camera_matrix(0, 1) = 0.5;
  invalid(model, bad);
  bad = camera;
  bad.camera_matrix(2, 2) = 0.0;
  invalid(model, bad);
  bad = camera;
  bad.camera_matrix(1, 1) = 0.0;
  invalid(model, bad);
  bad = camera;
  bad.camera_matrix(0, 2) = std::numeric_limits<double>::quiet_NaN();
  invalid(model, bad);
  bad = camera;
  bad.distortion.resize(3);
  invalid(model, bad);
  bad = camera;
  bad.distortion[1] = std::numeric_limits<double>::infinity();
  invalid(model, bad);
  model.points[0].z() = std::numeric_limits<double>::quiet_NaN();
  invalid(model, camera);
}

TEST_F(PnPTest, ValidatesSolverOptions)
{
  const auto invalid = [&](const PnPConfig & config) {
    EXPECT_FALSE(PnPEstimator::create(model, camera, config, error));
    EXPECT_FALSE(error.empty());
  };
  PnPConfig config;
  config.reprojection_threshold = 0.0;
  invalid(config);
  config = {};
  config.ransac_iterations = 0;
  invalid(config);
  config = {};
  config.ransac_confidence = 1.0;
  invalid(config);
  config = {};
  config.local_optimization_steps = -1;
  invalid(config);
  config.local_optimization_steps = 11;
  invalid(config);
  config.local_optimization_steps = 0;
  config.huber_refinement = false;
  const auto solver = PnPEstimator::create(model, camera, config, error);
  ASSERT_TRUE(solver) << error;
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(solver->estimate(project()).failure, PoseFailure::kNone);
}

}  // namespace
}  // namespace tech_core_estimation
