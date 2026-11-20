#include "conversions.hpp"

#include <gtest/gtest.h>

namespace tech_core_estimation_ros
{
namespace
{

TEST(Conversions, ReadsRowMajorRawIntrinsicsAndDistortion)
{
  sensor_msgs::msg::CameraInfo info;
  info.k = {301.0, 0.0, 161.5, 0.0, 310.0, 119.5, 0.0, 0.0, 1.0};
  info.d = {-0.2, 0.1, 0.002, -0.003, 0.01};
  const auto camera = intrinsics(info);
  EXPECT_EQ(camera.camera_matrix(0, 0), 301.0);
  EXPECT_EQ(camera.camera_matrix(1, 1), 310.0);
  EXPECT_EQ(camera.camera_matrix(0, 2), 161.5);
  EXPECT_EQ(camera.camera_matrix(1, 2), 119.5);
  EXPECT_EQ(camera.camera_matrix.row(2), Eigen::RowVector3d::UnitZ());
  EXPECT_EQ(camera.distortion, info.d);
}

TEST(Conversions, AppliesRoiBeforeBinningInEachAxis)
{
  sensor_msgs::msg::CameraInfo info;
  info.k = {300., 0., 162., 0., 360., 120., 0., 0., 1.};
  info.roi.x_offset = 22;
  info.roi.y_offset = 15;
  info.binning_x = 2;
  info.binning_y = 3;
  info.d = {-0.2, 0.1, 0.002, -0.003, 0.01};
  const auto camera = intrinsics(info);
  Eigen::Matrix3d expected;
  expected << 150., 0., 70., 0., 120., 35., 0., 0., 1.;
  EXPECT_EQ(camera.camera_matrix, expected);
  EXPECT_EQ(camera.distortion, info.d);
}

TEST(Conversions, SharesBgrAndReordersRgb)
{
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->height = image->width = 1;
  image->step = 3;
  image->encoding = "bgr8";
  image->data = {10, 20, 30};
  EXPECT_EQ(bgrImage(image)->image.data, image->data.data());
  image->encoding = "rgb8";
  EXPECT_EQ(bgrImage(image)->image.at<cv::Vec3b>(0, 0), cv::Vec3b(30, 20, 10));
}

TEST(Conversions, RejectsUnsupportedEncoding)
{
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->height = image->width = image->step = 1;
  image->data = {12};
  image->encoding = "mono8";
  EXPECT_THROW(bgrImage(image), cv_bridge::Exception);
}

TEST(Conversions, PreservesLandmarkOrderAndValidity)
{
  tech_core_estimation_model::ModelOutput output;
  for (std::size_t i = 0; i < output.pixels.size(); ++i) {
    output.pixels[i] = {i + 0.25, i + 0.5};
    output.valid[i] = i % 2 == 0;
  }
  const auto result = observation(output);
  ASSERT_EQ(result.pixels.size(), output.pixels.size());
  ASSERT_EQ(result.valid.size(), output.valid.size());
  for (std::size_t i = 0; i < output.pixels.size(); ++i) {
    EXPECT_EQ(result.pixels[i], Eigen::Vector2d(output.pixels[i].x, output.pixels[i].y));
    EXPECT_EQ(result.valid[i], output.valid[i]);
  }
}

TEST(Conversions, ComposesInWorldFrameAtImageTime)
{
  geometry_msgs::msg::TransformStamped world_from_camera;
  world_from_camera.header.frame_id = "base_link";
  world_from_camera.transform.translation.y = 2.0;
  world_from_camera.transform.rotation.z = std::sqrt(0.5);
  world_from_camera.transform.rotation.w = std::sqrt(0.5);
  Eigen::Isometry3d camera_from_core = Eigen::Isometry3d::Identity();
  camera_from_core.translation() = Eigen::Vector3d::UnitX();
  camera_from_core.linear() = Eigen::AngleAxisd(std::acos(0.0), Eigen::Vector3d::UnitX()).toRotationMatrix();
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 13;
  const auto pose = corePose(world_from_camera, camera_from_core, stamp);
  EXPECT_EQ(pose.header.frame_id, "base_link");
  EXPECT_EQ(pose.header.stamp, stamp);
  EXPECT_NEAR(pose.pose.position.x, 0.0, 1e-12);
  EXPECT_NEAR(pose.pose.position.y, 3.0, 1e-12);
  EXPECT_NEAR(pose.pose.position.z, 0.0, 1e-12);
  const auto & q = pose.pose.orientation;
  EXPECT_NEAR(Eigen::Quaterniond(q.w, q.x, q.y, q.z).angularDistance(Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5)),
              0.0, 1e-12);
}

TEST(Conversions, DrawsDetectionValidLandmarksAndDistortedReprojectionsOnCopy)
{
  cv_bridge::CvImage input(std_msgs::msg::Header{}, "bgr8", cv::Mat::zeros(80, 120, CV_8UC3));
  input.header.frame_id = "camera_optical";
  input.header.stamp.sec = 27;
  tech_core_estimation_model::ModelOutput output;
  output.detected = true;
  output.box = {4, 6, 78, 45};
  output.pixels[0] = {15, 12};
  output.valid[0] = true;
  output.pixels[1] = {30, 12};
  tech_core_estimation_model::Geometry geometry;
  geometry.points_m.fill({0.5, 0.0, 0.0});
  sensor_msgs::msg::CameraInfo info;
  info.k = {100.0, 0.0, 40.0, 0.0, 100.0, 30.0, 0.0, 0.0, 1.0};
  info.d = {0.8, 0.0, 0.0, 0.0};
  tech_core_estimation::PoseEstimate estimate;
  estimate.failure = tech_core_estimation::PoseFailure::kNone;
  estimate.camera_from_core.translation().z() = 1.0;
  const auto overlay = bgrImage(debugImage(input, output, geometry, intrinsics(info), estimate));
  EXPECT_EQ(overlay->header, input.header);
  EXPECT_EQ(overlay->image.at<cv::Vec3b>(6, 4), cv::Vec3b(0, 255, 0));
  EXPECT_EQ(overlay->image.at<cv::Vec3b>(12, 15), cv::Vec3b(0, 255, 255));
  EXPECT_EQ(overlay->image.at<cv::Vec3b>(12, 30), cv::Vec3b(0, 0, 0));
  EXPECT_EQ(overlay->image.at<cv::Vec3b>(30, 100), cv::Vec3b(255, 0, 255));
  EXPECT_EQ(cv::countNonZero(input.image.reshape(1)), 0);
}

}  // namespace
}  // namespace tech_core_estimation_ros
