#include "conversions.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace tech_core_estimation_ros
{

cv_bridge::CvImageConstPtr bgrImage(const sensor_msgs::msg::Image::ConstSharedPtr & image)
{
  namespace encoding = sensor_msgs::image_encodings;
  if (image->encoding != encoding::BGR8 && image->encoding != encoding::RGB8) {
    throw cv_bridge::Exception("Expected bgr8 or rgb8, received " + image->encoding);
  }
  return cv_bridge::toCvShare(image, encoding::BGR8);
}

tech_core_estimation::CameraIntrinsics intrinsics(const sensor_msgs::msg::CameraInfo & info)
{
  Eigen::Matrix3d matrix = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(info.k.data());
  const double bx = info.binning_x ? info.binning_x : 1;
  const double by = info.binning_y ? info.binning_y : 1;
  matrix(0, 0) /= bx;
  matrix(1, 1) /= by;
  matrix(0, 2) = (matrix(0, 2) - info.roi.x_offset) / bx;
  matrix(1, 2) = (matrix(1, 2) - info.roi.y_offset) / by;
  return {matrix, info.d};
}

tech_core_estimation::Observation observation(const tech_core_estimation_model::ModelOutput & output)
{
  tech_core_estimation::Observation result;
  result.pixels.reserve(output.pixels.size());
  for (const auto & pixel : output.pixels) {
    result.pixels.emplace_back(pixel.x, pixel.y);
  }
  result.valid.assign(output.valid.begin(), output.valid.end());
  return result;
}

geometry_msgs::msg::PoseStamped corePose(const geometry_msgs::msg::TransformStamped & world_from_camera,
                                         const Eigen::Isometry3d & camera_from_core,
                                         const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = world_from_camera.header.frame_id;
  pose.header.stamp = stamp;
  pose.pose = tf2::toMsg(tf2::transformToEigen(world_from_camera) * camera_from_core);
  return pose;
}

sensor_msgs::msg::Image::UniquePtr debugImage(const cv_bridge::CvImage & bgr,
                                             const tech_core_estimation_model::ModelOutput & output,
                                             const tech_core_estimation_model::Geometry & geometry,
                                             const tech_core_estimation::CameraIntrinsics & camera,
                                             const tech_core_estimation::PoseEstimate & estimate)
{
  auto image = std::make_unique<sensor_msgs::msg::Image>();
  bgr.toImageMsg(*image);
  if (!output.detected) {
    return image;
  }
  cv::Mat overlay(bgr.image.size(), CV_8UC3, image->data.data(), image->step);
  cv::rectangle(overlay, output.box, {0, 255, 0});
  for (std::size_t i = 0; i < output.pixels.size(); ++i) {
    if (output.valid[i]) {
      cv::circle(overlay, output.pixels[i], 2, {0, 255, 255}, cv::FILLED);
    }
  }
  if (estimate.failure == tech_core_estimation::PoseFailure::kNone) {
    cv::Mat rotation, translation, matrix;
    cv::eigen2cv(Eigen::Matrix3d(estimate.camera_from_core.linear()), rotation);
    cv::eigen2cv(Eigen::Vector3d(estimate.camera_from_core.translation()), translation);
    cv::eigen2cv(camera.camera_matrix, matrix);
    std::vector<cv::Point2d> projected;
    cv::projectPoints(geometry.points_m, rotation, translation, matrix, camera.distortion, projected);
    for (const auto & pixel : projected) {
      cv::drawMarker(overlay, pixel, {255, 0, 255}, cv::MARKER_CROSS, 5);
    }
  }
  return image;
}

}  // namespace tech_core_estimation_ros
