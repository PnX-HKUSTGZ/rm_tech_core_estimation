/**
 * @file conversions.hpp
 * @brief ROS camera messages and model outputs used by the estimator component.
 */
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tech_core_estimation/pnp_estimator.hpp>
#include <tech_core_estimation_model/tech_core_model.hpp>

namespace tech_core_estimation_ros
{

/**
 * @brief Read a synchronized camera image in the model's BGR channel order.
 * @param[in] image Raw RGB8 or BGR8 camera image, axes right and down.
 * @return BGR8 image retaining the input header and sharing storage for BGR8.
 * @throws cv_bridge::Exception Unsupported encoding or malformed image data.
 */
cv_bridge::CvImageConstPtr bgrImage(const sensor_msgs::msg::Image::ConstSharedPtr & image);

/**
 * @brief Read raw pinhole intrinsics for the camera optical frame.
 * @param[in] info Row-major camera matrix (focal lengths and principal point in pixel) and distortion.
 * @return PnP intrinsics in the original image coordinates (pixel).
 */
tech_core_estimation::CameraIntrinsics intrinsics(const sensor_msgs::msg::CameraInfo & info);

/**
 * @brief Map landmark locations and validity flags in release order.
 * @param[in] output Detected landmarks in original image coordinates (pixel).
 * @return PnP observations in the same image coordinates (pixel).
 */
tech_core_estimation::Observation observation(const tech_core_estimation_model::ModelOutput & output);

/**
 * @brief Compose the core pose in the world frame at image acquisition time.
 * @param[in] world_from_camera Camera optical frame in the world frame, translation in m.
 * @param[in] camera_from_core Core frame in the camera optical frame, translation in m.
 * @param[in] stamp Camera image acquisition time (ROS time).
 * @return Stamped pose of the core in the world frame, translation in m.
 */
geometry_msgs::msg::PoseStamped corePose(const geometry_msgs::msg::TransformStamped & world_from_camera,
                                         const Eigen::Isometry3d & camera_from_core,
                                         const builtin_interfaces::msg::Time & stamp);

/**
 * @brief Draw a detection, its valid landmarks and the available model reprojections on a copy.
 * @param[in] bgr BGR8 camera image in original image coordinates (pixel).
 * @param[in] output Detection and landmark observations (pixel).
 * @param[in] geometry Release landmarks in the core frame (m).
 * @param[in] info Raw camera calibration, focal lengths and principal point in pixel.
 * @param[in] estimate PnP result in the camera optical frame, translation in m.
 * @return BGR8 debug image with the input stamp and optical frame.
 */
sensor_msgs::msg::Image::UniquePtr debugImage(const cv_bridge::CvImage & bgr,
                                             const tech_core_estimation_model::ModelOutput & output,
                                             const tech_core_estimation_model::Geometry & geometry,
                                             const sensor_msgs::msg::CameraInfo & info,
                                             const tech_core_estimation::PoseEstimate & estimate);

}  // namespace tech_core_estimation_ros
