/**
 * @file conversions.hpp
 * @brief 位姿估计组件使用的 ROS 相机消息与模型输出之间的转换。
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
 * @brief 以模型使用的 BGR 通道顺序读取同步后的相机图像。
 * @param[in] image 原始 RGB8 或 BGR8 相机图像，坐标轴向右、向下。
 * @return 保留输入消息头的 BGR8 图像；输入为 BGR8 时共享内存。
 * @throws cv_bridge::Exception 编码不受支持或图像数据格式错误。
 */
cv_bridge::CvImageConstPtr bgrImage(const sensor_msgs::msg::Image::ConstSharedPtr & image);

/**
 * @brief 根据 ROI 与像素合并因子换算所收原始图像的针孔内参。
 * @param[in] info 相机光学系全分辨率标定、ROI 偏移（单位 pixel）及各轴合并因子（0 视作 1）。
 * @return 所收图像的焦距与主点（单位 pixel）及原标定的畸变系数。
 */
tech_core_estimation::CameraIntrinsics intrinsics(const sensor_msgs::msg::CameraInfo & info);

/**
 * @brief 按发布目录顺序转换关键点坐标与有效标记。
 * @param[in] output 检出的关键点，原图坐标，单位 pixel。
 * @return 同一图像坐标下的 PnP 观测，单位 pixel。
 */
tech_core_estimation::Observation observation(const tech_core_estimation_model::ModelOutput & output);

/**
 * @brief 合成图像采集时刻核心系在世界系中的位姿。
 * @param[in] world_from_camera 相机光学系在世界系中的位姿，平移单位 m。
 * @param[in] camera_from_core 核心系在相机光学系中的位姿，平移单位 m。
 * @param[in] stamp 相机图像采集时刻（ROS 时间）。
 * @return 带时间戳的核心系在世界系中的位姿，平移单位 m。
 */
geometry_msgs::msg::PoseStamped corePose(const geometry_msgs::msg::TransformStamped & world_from_camera,
                                         const Eigen::Isometry3d & camera_from_core,
                                         const builtin_interfaces::msg::Time & stamp);

/**
 * @brief 在图像副本上绘制检测框、有效关键点及可用的模型重投影。
 * @param[in] bgr BGR8 相机图像，原图坐标，单位 pixel。
 * @param[in] output 检测框与关键点观测，单位 pixel。
 * @param[in] geometry 发布目录中的核心系关键点，单位 m。
 * @param[in] camera PnP 使用的相机光学系标定，焦距与主点以所收图像的 pixel 为单位。
 * @param[in] estimate 相机光学系中的 PnP 结果，平移单位 m。
 * @return 带输入时间戳与相机光学系 frame_id 的 BGR8 调试图像。
 */
sensor_msgs::msg::Image::UniquePtr debugImage(const cv_bridge::CvImage & bgr,
                                             const tech_core_estimation_model::ModelOutput & output,
                                             const tech_core_estimation_model::Geometry & geometry,
                                             const tech_core_estimation::CameraIntrinsics & camera,
                                             const tech_core_estimation::PoseEstimate & estimate);

}  // namespace tech_core_estimation_ros
