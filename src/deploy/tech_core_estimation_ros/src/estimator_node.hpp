/**
 * @file estimator_node.hpp
 * @brief 由已标定相机图像流发布科技核心世界系位姿的 ROS 组件声明。
 */
#pragma once

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <image_transport/camera_subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tech_core_estimation/pnp_estimator.hpp>
#include <tech_core_estimation_model/tech_core_model.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <Eigen/Geometry>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace tech_core_estimation_ros
{

/** @brief 由已标定相机图像流发布核心位姿的 ROS 组件。 */
class EstimatorNode : public rclcpp::Node
{
public:
  /**
   * @brief 加载发布目录并订阅同步的原始相机话题。
   * @param[in] options 组件选项及启动参数覆盖。
   * @throws std::invalid_argument 请求的推理后端不可用。
   * @throws std::runtime_error 无法加载模型发布目录或推理设备。
   */
  explicit EstimatorNode(const rclcpp::NodeOptions & options);

private:
  using CameraInfo = sensor_msgs::msg::CameraInfo;  ///< 相机标定消息。
  using Image = sensor_msgs::msg::Image;            ///< 相机图像消息。
  using Clock = std::chrono::steady_clock;          ///< 计时与诊断周期使用的单调时钟。

  /**
   * @brief 处理一帧原始图像及其时间匹配的相机光学系标定。
   * @param[in] image 原始 RGB8 或 BGR8 相机图像，坐标单位 pixel。
   * @param[in] info 同一时间戳的原始针孔标定，焦距与主点单位 pixel。
   * @note 零时间戳帧在推理前丢弃，计入 invalid_stamp。
   */
  void onFrame(const Image::ConstSharedPtr & image, const CameraInfo::ConstSharedPtr & info);

  /**
   * @brief 查询曝光时刻的相机位姿，并发布得到的世界系位姿。
   * @param[in] header 图像消息头，frame_id 为相机光学系，stamp 为曝光时刻。
   * @param[in] camera_from_core 核心系在相机光学系中的位姿，平移单位 m。
   */
  void publishPose(const std_msgs::msg::Header & header, const Eigen::Isometry3d & camera_from_core);

  /**
   * @brief 报告上一诊断周期内的频率、累计结果计数及最近一次各阶段耗时。
   * @param[out] status 写入摘要与键值（频率单位 Hz，耗时单位 s）的诊断状态。
   */
  void diagnostics(diagnostic_updater::DiagnosticStatusWrapper & status);

  std::unique_ptr<tech_core_estimation_model::TechCoreModel> model_;  ///< 本图像流加载的发布目录。
  tech_core_estimation::KeypointModel keypoints_;  ///< 发布目录中的核心系关键点，单位 m。
  tech_core_estimation::PnPConfig config_;  ///< 启动时读取的 PnP 参数。
  std::optional<tech_core_estimation::CameraIntrinsics> calibration_;  ///< 最近提交给解算器的有效内参与畸变系数。
  std::optional<tech_core_estimation::PnPEstimator> estimator_;  ///< 对应该标定的解算器；创建失败时为空。
  std::string world_frame_;  ///< 机械臂基座（世界系）的 TF frame。
  std::string core_frame_;  ///< 输出 TF 的子 frame。
  double tf_timeout_;  ///< 等待相机变换的最长时间，单位 s。
  tf2_ros::Buffer buffer_;  ///< 按时间索引的变换缓存。
  tf2_ros::TransformListener listener_;  ///< 在相机回调等待期间接收 TF。
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;  ///< 由 publish_tf 参数启用。
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;  ///< 世界系中的核心位姿。
  rclcpp::Publisher<Image>::SharedPtr debug_pub_;  ///< 标注后的相机图像。
  image_transport::CameraSubscriber camera_sub_;  ///< 同步后的图像与内参。
  diagnostic_updater::Updater updater_;  ///< 在默认回调组上运行的周期诊断。
  std::uint64_t frames_ = 0;  ///< 同步输入对计数。
  std::uint64_t published_ = 0;  ///< 已发布的世界系位姿计数。
  std::uint64_t no_detection_ = 0;  ///< 未检出核心的图像计数。
  std::uint64_t invalid_stamp_ = 0;  ///< 图像采集时间戳为零的帧计数。
  std::uint64_t invalid_calibration_ = 0;  ///< 无可用针孔标定的帧计数。
  std::uint64_t invalid_image_ = 0;  ///< 图像转换失败的帧计数。
  std::uint64_t tf_timeout_count_ = 0;  ///< 解算成功但缺少相机变换的计数。
  std::map<tech_core_estimation::PoseFailure, std::uint64_t> pnp_failures_;  ///< 按解算失败原因分类的计数。
  double inference_s_ = 0.0;  ///< 最近一次推理耗时，单位 s。
  double solve_s_ = 0.0;  ///< 最近一帧的解算耗时，未检出时为零，单位 s。
  Clock::time_point last_report_ = Clock::now();  ///< 诊断周期起始时刻。
  std::uint64_t previous_frames_ = 0;  ///< 上次报告时的输入计数。
  std::uint64_t previous_published_ = 0;  ///< 上次报告时的位姿计数。
};

}  // namespace tech_core_estimation_ros
