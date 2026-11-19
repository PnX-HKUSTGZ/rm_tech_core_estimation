#include "conversions.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <image_transport/image_transport.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/distortion_models.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace tech_core_estimation_ros
{
namespace pnp = tech_core_estimation;
namespace model = tech_core_estimation_model;
using pnp::PoseFailure;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
using Clock = std::chrono::steady_clock;

/** @brief ROS component publishing the core pose from a calibrated camera stream. */
class EstimatorNode : public rclcpp::Node
{
public:
  /**
   * @brief Load the release and subscribe to the synchronized raw camera topics.
   * @param[in] options Component options and startup parameter overrides.
   * @throws std::invalid_argument The requested inference backend is unavailable.
   * @throws std::runtime_error The model release or device could not be loaded.
   */
  explicit EstimatorNode(const rclcpp::NodeOptions & options)
  : Node("tech_core_estimation", options), buffer_(get_clock()), listener_(buffer_, this), updater_(this)
  {
    rcl_interfaces::msg::ParameterDescriptor startup;
    startup.read_only = true;
    model::ModelOptions model_options;
    model_options.model_dir = declare_parameter<std::string>("model_dir", startup);
    const auto backend = declare_parameter<std::string>("backend", "openvino", startup);
    if (backend != "openvino") {
      throw std::invalid_argument("Unsupported backend '" + backend + "'. This build supports openvino.");
    }
    model_options.device = declare_parameter("device", model_options.device, startup);
    std::string error;
    model_ = model::TechCoreModel::load(model_options, error);
    if (!model_) {
      RCLCPP_ERROR(get_logger(), "Model load failed: %s", error.c_str());
      throw std::runtime_error(error);
    }
    for (const auto & point : model_->geometry().points_m) {
      keypoints_.points.emplace_back(point.x, point.y, point.z);
    }

    world_frame_ = declare_parameter<std::string>("world_frame", "base_link", startup);
    core_frame_ = declare_parameter<std::string>("core_frame", "tech_core", startup);
    tf_timeout_ = declare_parameter("tf_timeout", 0.05, startup);
    config_.reprojection_threshold = declare_parameter(
      "pnp.reprojection_threshold", config_.reprojection_threshold, startup);
    config_.ransac_iterations = declare_parameter("pnp.ransac_iterations", config_.ransac_iterations, startup);
    config_.ransac_confidence = declare_parameter("pnp.ransac_confidence", config_.ransac_confidence, startup);
    config_.local_optimization_steps = declare_parameter(
      "pnp.local_optimization_steps", config_.local_optimization_steps, startup);
    config_.huber_refinement = declare_parameter("pnp.huber_refinement", config_.huber_refinement, startup);
    if (declare_parameter("publish_tf", true, startup)) {
      broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/core_pose", 10);
    debug_pub_ = create_publisher<Image>("~/debug_image", 1);
    updater_.setHardwareID(model_options.device);
    updater_.add("Pose estimation", this, &EstimatorNode::diagnostics);
    camera_sub_ = image_transport::create_camera_subscription(this, "image",
      std::bind(&EstimatorNode::onFrame, this, std::placeholders::_1, std::placeholders::_2),
      "raw", rmw_qos_profile_sensor_data);
  }

private:
  /** @brief Process one raw image and its time-matched calibration in the camera optical frame. */
  void onFrame(const Image::ConstSharedPtr & image, const CameraInfo::ConstSharedPtr & info)
  {
    ++frames_;
    namespace distortion = sensor_msgs::distortion_models;
    if (!info->distortion_model.empty() && info->distortion_model != distortion::PLUMB_BOB &&
        info->distortion_model != distortion::RATIONAL_POLYNOMIAL) {
      ++invalid_calibration_;
      return;
    }
    if (!calibration_ || info->k != calibration_->k || info->d != calibration_->d) {
      calibration_ = info;
      std::string error;
      estimator_ = pnp::PnPEstimator::create(keypoints_, intrinsics(*info), config_, error);
      if (!estimator_) {
        RCLCPP_ERROR(get_logger(), "Cannot create PnP estimator: %s", error.c_str());
      }
    }
    if (!estimator_) {
      ++invalid_calibration_;
      return;
    }
    cv_bridge::CvImageConstPtr bgr;
    try {
      bgr = bgrImage(image);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", error.what());
      ++invalid_image_;
      return;
    }
    const auto start = Clock::now();
    const auto output = model_->infer(bgr->image);
    const auto inferred = Clock::now();
    inference_s_ = std::chrono::duration<double>(inferred - start).count();
    pnp::PoseEstimate estimate;
    if (output.detected) {
      estimate = estimator_->estimate(observation(output));
      solve_s_ = std::chrono::duration<double>(Clock::now() - inferred).count();
      if (estimate.failure == PoseFailure::kNone) {
        publishPose(image->header, estimate.camera_from_core);
      } else {
        ++pnp_failures_[estimate.failure];
      }
    } else {
      ++no_detection_;
      solve_s_ = 0.0;
    }
    if (debug_pub_->get_subscription_count() > 0) {
      debug_pub_->publish(debugImage(*bgr, output, model_->geometry(), *info, estimate));
    }
  }

  /** @brief Look up the camera pose at exposure time and publish the resulting world-frame pose. */
  void publishPose(const std_msgs::msg::Header & header, const Eigen::Isometry3d & camera_from_core)
  {
    geometry_msgs::msg::TransformStamped world_from_camera;
    try {
      world_from_camera = buffer_.lookupTransform(world_frame_, header.frame_id, header.stamp,
                                                  rclcpp::Duration::from_seconds(tf_timeout_));
    } catch (const tf2::TransformException &) {
      ++tf_timeout_count_;
      return;
    }
    const auto pose = corePose(world_from_camera, camera_from_core, header.stamp);
    pose_pub_->publish(pose);
    ++published_;
    if (broadcaster_) {
      geometry_msgs::msg::TransformStamped world_from_core;
      world_from_core.header = pose.header;
      world_from_core.child_frame_id = core_frame_;
      world_from_core.transform.translation.x = pose.pose.position.x;
      world_from_core.transform.translation.y = pose.pose.position.y;
      world_from_core.transform.translation.z = pose.pose.position.z;
      world_from_core.transform.rotation = pose.pose.orientation;
      broadcaster_->sendTransform(world_from_core);
    }
  }

  /** @brief Report rates over the last diagnostic interval, cumulative outcomes and last stage durations. */
  void diagnostics(diagnostic_updater::DiagnosticStatusWrapper & status)
  {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_report_).count();
    const auto received = frames_ - previous_frames_;
    using Status = diagnostic_msgs::msg::DiagnosticStatus;
    if (received == 0) {
      status.summary(Status::STALE, "No synchronized images");
    } else if (published_ == previous_published_) {
      status.summary(Status::WARN, "No poses published");
    } else {
      status.summary(Status::OK, "Publishing poses");
    }
    status.add("frame_rate_hz", received / elapsed);
    status.add("published", published_);
    status.add("no_detection", no_detection_);
    status.add("invalid_calibration", invalid_calibration_);
    status.add("invalid_image", invalid_image_);
    status.add("tf_timeout", tf_timeout_count_);
    status.add("too_few_points", pnp_failures_[PoseFailure::kTooFewPoints]);
    status.add("degenerate", pnp_failures_[PoseFailure::kDegenerate]);
    status.add("no_consensus", pnp_failures_[PoseFailure::kNoConsensus]);
    status.add("no_positive_depth", pnp_failures_[PoseFailure::kNoPositiveDepth]);
    status.add("inference_s", inference_s_);
    status.add("solve_s", solve_s_);
    previous_frames_ = frames_;
    previous_published_ = published_;
    last_report_ = now;
  }

  std::unique_ptr<model::TechCoreModel> model_;  ///< Loaded release for this stream.
  pnp::KeypointModel keypoints_;  ///< Release points in the core frame (m).
  pnp::PnPConfig config_;  ///< Startup PnP settings.
  CameraInfo::ConstSharedPtr calibration_;  ///< Last K and D submitted to the solver factory.
  std::optional<pnp::PnPEstimator> estimator_;  ///< Solver for that calibration, empty on creation failure.
  std::string world_frame_;  ///< Arm base TF frame.
  std::string core_frame_;  ///< Output TF child frame.
  double tf_timeout_;  ///< Maximum wait for the camera transform (s).
  tf2_ros::Buffer buffer_;  ///< Time-indexed transforms.
  tf2_ros::TransformListener listener_;  ///< Receives TF while the camera callback waits.
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;  ///< Enabled by publish_tf.
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;  ///< World-frame core poses.
  rclcpp::Publisher<Image>::SharedPtr debug_pub_;  ///< Annotated camera images.
  image_transport::CameraSubscriber camera_sub_;  ///< Synchronized image and intrinsics.
  diagnostic_updater::Updater updater_;  ///< Periodic diagnostics on the default callback group.
  std::uint64_t frames_ = 0;  ///< Synchronized input pairs.
  std::uint64_t published_ = 0;  ///< Published world-frame poses.
  std::uint64_t no_detection_ = 0;  ///< Images without a core detection.
  std::uint64_t invalid_calibration_ = 0;  ///< Frames without a usable pinhole calibration.
  std::uint64_t invalid_image_ = 0;  ///< Frames rejected by image conversion.
  std::uint64_t tf_timeout_count_ = 0;  ///< Successful estimates without a camera transform.
  std::map<PoseFailure, std::uint64_t> pnp_failures_;  ///< Failed estimates per solver reason.
  double inference_s_ = 0.0;  ///< Last inference duration (s).
  double solve_s_ = 0.0;  ///< Last detected frame's solve duration, zero on a miss (s).
  Clock::time_point last_report_ = Clock::now();  ///< Start of the diagnostic interval.
  std::uint64_t previous_frames_ = 0;  ///< Input count at the previous report.
  std::uint64_t previous_published_ = 0;  ///< Pose count at the previous report.
};

}  // namespace tech_core_estimation_ros

RCLCPP_COMPONENTS_REGISTER_NODE(tech_core_estimation_ros::EstimatorNode)
