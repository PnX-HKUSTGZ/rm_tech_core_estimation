/**
 * @file tech_core_model.hpp
 * @brief 科技核心单帧检测与关键点推理接口。
 */
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace tech_core_estimation_model
{

inline constexpr std::size_t kKeypointCount = 16;  ///< 关键点数量，按发布目录中的顺序。

/** @brief 两个推理阶段使用的推理运行时。 */
enum class Backend
{
  kOpenVino  ///< OpenVINO 加载 ONNX 运行。
};

/** @brief 加载模型发布目录的选项。 */
struct ModelOptions
{
  std::string model_dir;                 ///< model_train 导出的发布目录。
  Backend backend = Backend::kOpenVino;  ///< 推理运行时。
  std::string device = "CPU";            ///< OpenVINO 设备名，如 CPU、GPU 或 AUTO。
};

/** @brief 两个模型共用的有序关键点几何。 */
struct Geometry
{
  std::string profile;                                 ///< 几何配置名。
  std::array<std::string, kKeypointCount> names;       ///< 关键点标识，按输出顺序。
  std::array<std::string, kKeypointCount> groups;      ///< 每个关键点所属的几何分组。
  std::array<cv::Point3d, kKeypointCount> points_m;    ///< 关键点在核心系中的坐标，单位 m。
  std::string sha256;                                  ///< 规范化几何数据的 SHA-256。
};

/** @brief 原图中的检测结果与有序关键点，x 向右、y 向下。 */
struct ModelOutput
{
  bool detected = false;                             ///< 是否检出；其余字段仅在检出时有意义。
  cv::Rect2d box;                                    ///< 裁剪到图像内的检测框，右下端不含，单位 pixel。
  double score = 0.0;                                ///< 检测置信度，取值 [0, 1]。
  std::array<cv::Point2d, kKeypointCount> pixels{};  ///< 关键点坐标，单位 pixel。
  std::array<double, kKeypointCount> confidence{};   ///< 逐点置信度，取值 [0, 1]。
  std::array<bool, kKeypointCount> valid{};          ///< 坐标有限、位于图像内且达到发布目录门限时为真。
};

/** @brief 按模型发布目录的预处理与门限执行两阶段图像推理。 */
class TechCoreModel
{
public:
  /**
   * @brief 校验发布目录的几何并编译两个推理模型。
   * @param[in] options 发布目录、推理运行时与推理设备。
   * @param[out] error 失败原因；成功时清空。
   * @return 成功时返回模型实例；发布目录或设备出错时返回 nullptr。
   */
  static std::unique_ptr<TechCoreModel> load(const ModelOptions & options, std::string & error);

  /** @brief 释放模型及其推理请求。 */
  ~TechCoreModel();

  /**
   * @brief 在一帧图像中检测核心并定位其关键点。
   * @param[in] bgr 非空 BGR8 图像，原图坐标，单位 pixel。
   * @return 输入图像中的检测框、置信度与关键点坐标，单位 pixel。
   * @note 推理不是线程安全的；每个推理线程使用一个实例。
   */
  ModelOutput infer(const cv::Mat & bgr);

  /**
   * @brief 获取发布目录的关键点顺序与核心系几何。
   * @return 在本模型生命周期内有效的引用，坐标单位 m。
   */
  const Geometry & geometry() const;

private:
  /** @brief 已加载的运行时状态。 */
  struct Impl;

  /**
   * @brief 接管已校验的发布目录及其推理请求。
   * @param[in] impl 已初始化的推理状态。
   */
  explicit TechCoreModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;  ///< 每个实例独有的推理状态。
};

}  // namespace tech_core_estimation_model
