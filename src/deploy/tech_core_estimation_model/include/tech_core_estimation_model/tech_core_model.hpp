/**
 * @file tech_core_model.hpp
 * @brief Single-frame technology core detection and landmark inference.
 */
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace tech_core_estimation_model
{

inline constexpr std::size_t kKeypointCount = 16;  ///< Landmarks in release order.

/** @brief Inference runtime for both stages. */
enum class Backend
{
  kOpenVino  ///< OpenVINO ONNX runtime.
};

/** @brief Options for loading a model release. */
struct ModelOptions
{
  std::string model_dir;                 ///< Directory exported by model_train.
  Backend backend = Backend::kOpenVino;  ///< Inference runtime.
  std::string device = "CPU";            ///< OpenVINO device name, e.g. CPU, GPU or AUTO.
};

/** @brief Ordered landmark geometry shared by the two models. */
struct Geometry
{
  std::string profile;                                 ///< Geometry profile name.
  std::array<std::string, kKeypointCount> names;       ///< Landmark identifiers in output order.
  std::array<std::string, kKeypointCount> groups;      ///< Geometry group of each landmark.
  std::array<cv::Point3d, kKeypointCount> points_m;    ///< Landmark positions in the core frame (m).
  std::string sha256;                                  ///< SHA-256 of the canonical geometry payload.
};

/** @brief Detection and ordered landmarks in the original image, x right and y down. */
struct ModelOutput
{
  bool detected = false;                             ///< Remaining fields are meaningful on a hit.
  cv::Rect2d box;                                    ///< Clipped detection box with exclusive end (pixel).
  double score = 0.0;                                ///< Detection confidence in [0, 1].
  std::array<cv::Point2d, kKeypointCount> pixels{};  ///< Landmark locations (pixel).
  std::array<double, kKeypointCount> confidence{};   ///< Per-landmark confidence in [0, 1].
  std::array<bool, kKeypointCount> valid{};          ///< Finite, in-image point meeting the release threshold.
};

/** @brief Two-stage image inference using the preprocessing and thresholds of a model release. */
class TechCoreModel
{
public:
  /**
   * @brief Check the release geometry and compile both models for inference.
   * @param[in] options Release directory, runtime and inference device.
   * @param[out] error Failure reason, cleared on success.
   * @return Model instance on success, nullptr for a release or device error.
   */
  static std::unique_ptr<TechCoreModel> load(const ModelOptions & options, std::string & error);

  /** @brief Release the model and its inference requests. */
  ~TechCoreModel();

  /**
   * @brief Detect the core and localize its landmarks in one frame.
   * @param[in] bgr Nonempty BGR8 image in original image coordinates (pixel).
   * @return Detection, confidence and landmark coordinates in the input image (pixel).
   * @note Inference is not thread-safe. Use one instance per inference thread.
   */
  ModelOutput infer(const cv::Mat & bgr);

  /**
   * @brief Access the release's landmark order and core-frame geometry.
   * @return Reference valid for the lifetime of this model, with positions in m.
   */
  const Geometry & geometry() const;

private:
  /** @brief Loaded runtime state. */
  struct Impl;

  /**
   * @brief Take ownership of a validated release and its requests.
   * @param[in] impl Initialized inference state.
   */
  explicit TechCoreModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;  ///< Per-instance inference state.
};

}  // namespace tech_core_estimation_model
