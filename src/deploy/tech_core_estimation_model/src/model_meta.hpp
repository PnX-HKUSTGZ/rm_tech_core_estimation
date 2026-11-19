#pragma once

#include "tech_core_estimation_model/tech_core_model.hpp"

#include <filesystem>

namespace tech_core_estimation_model
{

/** @brief Schema v1 preprocessing and decoding parameters shared by both stages. */
struct ModelMeta
{
  Geometry geometry;            ///< Landmark order and core-frame positions (m) with their fingerprint.
  int detector_size;            ///< Square detector input side (pixel).
  int pad_value;                ///< Letterbox padding intensity on the byte scale.
  double box_threshold;         ///< Minimum accepted detector score.
  int crop_size;                ///< Square keypoint input side (pixel).
  double padding;               ///< Crop side divided by the longer box dimension.
  int heatmap_size;             ///< Square heatmap side (pixel).
  double keypoint_threshold;    ///< Minimum valid landmark confidence.
};

/** @brief Parse schema v1 model_meta.yaml; the preprocessing it describes is implemented in image_ops. */
ModelMeta readModelMeta(const std::filesystem::path & file);

}  // namespace tech_core_estimation_model
