#include "model_meta.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace tech_core_estimation_model
{

ModelMeta readModelMeta(const std::filesystem::path & file)
{
  const auto root = YAML::LoadFile(file.string());
  if (root["schema_version"].as<int>() != 1) {
    throw std::runtime_error("unsupported model_meta.yaml schema_version (expected 1)");
  }
  const auto node = root["geometry"];
  if (node["names"].size() != kKeypointCount || node["points_m"].size() != kKeypointCount) {
    throw std::runtime_error("geometry must contain 16 names and points_m");
  }
  ModelMeta meta;
  auto & geometry = meta.geometry;
  geometry.profile = node["profile"].as<std::string>();
  geometry.names = node["names"].as<decltype(geometry.names)>();
  geometry.groups = node["groups"].as<decltype(geometry.groups)>();
  for (std::size_t i = 0; i < kKeypointCount; ++i) {
    const auto xyz = node["points_m"][i].as<std::array<double, 3>>();
    geometry.points_m[i] = {xyz[0], xyz[1], xyz[2]};
  }
  geometry.sha256 = node["sha256"].as<std::string>();
  const auto detector = root["detector"];
  meta.detector_size = detector["input_size"].as<int>();
  meta.pad_value = detector["letterbox"]["pad_value"].as<int>();
  meta.box_threshold = detector["threshold"].as<double>();
  const auto keypoints = root["keypoints"];
  meta.crop_size = keypoints["crop_size"].as<int>();
  meta.heatmap_size = keypoints["heatmap_size"].as<int>();
  meta.padding = keypoints["padding"].as<double>();
  meta.keypoint_threshold = keypoints["threshold"].as<double>();
  return meta;
}

}  // namespace tech_core_estimation_model
