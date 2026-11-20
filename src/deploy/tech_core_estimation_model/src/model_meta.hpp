#pragma once

#include "tech_core_estimation_model/tech_core_model.hpp"

#include <filesystem>

namespace tech_core_estimation_model
{

/** @brief schema v1 中两阶段共用的预处理与解码参数。 */
struct ModelMeta
{
  Geometry geometry;            ///< 关键点顺序、核心系坐标（单位 m）及其指纹。
  int detector_size;            ///< 检测器正方形输入边长，单位 pixel。
  int pad_value;                ///< letterbox 填充灰度值，0–255 字节刻度。
  double box_threshold;         ///< 检测得分下限。
  int crop_size;                ///< 关键点网络正方形输入边长，单位 pixel。
  double padding;               ///< 裁剪边长与检测框长边之比。
  int heatmap_size;             ///< 正方形热图边长，单位 pixel。
  double keypoint_threshold;    ///< 关键点有效的最低置信度。
};

/** @brief 解析 schema v1 的 model_meta.yaml；其描述的预处理在 image_ops 中实现。 */
ModelMeta readModelMeta(const std::filesystem::path & file);

}  // namespace tech_core_estimation_model
