#pragma once

#include "model_meta.hpp"

namespace tech_core_estimation_model
{

/** @brief Mapping from original image pixels to detector input pixels. */
struct Letterbox
{
  double gain;    ///< Isotropic resize scale.
  cv::Point pad;  ///< Integer left and top padding (pixel).
};

/** @brief Resize and center the BGR8 image on a square canvas, returning the forward mapping. */
Letterbox letterbox(const cv::Mat & bgr, int side, int pad_value, cv::Mat & padded);

/** @brief Decode the highest scoring column of a 5-row candidate matrix into original image bounds. */
ModelOutput bestBox(const cv::Mat & candidates, const Letterbox & mapping, cv::Size image_size,
                    double threshold);

/** @brief Warp a padded square around the detection and return the crop-to-image affine (pixel). */
cv::Matx23d squareCrop(const cv::Mat & bgr, cv::Rect2d box, int side, double padding, cv::Mat & crop);

/**
 * @brief Refine each heatmap maximum, restore its original image coordinates and apply the frozen threshold.
 * @see src/train/model_train/src/tech_core_model_train/litehrnet/model.py (local log-quadratic fit).
 */
void decodeKeypoints(const float * heatmaps, const float * logits, const ModelMeta & meta,
                     const cv::Matx23d & image_from_crop, cv::Size image_size, ModelOutput & output);

}  // namespace tech_core_estimation_model
