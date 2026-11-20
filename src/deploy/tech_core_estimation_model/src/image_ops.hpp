#pragma once

#include "model_meta.hpp"

namespace tech_core_estimation_model
{

/** @brief 原图像素到检测器输入像素的映射。 */
struct Letterbox
{
  double gain;    ///< 各向同性缩放比例。
  cv::Point pad;  ///< 左侧与上侧的整数填充量，单位 pixel。
};

/** @brief 把 BGR8 图像缩放并居中放到正方形画布上，返回正向映射。 */
Letterbox letterbox(const cv::Mat & bgr, int side, int pad_value, cv::Mat & padded);

/** @brief 把 5 行候选矩阵中得分最高的一列解码为原图中的检测框。 */
ModelOutput bestBox(const cv::Mat & candidates, const Letterbox & mapping, cv::Size image_size,
                    double threshold);

/** @brief 围绕检测框截取扩展后的正方形区域，返回裁剪图到原图的仿射变换，单位 pixel。 */
cv::Matx23d squareCrop(const cv::Mat & bgr, cv::Rect2d box, int side, double padding, cv::Mat & crop);

/**
 * @brief 细化每张热图的极大值，还原到原图坐标并应用冻结的门限。
 * @see src/train/model_train/src/tech_core_model_train/litehrnet/model.py（局部对数二次拟合）。
 */
void decodeKeypoints(const float * heatmaps, const float * logits, const ModelMeta & meta,
                     const cv::Matx23d & image_from_crop, cv::Size image_size, ModelOutput & output);

}  // namespace tech_core_estimation_model
