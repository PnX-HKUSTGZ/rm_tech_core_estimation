#include "image_ops.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace tech_core_estimation_model
{

Letterbox letterbox(const cv::Mat & bgr, int side, int pad_value, cv::Mat & padded)
{
  const double gain = static_cast<double>(side) / std::max(bgr.cols, bgr.rows);
  const cv::Size resized(std::max(1, cvRound(bgr.cols * gain)),
                         std::max(1, cvRound(bgr.rows * gain)));
  const cv::Point pad((side - resized.width) / 2, (side - resized.height) / 2);
  padded.create(side, side, CV_8UC3);
  padded.setTo(cv::Scalar::all(pad_value));
  cv::resize(bgr, padded(cv::Rect(pad, resized)), resized, 0, 0, cv::INTER_LINEAR);
  return {gain, pad};
}

ModelOutput bestBox(const cv::Mat & candidates, const Letterbox & mapping, cv::Size image_size,
                    double threshold)
{
  const float * scores = candidates.ptr<float>(4);
  const int best = static_cast<int>(std::max_element(scores, scores + candidates.cols) - scores);
  if (scores[best] < threshold) {
    return {};
  }
  const float x = candidates.at<float>(0, best), y = candidates.at<float>(1, best);
  const float half_width = candidates.at<float>(2, best) / 2, half_height = candidates.at<float>(3, best) / 2;
  const auto unpad = [&](float value, int pad, int end) {
    return std::clamp((static_cast<double>(value) - pad) / mapping.gain, 0.0, static_cast<double>(end));
  };
  const double left = unpad(x - half_width, mapping.pad.x, image_size.width);
  const double right = unpad(x + half_width, mapping.pad.x, image_size.width);
  const double top = unpad(y - half_height, mapping.pad.y, image_size.height);
  const double bottom = unpad(y + half_height, mapping.pad.y, image_size.height);
  if (right <= left || bottom <= top) {
    return {};
  }
  ModelOutput output;
  output.detected = true;
  output.box = {left, top, right - left, bottom - top};
  output.score = scores[best];
  return output;
}

cv::Matx23d squareCrop(const cv::Mat & bgr, cv::Rect2d box, int side, double padding, cv::Mat & crop)
{
  const double step = padding * std::max(box.width, box.height) / side;
  const double half_span = step * (side - 1) / 2;
  const cv::Matx23d image_from_crop(step, 0, box.x + box.width / 2 - half_span,
                                    0, step, box.y + box.height / 2 - half_span);
  // The training crop uses full bilinear weights. OpenCV 4 warpAffine rounds positions to 1/32 pixel.
  const auto sample = [&](int x, int y) -> cv::Vec3d {
    if (x >= 0 && x < bgr.cols && y >= 0 && y < bgr.rows) {
      return bgr.at<cv::Vec3b>(y, x);
    }
    return {};
  };
  crop.create(side, side, CV_8UC3);
  for (int v = 0; v < side; ++v) {
    auto * row = crop.ptr<cv::Vec3b>(v);
    const double y = image_from_crop(1, 1) * v + image_from_crop(1, 2);
    const int top = cvFloor(y);
    const double dy = y - top;
    for (int u = 0; u < side; ++u) {
      const double x = image_from_crop(0, 0) * u + image_from_crop(0, 2);
      const int left = cvFloor(x);
      const double dx = x - left;
      const auto above = (1 - dx) * sample(left, top) + dx * sample(left + 1, top);
      const auto below = (1 - dx) * sample(left, top + 1) + dx * sample(left + 1, top + 1);
      row[u] = (1 - dy) * above + dy * below;
    }
  }
  return image_from_crop;
}

namespace
{

/** @brief Fit the stationary point of the local log heatmap, retaining a concave nearby maximum. */
cv::Point2d peakOffset(const cv::Mat & map, cv::Point peak)
{
  if (peak.x < 1 || peak.y < 1 || peak.x >= map.cols - 1 || peak.y >= map.rows - 1) {
    return {};
  }
  // Leave the ROI attached so the Gaussian uses its neighbours in the original heatmap.
  const cv::Rect region(peak.x - 1, peak.y - 1, 3, 3);
  cv::Mat1f smooth;
  cv::GaussianBlur(map(region), smooth, {3, 3}, 0);
  const auto log_at = [&](int y, int x) { return std::log(std::max(smooth(y, x), 1e-10f)); };
  const double center = log_at(1, 1);
  const double dx = (log_at(1, 2) - log_at(1, 0)) / 2;
  const double dy = (log_at(2, 1) - log_at(0, 1)) / 2;
  const double xx = log_at(1, 2) + log_at(1, 0) - 2 * center;
  const double yy = log_at(2, 1) + log_at(0, 1) - 2 * center;
  const double xy = (log_at(2, 2) - log_at(2, 0) - log_at(0, 2) + log_at(0, 0)) / 4;
  const double det = xx * yy - xy * xy;
  if (det > 1e-8 && xx + yy < 0) {
    const cv::Point2d offset((xy * dy - yy * dx) / det, (xy * dx - xx * dy) / det);
    if (std::max(std::abs(offset.x), std::abs(offset.y)) < 1.5) {
      return offset;
    }
  }
  return {};
}

}  // namespace

void decodeKeypoints(const float * heatmaps, const float * logits, const ModelMeta & meta,
                     const cv::Matx23d & image_from_crop, cv::Size image_size, ModelOutput & output)
{
  const double stride = static_cast<double>(meta.crop_size) / meta.heatmap_size;
  for (std::size_t i = 0; i < kKeypointCount; ++i) {
    const cv::Mat map(meta.heatmap_size, meta.heatmap_size, CV_32F, const_cast<float *>(heatmaps));
    cv::Point peak;
    double value;
    cv::minMaxLoc(map, nullptr, &value, nullptr, &peak);
    const cv::Point2d crop = (cv::Point2d(peak) + peakOffset(map, peak)) * stride;
    const auto xy = image_from_crop * cv::Vec3d(crop.x, crop.y, 1);
    output.pixels[i] = {xy[0], xy[1]};
    output.confidence[i] = std::clamp(value, 0.0, 1.0) / (1 + std::exp(-static_cast<double>(logits[i])));
    output.valid[i] = output.confidence[i] >= meta.keypoint_threshold &&
      xy[0] >= 0 && xy[0] < image_size.width && xy[1] >= 0 && xy[1] < image_size.height;
    heatmaps += map.total();
  }
}

}  // namespace tech_core_estimation_model
