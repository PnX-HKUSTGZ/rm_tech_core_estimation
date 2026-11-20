#include "image_ops.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace tech_core_estimation_model
{
namespace
{

TEST(ImageOps, LetterboxRoundsHalfToEvenAndFloorsOddPadding)
{
  const cv::Mat frame(5, 14, CV_8UC3, cv::Scalar::all(255));
  cv::Mat padded;
  const auto mapping = letterbox(frame, 7, 114, padded);
  EXPECT_EQ(mapping.gain, 0.5);
  EXPECT_EQ(mapping.pad, cv::Point(0, 2));
  for (int y = 0; y < 7; ++y) {
    EXPECT_EQ(padded.at<cv::Vec3b>(y, 3)[0], y == 2 || y == 3 ? 255 : 114);
  }
}

TEST(ImageOps, HighestCandidateUsesInclusiveThresholdAndClippedOriginalBounds)
{
  const cv::Mat1f candidates = (cv::Mat1f(5, 2) << 1, 3, 1, 4, 1, 10, 1, 4, 0.1, 0.25);
  const Letterbox mapping{0.5, {0, 2}};
  const auto result = bestBox(candidates, mapping, {14, 5}, 0.25);
  ASSERT_TRUE(result.detected);
  EXPECT_EQ(result.box, cv::Rect2d(0, 0, 14, 5));
  EXPECT_EQ(result.score, 0.25);
  EXPECT_FALSE(bestBox(candidates, mapping, {14, 5}, 0.3).detected);
  EXPECT_FALSE(bestBox(candidates, {0.5, {20, 0}}, {14, 5}, 0.25).detected);
}

TEST(ImageOps, PaddedSquareCropRestoresSampledSourcePixels)
{
  cv::Mat3b frame(20, 20);
  for (int y = 0; y < frame.rows; ++y) {
    for (int x = 0; x < frame.cols; ++x) {
      frame(y, x) = cv::Vec3b(x, y, 255);
    }
  }
  cv::Mat crop;
  const auto image_from_crop = squareCrop(frame, {0, 0, 16, 8}, 10, 1.25, crop);
  EXPECT_EQ(crop.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
  EXPECT_EQ(crop.at<cv::Vec3b>(4, 6), frame(3, 11));
  EXPECT_EQ(image_from_crop * cv::Vec3d(6, 4, 1), cv::Vec2d(11, 3));
  EXPECT_EQ(image_from_crop * cv::Vec3d(4.5, 4.5, 1), cv::Vec2d(8, 4));
}

TEST(ImageOps, FractionalCropPreservesBilinearWeightsAtTheImageEdge)
{
  cv::Mat3b frame(3, 3, cv::Vec3b(200, 100, 0));
  cv::Mat crop;
  squareCrop(frame, {0, 0, 2, 2}, 3, 1.68, crop);
  // 左上角采样点位于 (-0.12, -0.12)，只有一个邻点在图像内。
  EXPECT_EQ(crop.at<cv::Vec3b>(0, 0), cv::Vec3b(155, 77, 0));
}

TEST(ImageOps, GaussianPeakIsSubpixelAndConfidenceCombinesPeakWithLogit)
{
  ModelMeta meta{};
  meta.crop_size = 32;
  meta.heatmap_size = 16;
  meta.keypoint_threshold = 0.25;
  const int side = meta.heatmap_size;
  std::vector<float> maps(kKeypointCount * side * side, 0.0f);
  std::array<float, kKeypointCount> logits{};
  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      maps[y * side + x] = std::exp(-(std::pow(x - 5.3, 2) + std::pow(y - 6.7, 2)) / (2 * 2.5 * 2.5));
    }
  }
  maps[side * side] = 0.5f;
  maps[2 * side * side + side - 1] = 1.2f;
  ModelOutput output;
  decodeKeypoints(maps.data(), logits.data(), meta, {1, 0, 2, 0, 1, 1}, {32, 32}, output);
  EXPECT_NEAR(output.pixels[0].x, 2 + 2 * 5.3, 0.01);
  EXPECT_NEAR(output.pixels[0].y, 1 + 2 * 6.7, 0.01);
  EXPECT_FLOAT_EQ(output.confidence[0], maps[7 * side + 5] / 2);
  EXPECT_TRUE(output.valid[0]);
  EXPECT_EQ(output.confidence[1], meta.keypoint_threshold);
  EXPECT_TRUE(output.valid[1]);
  EXPECT_EQ(output.pixels[2], cv::Point2d(32, 1));
  EXPECT_EQ(output.confidence[2], 0.5);
  EXPECT_FALSE(output.valid[2]);
  EXPECT_FALSE(output.valid[3]);
}

}  // namespace
}  // namespace tech_core_estimation_model
