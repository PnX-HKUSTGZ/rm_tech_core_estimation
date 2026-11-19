#include "tech_core_estimation_model/tech_core_model.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace tech_core_estimation_model
{
namespace
{

const std::filesystem::path release(TECH_CORE_GOLDEN_DIR);  ///< T4 release under test.

TEST(Golden, ReplaysEveryPyTorchFrame)
{
  std::string error;
  const auto model = TechCoreModel::load({release.string()}, error);
  ASSERT_TRUE(model) << error;
  EXPECT_TRUE(error.empty());
  const auto meta = YAML::LoadFile((release / "model_meta.yaml").string());
  const auto & geometry = model->geometry();
  EXPECT_EQ(geometry.sha256, meta["geometry"]["sha256"].as<std::string>());
  for (std::size_t i = 0; i < kKeypointCount; ++i) {
    EXPECT_EQ(geometry.names[i], meta["geometry"]["names"][i].as<std::string>());
    const auto xyz = meta["geometry"]["points_m"][i].as<std::array<double, 3>>();
    EXPECT_EQ(geometry.points_m[i], cv::Point3d(xyz[0], xyz[1], xyz[2]));
  }
  const double threshold = meta["keypoints"]["threshold"].as<double>();
  const auto golden = release / "golden";
  const auto cases = YAML::LoadFile((golden / "cases.yaml").string());
  std::size_t detections = 0;
  double maximum = 0;
  for (const auto & entry : cases) {
    const auto image = entry["image"].as<std::string>();
    SCOPED_TRACE(image);
    const auto frame = cv::imread((golden / image).string(), cv::IMREAD_COLOR);
    ASSERT_FALSE(frame.empty());
    const auto output = model->infer(frame);
    const auto bounds = entry["bbox_xyxy"];
    ASSERT_EQ(output.detected, !bounds.IsNull());
    if (!output.detected) {
      continue;
    }
    ++detections;
    EXPECT_NEAR(output.box.x, bounds[0].as<double>(), 0.5);
    EXPECT_NEAR(output.box.y, bounds[1].as<double>(), 0.5);
    EXPECT_NEAR(output.box.br().x, bounds[2].as<double>(), 0.5);
    EXPECT_NEAR(output.box.br().y, bounds[3].as<double>(), 0.5);
    EXPECT_NEAR(output.score, entry["score"].as<double>(), 1e-4);
    for (std::size_t i = 0; i < kKeypointCount; ++i) {
      SCOPED_TRACE(geometry.names[i]);
      const auto xy = entry["pixels"][i].as<std::array<double, 2>>();
      const auto deviation = cv::norm(output.pixels[i] - cv::Point2d(xy[0], xy[1]));
      EXPECT_LT(deviation, 0.5);
      maximum = std::max(maximum, deviation);
      const double confidence = entry["confidence"][i].as<double>();
      EXPECT_NEAR(output.confidence[i], confidence, 1e-3);
      if (std::abs(confidence - threshold) > 1e-3) {
        EXPECT_EQ(output.valid[i], entry["valid"][i].as<bool>());
      }
    }
  }
  EXPECT_GT(detections, 0u);
  std::cout << "Golden detections=" << detections << '/' << cases.size() << " max_deviation_px=" << maximum << '\n';
}

/** @brief Copy metadata into an isolated release directory for load-failure cases. */
class InvalidRelease : public ::testing::Test
{
protected:
  /** @brief Create a uniquely named scratch directory and load the original metadata. */
  void SetUp() override
  {
    std::string pattern = (::testing::TempDir() / std::filesystem::path("t5-release-XXXXXX")).string();
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory = pattern;
    meta = YAML::LoadFile((release / "model_meta.yaml").string());
  }

  /** @brief Remove the temporary release, including its local compilation cache. */
  void TearDown() override
  {
    if (!directory.empty()) {
      std::filesystem::remove_all(directory);
    }
  }

  /** @brief Run the public loader and require a diagnostic identifying the failing release field. */
  void expectError(const std::string & diagnostic, const std::string & device = "CPU")
  {
    std::ofstream(directory / "model_meta.yaml") << meta;
    std::string error;
    EXPECT_FALSE(TechCoreModel::load({directory.string(), Backend::kOpenVino, device}, error));
    EXPECT_NE(error.find(diagnostic), std::string::npos) << error;
  }

  std::filesystem::path directory;  ///< Isolated model release.
  YAML::Node meta;                  ///< Metadata to write for the next load.
};

TEST_F(InvalidRelease, RejectsUnknownSchemaAndWrongCount)
{
  meta["schema_version"] = 2;
  expectError("schema_version");
  meta["schema_version"] = 1;
  meta["geometry"]["names"].remove(0);
  expectError("16");
}

TEST_F(InvalidRelease, ReportsMissingGraphAndUnavailableDevice)
{
  expectError("detector.onnx");
  for (const auto name : {"detector.onnx", "keypoints.onnx"}) {
    std::filesystem::create_symlink(release / name, directory / name);
  }
  expectError("MISSING_DEVICE", "MISSING_DEVICE");
}

TEST_F(InvalidRelease, RejectsGraphExportedWithAnotherFingerprint)
{
  std::filesystem::create_symlink(release / "detector.onnx", directory / "detector.onnx");
  std::ifstream input(release / "keypoints.onnx", std::ios::binary);
  std::string bytes{std::istreambuf_iterator<char>(input), {}};
  const auto fingerprint = bytes.rfind(meta["geometry"]["sha256"].as<std::string>());
  ASSERT_NE(fingerprint, std::string::npos);
  bytes[fingerprint] = bytes[fingerprint] == '0' ? '1' : '0';
  std::ofstream(directory / "keypoints.onnx", std::ios::binary) << bytes;
  expectError("keypoints.onnx: geometry SHA-256");
}

}  // namespace
}  // namespace tech_core_estimation_model
