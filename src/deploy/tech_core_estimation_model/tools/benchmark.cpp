#include "tech_core_estimation_model/tech_core_model.hpp"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace
{

/** @brief 对已排序的单帧耗时线性插值求分位数，单位 s。 */
double quantile(const std::vector<double> & seconds, double fraction)
{
  const double index = fraction * (seconds.size() - 1);
  const auto lower = static_cast<std::size_t>(index);
  const auto upper = static_cast<std::size_t>(std::ceil(index));
  return seconds[lower] + (index - lower) * (seconds[upper] - seconds[lower]);
}

}  // namespace

/** @brief 预热后在发布目录的金标准帧上测量同步端到端推理耗时。 */
int main(int argc, char ** argv)
{
  using namespace tech_core_estimation_model;
  if (argc < 2 || argc % 2 != 0) {
    std::cerr << "Usage: tech_core_model_benchmark <model_dir> [--device CPU] [--backend openvino]\n";
    return 2;
  }
  ModelOptions options;
  options.model_dir = argv[1];
  for (int i = 2; i < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--device") {
      options.device = argv[i + 1];
    } else if (flag != "--backend" || std::string(argv[i + 1]) != "openvino") {
      std::cerr << "Unsupported option: " << flag << ' ' << argv[i + 1] << '\n';
      return 2;
    }
  }
  std::string error;
  const auto model = TechCoreModel::load(options, error);
  if (!model) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto golden = std::filesystem::path(options.model_dir) / "golden";
  std::vector<cv::Mat> frames;
  try {
    const auto cases = YAML::LoadFile((golden / "cases.yaml").string());
    for (const auto & entry : cases) {
      const auto path = golden / entry["image"].as<std::string>();
      auto frame = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (frame.empty()) {
        std::cerr << "Failed to read " << path << '\n';
        return 1;
      }
      frames.push_back(frame);
    }
  } catch (const YAML::Exception & exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  if (frames.empty()) {
    std::cerr << "No frames in golden/cases.yaml\n";
    return 1;
  }
  for (const auto & frame : frames) {
    model->infer(frame);
  }
  std::vector<double> seconds;
  for (const auto & frame : frames) {
    const auto begin = std::chrono::steady_clock::now();
    model->infer(frame);
    seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
  }
  std::sort(seconds.begin(), seconds.end());
  std::cout << "device=" << options.device << " frames=" << seconds.size()
            << " median_s=" << quantile(seconds, 0.5) << " p95_s=" << quantile(seconds, 0.95) << '\n';
}
