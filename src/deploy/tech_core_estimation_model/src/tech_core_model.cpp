#include "tech_core_estimation_model/tech_core_model.hpp"

#include "image_ops.hpp"

#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>

#include <stdexcept>
#include <utility>

namespace tech_core_estimation_model
{

struct TechCoreModel::Impl
{
  ModelMeta meta;              ///< Checked release metadata.
  ov::InferRequest detector;   ///< Detector request and its input/output tensors.
  ov::InferRequest keypoints;  ///< Landmark request and its input/output tensors.
};

namespace
{

/** @brief Load an ONNX graph exported for the same geometry as model_meta.yaml. */
std::shared_ptr<ov::Model> readGraph(ov::Core & core, const std::filesystem::path & path, const Geometry & geometry)
{
  auto graph = core.read_model(path.string());
  if (!graph->has_rt_info("framework", "geometry_sha256") ||
      graph->get_rt_info<std::string>("framework", "geometry_sha256") != geometry.sha256) {
    throw std::runtime_error(path.string() + ": geometry SHA-256 differs from model_meta.yaml");
  }
  return graph;
}

/** @brief Fill the request's float32 NCHW input in RGB order and run one stage. */
void runStage(ov::InferRequest & request, const cv::Mat & bgr, double scale)
{
  auto tensor = request.get_input_tensor();
  const int shape[] = {1, 3, bgr.rows, bgr.cols};
  cv::Mat blob(4, shape, CV_32F, tensor.data<float>());
  cv::dnn::blobFromImage(bgr, blob, scale, {}, {}, true);
  request.infer();
}

}  // namespace

std::unique_ptr<TechCoreModel> TechCoreModel::load(const ModelOptions & options, std::string & error)
{
  try {
    const std::filesystem::path directory(options.model_dir);
    auto impl = std::make_unique<Impl>();
    auto & meta = impl->meta;
    meta = readModelMeta(directory / "model_meta.yaml");
    ov::Core core;
    const auto detector = readGraph(core, directory / "detector.onnx", meta.geometry);
    const auto keypoints = readGraph(core, directory / "keypoints.onnx", meta.geometry);
    const auto cache = directory / "cache" / "openvino";
    std::filesystem::create_directories(cache);
    core.set_property(ov::cache_dir(cache.string()));
    const auto precision = ov::hint::inference_precision(ov::element::f32);
    impl->detector = core.compile_model(detector, options.device, precision).create_infer_request();
    impl->keypoints = core.compile_model(keypoints, options.device, precision).create_infer_request();
    error.clear();
    return std::unique_ptr<TechCoreModel>(new TechCoreModel(std::move(impl)));
  } catch (const std::exception & exception) {
    error = exception.what();
    return nullptr;
  }
}

TechCoreModel::TechCoreModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TechCoreModel::~TechCoreModel() = default;

const Geometry & TechCoreModel::geometry() const
{
  return impl_->meta.geometry;
}

ModelOutput TechCoreModel::infer(const cv::Mat & bgr)
{
  const auto & meta = impl_->meta;
  cv::Mat padded;
  const auto mapping = letterbox(bgr, meta.detector_size, meta.pad_value, padded);
  runStage(impl_->detector, padded, 1.0 / 255.0);
  auto candidates = impl_->detector.get_tensor("candidates");
  const cv::Mat rows(5, static_cast<int>(candidates.get_shape()[2]), CV_32F, candidates.data<float>());
  auto output = bestBox(rows, mapping, bgr.size(), meta.box_threshold);
  if (output.detected) {
    cv::Mat crop;
    const auto image_from_crop = squareCrop(bgr, output.box, meta.crop_size, meta.padding, crop);
    runStage(impl_->keypoints, crop, 1.0);
    const auto heatmaps = impl_->keypoints.get_tensor("heatmaps");
    const auto logits = impl_->keypoints.get_tensor("logits");
    decodeKeypoints(heatmaps.data<float>(), logits.data<float>(), meta, image_from_crop, bgr.size(), output);
  }
  return output;
}

}  // namespace tech_core_estimation_model
