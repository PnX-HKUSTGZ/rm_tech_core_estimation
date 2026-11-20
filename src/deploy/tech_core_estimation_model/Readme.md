# tech_core_estimation_model

该库封装两阶段关键点模型的全部推理细节。调用方传入一张 BGR 图像，得到原图坐标下的检测框、16 个关键点的像素坐标、置信度以及有效标记。库内部完成预处理、两次推理、热图解码和坐标还原，调用方无需了解 YOLO、LiteHRNet 或推理后端。

## 职责

| 负责 | 不负责 |
|---|---|
| 读取并校验发布目录（`detector.onnx`、`keypoints.onnx`、`model_meta.yaml`） | 位姿解算、相机畸变处理（见 [tech_core_estimation](../tech_core_estimation/Readme.md)） |
| 检测：letterbox、推理、取最高分候选框（单目标，不做 NMS） | ROS 消息与参数（见 [tech_core_estimation_ros](../tech_core_estimation_ros/Readme.md)） |
| 关键点：1.25 倍正方形裁剪到 256×256、推理、亚像素热图解码、逆变换回原图 | 训练与门限选择 |
| 应用随模型冻结的框门限和关键点门限，给出逐点有效标记 | |
| OpenVINO 推理 | |

预处理、解码和门限的参数全部从 `model_meta.yaml` 读取，与 model_train 中 PyTorch 端的实现逐项对应；库内不提供覆盖这些参数的接口。

## 接口

```cpp
namespace tech_core_estimation_model {

/** @brief 推理后端。 */
enum class Backend { kOpenVino };

/** @brief 模型加载选项。 */
struct ModelOptions {
  std::string model_dir;                 ///< model_train 导出的发布目录。
  Backend backend = Backend::kOpenVino;  ///< 推理后端。
  std::string device = "CPU";            ///< OpenVINO 设备名（"CPU"/"GPU"/"AUTO"）。
};

/** @brief 单帧推理结果，坐标均在输入原图中。 */
struct ModelOutput {
  bool detected;                                   ///< 是否检出目标；为 false 时其余字段无意义。
  cv::Rect2d box;                                  ///< 检测框，单位 pixel。
  double score;                                    ///< 检测置信度，范围 [0, 1]。
  std::array<cv::Point2d, kKeypointCount> pixels;  ///< 16 点像素坐标，单位 pixel，顺序同 model_meta.yaml。
  std::array<double, kKeypointCount> confidence;   ///< 逐点置信度，范围 [0, 1]。
  std::array<bool, kKeypointCount> valid;          ///< 置信度不低于关键点门限、坐标有限且位于图内。
};

class TechCoreModel {
public:
  /** @brief 加载模型；失败时返回 nullptr，并在 error 中写入原因。 */
  static std::unique_ptr<TechCoreModel> load(const ModelOptions & options, std::string & error);

  /** @brief 对一帧 BGR8 图像推理。@note 非线程安全，每个线程使用独立实例。 */
  ModelOutput infer(const cv::Mat & bgr);

  /** @brief 发布目录中的几何契约：点名、核心系坐标（m）及其 SHA-256。 */
  const Geometry & geometry() const;
};

}  // namespace tech_core_estimation_model
```

输入 BGR8，与 OpenCV 和 ROS `bgr8` 一致；库内按 `model_meta.yaml` 转换为模型所需的 RGB。公共头文件仅使用 OpenCV 和标准库类型。

## 后端

从 Intel 官方 APT 源安装 OpenVINO 开发包（`sudo apt install openvino`），CMake 通过 `find_package(OpenVINO)` 查找。`load` 读取两个 ONNX，核对其几何指纹与发布目录中的几何数据，以 FP32 编译，缓存目录为 `<model_dir>/cache/openvino/`。TensorRT 后端尚未实现。

## 测试与验收

- **金标准一致性**：对发布目录中 `golden/` 的每张图像，检测与否须与 PyTorch 一致，关键点像素最大偏差 < 0.5 pixel。配置 `-DTECH_CORE_GOLDEN_DIR=<发布目录>` 后构建并运行金标准测试。
- **耗时**：`tech_core_model_benchmark <model_dir> --device CPU` 在 `golden/` 上统计预热后的单帧耗时中位数与 P95（s），比较时须注明硬件型号。
- **加载失败**：几何指纹不一致、文件缺失、设备编译失败时，`load` 返回 nullptr 并给出可读的错误信息。

## 依赖

`package.xml`：`ament_cmake`、`libopencv-dev`、`yaml-cpp`。OpenVINO 的安装方式见上文。
