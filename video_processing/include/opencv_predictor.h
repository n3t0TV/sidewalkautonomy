#ifndef OPENCV_PREDICTOR_H_
#define OPENCV_PREDICTOR_H_

#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "predictor.h"

using namespace std;

class OpenCVPredictor : public Predictor {
 public:
  static std::shared_ptr<Predictor> Create(
      const std::string &model_path, const std::string &config_path,
      const std::string &framework = "ONNX", int async_count = 0);
  static std::shared_ptr<Predictor> Create(
      const char *model_buffer, const size_t model_size,
      const char *config_buffer, const size_t config_size,
      const std::string &framework = "ONNX", int async_count = 0);

  void Predict(const cv::Mat &img, std::vector<cv::Mat> &outputs,
               const std::vector<std::string> &output_names = {}) override;
  bool EnqueueToInput(const cv::Mat &img, void *user_param = nullptr) override;
  bool ReadFromOutput(std::vector<cv::Mat> &outputs,
                      std::vector<std::string> onames = {},
                      void **user_param = nullptr) override;

  bool OkToRead() override;
  bool OkToWrite() override;

  int ActiveInferences() const override {
    return running_inferences_count_.load();
  }

  void SetTargetDevice(int target) {
    if (!net_.empty()) net_.setPreferableTarget(target);
  }

  void SetBackend(int backend) {
    if (!net_.empty()) net_.setPreferableBackend(backend);
  }

 private:
  OpenCVPredictor(){};
  bool Init(const std::string &model_path, const std::string &config_path,
            const std::string &framework = "ONNX", int async_count = 0);
  bool Init(const char *model_buffer, const size_t model_size,
            const char *config_buffer, const size_t config_size,
            const std::string &framework = "ONNX", int async_count = 0);

 private:
  cv::dnn::Net net_{};
  std::vector<cv::AsyncArray> inferences_;
  std::vector<void *> user_data_;
  std::atomic_size_t running_inferences_count_{0};
  size_t index_in_{0};
  size_t index_out_{0};
  int buffer_input_index_{-1};
};

#endif
