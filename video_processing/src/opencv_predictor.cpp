#include "opencv_predictor.h"

#include <fstream>
#include <iostream>

#include "predictor.h"

std::shared_ptr<Predictor> OpenCVPredictor::Create(
    const std::string &model_path, const std::string &config_path,
    const std::string &framework, int async_count) {
  auto ptr = new OpenCVPredictor();
  bool ok = ptr->Init(model_path, config_path, framework, async_count);
  if (ok) {
    return std::unique_ptr<Predictor>{ptr};
  }
  delete ptr;
  return {};
}

std::shared_ptr<Predictor> OpenCVPredictor::Create(const char *model_buffer,
                                                   const size_t model_size,
                                                   const char *config_buffer,
                                                   const size_t config_size,
                                                   const std::string &framework,
                                                   int async_count) {
  auto ptr = new OpenCVPredictor();
  bool ok = ptr->Init(model_buffer, model_size, config_buffer, config_size,
                      framework, async_count);
  if (ok) {
    return std::unique_ptr<Predictor>{ptr};
  }
  delete ptr;
  return {};
}

bool OpenCVPredictor::Init(const std::string &model_path,
                           const std::string &config_path,
                           const std::string &framework, int async_count) {
  std::string model_buffer{}, config_buffer{};
  Predictor::GetStringData(model_path, &model_buffer);
  if (!config_path.empty()) {
    Predictor::GetStringData(config_path, &config_buffer);
  }
  return Init(model_buffer.data(), model_buffer.size(), config_buffer.data(),
              config_buffer.size(), framework, async_count);
}

bool OpenCVPredictor::Init(const char *model_buffer, const size_t model_size,
                           const char *config_buffer, const size_t config_size,
                           const std::string &framework, int async_count) {
  if (framework == "ONNX") {
    net_ = cv::dnn::readNetFromONNX(model_buffer, model_size);
  } else if (framework == "Caffe") {
    net_ = cv::dnn::readNetFromCaffe(config_buffer, config_size, model_buffer,
                                     model_size);
  } else if (framework == "TensorFlow") {
    net_ = cv::dnn::readNetFromTensorflow(config_buffer, config_size);
  } else {
    std::cerr << "Failed reading the network! Supported frameworks are: "
                 "ONNX|Caffe|TensorFlow"
              << std::endl;
    return false;
  }

  if (net_.empty()) {
    std::cerr << "Failed loading the network " << std::endl;
    return false;
  }

  inferences_.resize(async_count);
  user_data_.resize(async_count, nullptr);
  return true;
}

void OpenCVPredictor::Predict(const cv::Mat &img, std::vector<cv::Mat> &outputs,
                              const std::vector<std::string> &output_names) {
  cv::Mat blob;
  if (normalization_fnt_) {
    blob.create({1, input_ch_, input_size_.height, input_size_.width}, CV_32F);
    normalization_fnt_(img, blob.ptr<float>(0));
  } else {
    blob = cv::dnn::blobFromImage(img, scale_, input_size_, mean_, swap_ch_,
                                  false, CV_32F);
  }

  net_.setInput(blob);
  if (output_names.empty()) {
    net_.forward(outputs);
  } else {
    net_.forward(outputs, output_names);
  }
}

bool OpenCVPredictor::EnqueueToInput(const cv::Mat &img, void *user_param) {
  if (running_inferences_count_.load() < inferences_.size()) {
    const cv::Mat input_blob = cv::dnn::blobFromImage(
        img, scale_, input_size_, mean_, swap_ch_, false, CV_32F);
    net_.setInput(input_blob);
    inferences_[index_in_] = net_.forwardAsync();
    user_data_[index_in_] = user_param;
    index_in_ = (index_in_ + 1) % inferences_.size();
    running_inferences_count_.fetch_add(1);
    return true;
  }
  return false;
}

bool OpenCVPredictor::ReadFromOutput(std::vector<cv::Mat> &outputs,
                                     std::vector<std::string> onames,
                                     void **user_param) {
  if (running_inferences_count_.load() > 0) {
    if (inferences_[index_out_].wait_for(std::chrono::seconds(1))) {
      outputs.resize(1);
      inferences_[index_out_].get(outputs[0]);
      if (user_param) {
        *user_param = user_data_[index_out_];
      }
      index_out_ = (index_out_ + 1) % inferences_.size();
      running_inferences_count_.fetch_sub(1);
      return true;
    }
  }
  return false;
}

bool OpenCVPredictor::OkToRead() {
  return (running_inferences_count_.load() > 0) &&
         (inferences_[index_out_].wait_for(std::chrono::seconds(0)));
}

bool OpenCVPredictor::OkToWrite() {
  return running_inferences_count_.load() < inferences_.size();
}
