#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <functional>
#include <opencv2/opencv.hpp>

class Predictor {
 public:
  static void GetStringData(const std::string &filePath, std::string *data);

  virtual void Predict(const cv::Mat &img, std::vector<cv::Mat> &outputs,
                       const std::vector<std::string> &output_names = {}) = 0;
  virtual bool EnqueueToInput(const cv::Mat &img, void *user_param = nullptr) {
    return false;
  }
  virtual bool ReadFromOutput(std::vector<cv::Mat> &outputs,
                              std::vector<std::string> onames = {},
                              void **user_param = nullptr) {
    return false;
  }

  virtual void setExplictInputSize(const cv::Size &input_size) {
    input_size_ = input_size;
  }

  void SetInputParamsNorm(float scale, const cv::Scalar &mean, bool swap,
                          std::function<void(cv::Mat, float *)> fnt = {}) {
    scale_ = scale;
    mean_ = mean;
    swap_ch_ = swap;
    normalization_fnt_ = fnt;
  }

  virtual bool OkToRead() { return false; }
  virtual bool OkToWrite() { return false; }
  virtual int ActiveInferences() const { return 0; }
  virtual ~Predictor(){};

 protected:
  Predictor() {}
  std::function<void(cv::Mat, float *)> normalization_fnt_{};
  cv::Size input_size_{};
  int input_ch_ = 3;
  cv::Scalar mean_{};
  double scale_ = 1.0;
  bool swap_ch_ = false;
};

#endif
