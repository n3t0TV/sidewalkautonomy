#ifndef DETECTION_DECODER_H
#define DETECTION_DECODER_H
#include <opencv2/opencv.hpp>
#include <set>

class DetectionDecoder {
 public:
  struct Object {
    cv::Rect2f rect;
    int label;
    float prob;
  };
  virtual void Decode(const std::vector<cv::Mat> &outRaw,
                      std::vector<Object> &objects,
                      const cv::Size &img_size = {}) = 0;

  virtual void ProcessInput(cv::Mat frame, float *out){};
  virtual ~DetectionDecoder() {}
  std::function<void(cv::Mat, float *)> GetNormalizationFunct() {
    return normalization_fnt_;
  };

  void SetLabels(const std::vector<std::string> &lb) { labels_ = lb; }
  std::vector<std::string> GetLabels() const { return labels_; }
  std::string IdToLabel(const int i) const {
    if (labels_.empty()) return {};
    return labels_[i];
  }
  static void DrawObjects(cv::Mat &bgr, const std::vector<Object> &objects,
                          const std::set<int> &selected_ids = {},
                          const std::vector<std::string> *labels = nullptr);

 protected:
  std::function<void(cv::Mat, float *)> normalization_fnt_{};
  std::vector<std::string> labels_;
};

#endif  // DETECTION_DECODER_H
