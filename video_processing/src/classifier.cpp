#include "classifier.h"

Classifier::Classifier() : cls_tracker_(1, 1) {}

void Classifier::Init(std::shared_ptr<Predictor> inf, int ncls, int nrecents,
                      const std::vector<std::string> &labels) {
  inf_ = inf;
  cls_tracker_ = ClassificationTracker(ncls, nrecents);
  if (labels.empty()) {
    for (int i = 0; i < ncls; ++i) {
      labels_.push_back("label" + std::to_string(i));
    }
  } else {
    labels_ = labels;
  }
}

void Classifier::Predict(const cv::Mat frame, std::vector<float> *scores) {
  std::vector<cv::Mat> outputs;
  inf_->Predict(frame, outputs);
  if (scores) *scores = outputs[0];
  cls_tracker_.Update(0, outputs[0]);
}
