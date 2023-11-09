#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include "classification_tracker.h"
#include "predictor.h"
#include <memory>

class Classifier {
public:
  Classifier();
  void Init(std::shared_ptr<Predictor> inf, int ncls, int nrecents = 1,
            const std::vector<std::string> &labels = {});

  void Predict(const cv::Mat frame, std::vector<float> *scores = nullptr);
  const ClassificationTracker &GetTracker() const { return cls_tracker_; }
  std::string GetLabel(int i) const { return labels_[i]; }

private:
  std::shared_ptr<Predictor> inf_;
  ClassificationTracker cls_tracker_;
  std::vector<std::string> labels_;
};

#endif // CLASSIFIER_H
