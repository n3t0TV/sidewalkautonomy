#ifndef FUNCT_UTILS_H
#define FUNCT_UTILS_H
#include "classification_tracker.h"
#include "classifier.h"
#include "detection_decoder.h"
#include "nlohmann/json.hpp"
#include "object_tracker.h"
#include "predictor.h"

nlohmann::json LoadJSon(const std::string &fpath);
std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf);
std::shared_ptr<Classifier> ClassifierFromJson(const nlohmann::json &conf);
std::vector<std::string> StringListFromJson(const nlohmann::json &json_array);
std::shared_ptr<DetectionDecoder> DetectionDecoderFromJson(
    const nlohmann::json &conf);
void ObjectTrackerFromJson(const nlohmann::json &conf, ObjectTracker &tracker);

void SoftMax(cv::Mat &out);
std::pair<int, float> ArgMax(const cv::Mat &out);
void ScaleRect(cv::Rect &r, float scale);
cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame);
cv::Mat Get4x3Roi(cv::Mat frame);

std::vector<std::string> Split(const std::string &s, char delimiter);

float LinearMap(float a, float b, float x);


class SmootherAve {
public:
   void Init(int nave, float initial_val=0);
   void Reset();
   float Update(float x);

private:
   int nave_{1};
   float val0_;
   float curr_val_;
   long counter_{0};
};


#endif  // FUNCT_UTILS_H
