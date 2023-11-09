#ifndef YOLOX_DECODER_H
#define YOLOX_DECODER_H
#include <opencv2/opencv.hpp>

#include "detection_decoder.h"

class YoloXDecoder : public DetectionDecoder {
  struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
  };
  std::vector<GridAndStride> grid_strides_;

 public:
  static std::shared_ptr<DetectionDecoder> Create(const cv::Size &input_size, int ncls,
                                                  float objTh, float nmsTh);

  void Init(const cv::Size &input_size, int ncls, float objTh, float nmsTh);
  virtual void Decode(const std::vector<cv::Mat> &outRaw,
                      std::vector<Object> &objects,
                      const cv::Size &img_size) override;
  virtual void ProcessInput(cv::Mat frame, float *out) override;

 private:
  void GenerateGridsAndStride(const std::vector<int> &strides,
                              std::vector<GridAndStride> &grid_strides);
  void NMSSortedBBoxes(const std::vector<Object> &faceobjects,
                       std::vector<int> &picked, float nms_threshold);
  void GenerateYoloxProposals(std::vector<GridAndStride> grid_strides,
                              const float *feat_blob, float prob_threshold,
                              std::vector<Object> &objects);
  inline float IntersectionArea(const Object &a, const Object &b) {
    return (a.rect & b.rect).area();
  }

  int INPUT_H_;
  int INPUT_W_;
  int NUM_CLASSES_;
  float BBOX_CONF_THRESH_;
  float NMS_THRESH_;
};

#endif  // YOLOX_DECODER_H
