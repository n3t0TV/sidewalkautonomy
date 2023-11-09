#ifndef SSD_DECODER_H
#define SSD_DECODER_H
#include <opencv2/opencv.hpp>

#include "detection_decoder.h"

class SSDDecoder : public DetectionDecoder {
 public:
  static std::shared_ptr<DetectionDecoder> Create(const cv::Size &input_size, float objTh,
                                                  float nmsTh);

 private:
  SSDDecoder(const cv::Size &input_size, float objTh, float nmsTh)
      : INPUT_H_(input_size.height),
        INPUT_W_(input_size.width),
        BBOX_CONF_THRESH_(objTh),
        NMS_THRESH_(nmsTh) {}

  virtual void Decode(const std::vector<cv::Mat> &outRaw,
                      std::vector<Object> &objects,
                      const cv::Size &img_size) override;

 private:
  int INPUT_H_;
  int INPUT_W_;
  float BBOX_CONF_THRESH_;
  float NMS_THRESH_;
};

#endif  // SSD_DECODER_H
