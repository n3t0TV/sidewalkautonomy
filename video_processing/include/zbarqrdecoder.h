#ifndef ZBARQRDECODER_H
#define ZBARQRDECODER_H
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ZBarQRDecoder {
 public:
  ZBarQRDecoder() = default;
  std::vector<std::string> Decode(const cv::Mat &image);
  std::string DecodeSingle(const cv::Mat &image);
};

#endif  // ZBARQRDECODER_H
