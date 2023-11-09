#include "ssd_decoder.h"

#include <cmath>
#include <iostream>

#define CLIP(x, x1, x2) std::max(x1, std::min(x, x2))

std::shared_ptr<DetectionDecoder> SSDDecoder::Create(const cv::Size &input_size,
                                                     float objTh, float nmsTh) {
  auto ptr = new SSDDecoder(input_size, objTh, nmsTh);
  return std::shared_ptr<DetectionDecoder>{ptr};
}

void SSDDecoder::Decode(const std::vector<cv::Mat> &outRaw,
                        std::vector<Object> &objects,
                        const cv::Size &img_size) {
  const float img_w = img_size.width;
  const float img_h = img_size.height;

  auto prob = outRaw[0];
  objects.clear();
  DetectionDecoder::Object obj;
  for (int row = 0; row < prob.size[2]; row++) {
    const float *prob_score = prob.ptr<float>(0, 0, row);
    if (prob_score[2] > BBOX_CONF_THRESH_) {
      int x0 = CLIP(prob_score[3] * img_w, 0.0f, img_w - 1.0f);
      int y0 = CLIP(prob_score[4] * img_h, 0.0f, img_h - 1.0f);
      int x1 = CLIP(prob_score[5] * img_w, 0.0f, img_w - 1.0f);
      int y1 = CLIP(prob_score[6] * img_h, 0.0f, img_h - 1.0f);
      obj.rect = cv::Rect(cv::Point{x0, y0}, cv::Point{x1, y1});
      obj.label = prob_score[1];
      obj.prob = prob_score[2];
      objects.push_back(obj);
    }
  }
}
