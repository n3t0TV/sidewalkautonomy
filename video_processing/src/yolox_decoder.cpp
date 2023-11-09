#include "yolox_decoder.h"

std::shared_ptr<DetectionDecoder> YoloXDecoder::Create(const cv::Size &input_size, int ncls,
                                                       float objTh,
                                                       float nmsTh) {
  auto ptr = new YoloXDecoder();
  ptr->Init(input_size, ncls, objTh, nmsTh);
  return std::shared_ptr<DetectionDecoder>{ptr};
}

void YoloXDecoder::Init(const cv::Size &input_size, int ncls, float objTh, float nmsTh) {
  INPUT_W_ = input_size.width;
  INPUT_H_ = input_size.height;
  NUM_CLASSES_ = ncls;
  BBOX_CONF_THRESH_ = objTh;
  NMS_THRESH_ = nmsTh;
  grid_strides_.clear();
  GenerateGridsAndStride({8, 16, 32}, grid_strides_);
  DetectionDecoder::normalization_fnt_ =
      std::bind(&YoloXDecoder::ProcessInput, this, std::placeholders::_1,
                std::placeholders::_2);
}

void YoloXDecoder::GenerateGridsAndStride(
    const std::vector<int> &strides, std::vector<GridAndStride> &grid_strides) {
  for (auto stride : strides) {
    int num_grid_y = INPUT_H_ / stride;
    int num_grid_x = INPUT_W_ / stride;
    for (int g1 = 0; g1 < num_grid_y; g1++) {
      for (int g0 = 0; g0 < num_grid_x; g0++) {
        grid_strides.emplace_back(GridAndStride{g0, g1, stride});
      }
    }
  }
}

void YoloXDecoder::NMSSortedBBoxes(const std::vector<Object> &faceobjects,
                                   std::vector<int> &picked,
                                   float nms_threshold) {
  picked.clear();
  const int n = faceobjects.size();
  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }
  for (int i = 0; i < n; i++) {
    const Object &a = faceobjects[i];
    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = faceobjects[picked[j]];
      // intersection over union
      float inter_area = IntersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      if (inter_area / union_area > nms_threshold) keep = 0;
    }
    if (keep) picked.push_back(i);
  }
}

void YoloXDecoder::GenerateYoloxProposals(
    std::vector<GridAndStride> grid_strides, const float *feat_blob,
    float prob_threshold, std::vector<Object> &objects) {
  const int num_anchors = grid_strides.size();

#pragma omp parallel for
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES_ + 5);

    // yolox/models/yolo_head.py decode logic
    float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
    float w = exp(feat_blob[basic_pos + 2]) * stride;
    float h = exp(feat_blob[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_blob[basic_pos + 4];
    for (int class_idx = 0; class_idx < NUM_CLASSES_; class_idx++) {
      float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;
#pragma omp critical
        { objects.push_back(obj); }
      }
    }  // class loop
  }    // point anchor loop
}

void YoloXDecoder::Decode(const std::vector<cv::Mat> &outRaw,
                          std::vector<Object> &objects,
                          const cv::Size &img_size) {
  const float *prob = outRaw[0].ptr<float>(0);
  const float img_w = img_size.width;
  const float img_h = img_size.height;
  const float scale = std::min(INPUT_W_ / img_w, INPUT_H_ / img_h);
  std::vector<Object> proposals;
  GenerateYoloxProposals(grid_strides_, prob, BBOX_CONF_THRESH_, proposals);

  // Sort descent inplace
  std::sort(proposals.begin(), proposals.end(),
            [](Object &o1, Object &o2) -> bool { return o1.prob > o2.prob; });

  // NMS
  std::vector<int> picked;
  NMSSortedBBoxes(proposals, picked, NMS_THRESH_);

  int count = picked.size();
  // std::cout << "num of boxes: " << count << std::endl;
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x) / scale;
    float y0 = (objects[i].rect.y) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

    // clip
    x0 = std::max(std::min(x0, img_w - 1), 0.f);
    y0 = std::max(std::min(y0, img_h - 1), 0.f);
    x1 = std::max(std::min(x1, img_w - 1), 0.f);
    y1 = std::max(std::min(y1, img_h - 1), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }
}

void YoloXDecoder::ProcessInput(cv::Mat img, float *blob) {
  const float r =
      std::min(INPUT_W_ / (img.cols * 1.0), INPUT_H_ / (img.rows * 1.0));
  int unpad_w = r * img.cols;
  int unpad_h = r * img.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(img, re, re.size());
  cv::Mat outMat(INPUT_H_, INPUT_W_, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(outMat(cv::Rect(0, 0, re.cols, re.rows)));

  for (size_t c = 0; c < 3; c++) {
    for (size_t h = 0; h < INPUT_H_; h++) {
      for (size_t w = 0; w < INPUT_W_; w++) {
        blob[c * INPUT_W_ * INPUT_H_ + h * INPUT_W_ + w] =
            static_cast<float>(outMat.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}
