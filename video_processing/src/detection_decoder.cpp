#include "detection_decoder.h"

void DetectionDecoder::DrawObjects(cv::Mat& image,
                                   const std::vector<Object>& objects,
                                   const std::set<int>& selected_ids,
                                   const std::vector<std::string>* labels) {
  for (size_t i = 0; i < objects.size(); i++) {
    const Object& obj = objects[i];

    if (!selected_ids.empty() &&
        selected_ids.find(obj.label) == selected_ids.end())
      continue;

    cv::Scalar color = cv::Scalar(255, 255, 0);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }
    cv::rectangle(image, obj.rect, color, 2);

    char text[256];
    if (labels) {
      sprintf(text, "%s %.1f%%", (*labels)[obj.label].c_str(), obj.prob * 100);
    } else {
      sprintf(text, "label-%d %.1f%%", obj.label, obj.prob * 100);
    }

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    // int y = obj.rect.y - label_size.height - baseLine;
    if (y > image.rows) y = image.rows;
    // if (x + label_size.width > image.cols)
    // x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
}
