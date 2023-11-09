#include "object_tracker.h"

#include <opencv2/opencv.hpp>

namespace {
constexpr float kMaxFloat32 = std::numeric_limits<float>::max();
}

void ObjectTracker::Init(int frames_to_count, int frames_to_discard,
                         float deformation_th) {
  frames_to_count_ = frames_to_count;
  frames_to_discard_ = frames_to_discard;
  tracked_objects_.clear();
  deformation_th_ = deformation_th;
  ncount_ = 0;
}

void ObjectTracker::Reset() {
  tracked_objects_.clear();
  ncount_ = 0;
  removed_count_ = 0;
}

void ObjectTracker::Update(const std::vector<DetectionDecoder::Object> &objects,
                           int obj_label) {
  std::vector<TrackedObject> new_objects;
  TrackedObject obj;
  for (auto &o : objects) {
    if (o.label != obj_label) continue;
    obj.id = kInitial;
    obj.last_frame = 0;
    obj.frames_count = 0;
    obj.frames_to_count = frames_to_count_;
    obj.rect = o.rect;
    obj.center.x = o.rect.x + 0.5 * o.rect.width;
    obj.center.y = o.rect.y + 0.5 * o.rect.height;
    new_objects.push_back(obj);
  }
  AddDetectionsInternal(new_objects);
}

int ObjectTracker::GetAcceptableObjectsCount() const {
  return std::count_if(
      tracked_objects_.begin(), tracked_objects_.end(),
      [](const TrackedObject &obj) { return obj.IsAcceptableDetection(); });
}

void ObjectTracker::Draw(cv::Mat &image, bool only_recents) const {
  for (const auto &obj : tracked_objects_) {
    if (  only_recents && !obj.InCurrentFrame() ) continue;

    cv::Scalar color = cv::Scalar(0, 0, 0);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }
    cv::rectangle(image, obj.rect,
                  cv::Scalar((17 * obj.id) % 256, (7 * obj.id) % 256,
                             (37 * obj.id) % 256),
                  2);

    char text[256];
    if (obj.InCurrentFrame()){
        sprintf(text, "id-%d", obj.id);
    } else {
        sprintf(text, "id-%d || ", obj.id);
    }

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    if (y > image.rows) y = image.rows;
    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color, -1);
    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
}

ulong ObjectTracker::GenNewUniqueId() { return std::max(ncount_++, ulong(0)); }

void ObjectTracker::AddDetectionsInternal(
    std::vector<TrackedObject> &new_objects) {
  std::vector<int> old_to_new(tracked_objects_.size());
  std::vector<float> distances(tracked_objects_.size());
  std::vector<int> new_to_old(new_objects.size());

  // best match from `tracked_objects_` to `new_objects`
  for (size_t ii = 0; ii < tracked_objects_.size(); ++ii) {
    TrackedObject old_obj = tracked_objects_[ii];
    int best_index = -1;
    float best_dist = kMaxFloat32;
    for (size_t jj = 0; jj < new_objects.size(); ++jj) {
      TrackedObject &new_obj = new_objects[jj];
      const float dist = Distance(new_obj, old_obj);
      if (dist < best_dist) {
        best_dist = dist;
        best_index = jj;
      }
    }
    old_to_new[ii] = best_index;
    distances[ii] = best_dist;
  }

  // best match from `new_objects` to `tracked_objects_`
  for (size_t ii = 0; ii < new_objects.size(); ++ii) {
    TrackedObject new_obj = new_objects[ii];
    int best_index = -1;
    float best_dist = kMaxFloat32;
    for (size_t jj = 0; jj < tracked_objects_.size(); ++jj) {
      TrackedObject &old_obj = tracked_objects_[jj];
      const float dist =
          Distance(new_obj, old_obj);  // IoU(new_obj.box, old_obj.box);
      if (dist < best_dist) {
        best_dist = dist;
        best_index = jj;
      }
    }
    new_to_old[ii] = best_index;
  }

  for (int i = 0; i < static_cast<int>(tracked_objects_.size()); ++i) {
    int j = old_to_new[i];
    if (j != -1 && i == new_to_old[j] && distances[i] < kMaxFloat32) {
      tracked_objects_[i].rect = new_objects[j].rect;
      tracked_objects_[i].center = new_objects[j].center;
      tracked_objects_[i].last_frame = 0;
      tracked_objects_[i].frames_count++;
      new_objects[j].id = kInvalid;
    } else {
      tracked_objects_[i].last_frame += 1;
      if (tracked_objects_[i].last_frame >= frames_to_discard_) {
        tracked_objects_[i].id = kInvalid;
      }
    }
  }

  removed_objects_.clear();
  for (auto iter = tracked_objects_.begin(); iter != tracked_objects_.end();) {
    if (iter->id == kInvalid) {
      std::iter_swap(iter, tracked_objects_.end() - 1);

      const TrackedObject &obj = *(tracked_objects_.end() - 1);
      if (obj.IsAcceptableDetection()) {
        removed_objects_.push_back(*(tracked_objects_.end() - 1));
      }
      tracked_objects_.erase(tracked_objects_.end() - 1);
      continue;
    }
    ++iter;
  }
  removed_count_ += removed_objects_.size();

  for (auto &d : new_objects) {
    if (d.id == kInvalid) continue;
    d.id = GenNewUniqueId();
    d.last_frame = 0;
    d.frames_count = 1;
    tracked_objects_.push_back(d);
  }
}

float ObjectTracker::Distance(const TrackedObject &obj1,
                              const TrackedObject &obj2) {
  const int area1 = obj1.rect.area();
  const int area2 = obj2.rect.area();
  const float iarea = (obj1.rect & obj2.rect).area();
  const float iou = iarea / (area1 + area2 - iarea);

  if (iou <= deformation_th_) {
    return kMaxFloat32;
  }
//  const float dx = obj1.center.x - obj2.center.x;
//  const float dy = obj1.center.y - obj2.center.y;
//  return dx * dx + dy * dy;
  return 1.0-iou;
}

void ObjectTracker::MinMax(int a, int b, int &minVal, int &maxVal) {
  if (a < b) {
    minVal = a;
    maxVal = b;
  } else {
    minVal = b;
    maxVal = a;
  }
}
