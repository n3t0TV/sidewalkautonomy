#ifndef OJECTT_RACKER_H
#define OJECTT_RACKER_H
#include <opencv2/core.hpp>

#include "detection_decoder.h"

class ObjectTracker {
 public:
  enum SpecialIds { kInvalid = -2, kInitial = -1 };
  struct TrackedObject {
    bool InCurrentFrame() const { return last_frame == 0; }
    bool IsAcceptableDetection() const {
      return frames_count >= frames_to_count;
    }
    int id;
    int last_frame;
    int frames_count;
    cv::Rect rect;
    cv::Point2f center;
    int frames_to_count;
  };

  ObjectTracker() = default;
  void Init(int frames_to_count = 3, int frames_to_discard = 5,
            float deformation_th = 0.5f);
  void Reset();
  void Update(const std::vector<DetectionDecoder::Object> &objects,
              int obj_label);

  const std::vector<TrackedObject> &GetTrackedObjects() const {
    return tracked_objects_;
  }
  const std::vector<TrackedObject> &GetRemovedObjects() const {
    return removed_objects_;
  }

  int GetAcceptableObjectsCount() const;
  int GetRemovedCount() const { return removed_count_; }

  void Draw(cv::Mat &frame, bool only_recents=true) const;

 private:
  int frames_to_discard_{3};
  ulong ncount_{0};
  int frames_to_count_{0};
  float deformation_th_{0.4};
  int removed_count_{0};

  std::vector<TrackedObject> tracked_objects_;
  std::vector<TrackedObject> removed_objects_;

  // methods
  ulong GenNewUniqueId();
  void AddDetectionsInternal(std::vector<TrackedObject> &dets);
  float Distance(const TrackedObject &obj1, const TrackedObject &obj2);

  inline void MinMax(int a, int b, int &minVal, int &maxVal);
};

#endif  // OJECTT_RACKER_H
