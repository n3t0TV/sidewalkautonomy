#ifndef PERSON_ANALYTICS_H
#define PERSON_ANALYTICS_H
#include "detection_decoder.h"
#include "nlohmann/json.hpp"
#include "object_tracker.h"
#include "predictor.h"
#include "zbarqrdecoder.h"

class PersonAnalytics {
 public:
  static std::shared_ptr<PersonAnalytics> Create(const nlohmann::json &config);

  PersonAnalytics() = default;
  void Init(const nlohmann::json &config);
  void Reset();
  void ProcessFrame(const cv::Mat &bgr_frame, const cv::Mat gray_frame = {});
  void ProcessFrameFollowPerson(const cv::Mat &bgr, const cv::Mat gray = {});


  int GetCurrPersonCount() const { return curr_no_persons_; }

  bool IsDataReady() const { return data_ready_; }
  bool IsDataReady5Min() const { return data_ready_5min_; }

  int GetTotalPersonCount() const { return total_no_persons_; }
  int GetTotalPersonCount5Min() const { return total_no_persons_5min_; }

  std::string GetDateTime5MinBeg() const { return dt_5min_beg_; }
  std::string GetDateTime5MinEnd() const { return dt_5min_end_; }

  ulong GetElapsedTime() const { return elapsed_time_seg_; }
  const ObjectTracker &GetPersonTracker() const { return person_tracker_; }
  const ObjectTracker &GetQRTracker() const { return qr_tracker_; }
  std::string GetQRMessage() const { return qr_curr_msg_; }
  void SetActive(bool qr, bool person);

 private:
  std::shared_ptr<Predictor> person_detector_;
  std::shared_ptr<DetectionDecoder> pdet_decoder_;
  ObjectTracker person_tracker_;

  std::shared_ptr<Predictor> qr_detector_;
  std::shared_ptr<DetectionDecoder> qr_decoder_;
  ObjectTracker qr_tracker_;

  int skip_frames_;
  ulong frames_count_;
  double time_interval_{60.0};
  int person_id_{-1};
  float nearby_det_th_{0.25};

  int curr_no_persons_;
  int total_no_persons_;
  int total_no_persons_5min_;

  int last_total_no_persons_;
  std::string dt_5min_beg_;
  std::string dt_5min_end_;
  std::string datetime_str5min_;

  ulong elapsed_time_seg_;
  std::chrono::system_clock::time_point last_tpoint_, beg_tpoint_;
  bool data_ready_{false};
  bool data_ready_5min_{false};

  long qr_curr_id_;
  int qr_skip_frames_;
  ZBarQRDecoder zbarqr_decoder_;
  std::string qr_curr_msg_;
  void OnlyNearbyDetections(std::vector<DetectionDecoder::Object> &objs,
                            int hth);
  bool qr_active_{true};
  bool person_active_{true};
};

#endif  // PERSON_ANALYTICS_H
