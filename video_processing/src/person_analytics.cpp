#include "person_analytics.h"

#include "funct_utils.h"
#include "ssd_decoder.h"
#include "yolox_decoder.h"
using namespace std;

namespace {
// TODO(rbt): put this function in funct_utils.h

std::string GetFormatedDateTime(
    const std::chrono::system_clock::time_point &tt) {
  auto in_time_t = std::chrono::system_clock::to_time_t(tt);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

}  // namespace

shared_ptr<PersonAnalytics> PersonAnalytics::Create(
    const nlohmann::json &config) {
  PersonAnalytics *impl = new PersonAnalytics();
  impl->Init(config);
  return shared_ptr<PersonAnalytics>{impl};
}

void PersonAnalytics::Init(const nlohmann::json &config) {
  for (const auto &o : config["models"]) {
    if (o["type"] == "detection" && o["name"] == "yolox") {
      person_detector_ = PredictorFromJson(o);

      pdet_decoder_ = DetectionDecoderFromJson(o);
      pdet_decoder_->SetLabels(StringListFromJson(o["labels"]));

      if (pdet_decoder_->GetNormalizationFunct()) {
        person_detector_->SetInputParamsNorm(
            1.0, {0, 0, 0}, false, pdet_decoder_->GetNormalizationFunct());
      }

      ObjectTrackerFromJson(o, person_tracker_);
      continue;
    }

// disable QR detection
//    if (o["type"] == "detection" && o["name"] == "QRDetector") {
//      qr_detector_ = PredictorFromJson(o);

//      qr_decoder_ = DetectionDecoderFromJson(o);
//      qr_decoder_->SetLabels(StringListFromJson(o["labels"]));

//      if (qr_decoder_->GetNormalizationFunct()) {
//        qr_detector_->SetInputParamsNorm(1.0, {0, 0, 0}, false,
//                                         qr_decoder_->GetNormalizationFunct());
//      }

//      ObjectTrackerFromJson(o, qr_tracker_);
//      continue;
//    }
  }

  qr_skip_frames_ = skip_frames_ = config.value("skip_frames", 0) + 1;
  time_interval_ = config.value("time_interval", 60);
  person_id_ = config.value("person_id", 0);
  nearby_det_th_ = config.value("nearby_det_th", 0.0);
}

void PersonAnalytics::Reset() {
  last_tpoint_ = beg_tpoint_ = chrono::system_clock::now();
  person_tracker_.Reset();
  qr_tracker_.Reset();
  curr_no_persons_ = 0;
  total_no_persons_ = 0;
  total_no_persons_5min_ = 0;
  frames_count_ = 0;
  data_ready_ = false;
  qr_curr_id_ = -1;
  qr_skip_frames_ = skip_frames_;
  data_ready_ = data_ready_5min_ = false;
  dt_5min_end_ = GetFormatedDateTime(last_tpoint_);
}

void PersonAnalytics::ProcessFrame(const cv::Mat &bgr, const cv::Mat gray) {
  vector<cv::Mat> outputs;
  vector<DetectionDecoder::Object> detected_objects;

// disable QR detector
//  if (qr_active_ && frames_count_ % qr_skip_frames_ == 0) {
//    // QR part
//    qr_detector_->Predict(bgr, outputs);
//    qr_decoder_->Decode(outputs, detected_objects, bgr.size());
//    qr_tracker_.Update(detected_objects, 1);
//    cv::Rect rr;
//    for (auto &o : qr_tracker_.GetTrackedObjects()) {
//      if (o.InCurrentFrame() && o.IsAcceptableDetection()) {
//        if (o.id == qr_curr_id_) {
//          break;
//        } else {
//          rr = o.rect;
//          ScaleRect(rr, 1.2);
//          rr = RectInsideFrame(rr, bgr);
//          qr_curr_msg_ = zbarqr_decoder_.DecodeSingle(bgr(rr));
//          if (!qr_curr_msg_.empty()) qr_curr_id_ = o.id;
//        }
//      }
//    }
//    if (qr_tracker_.GetTrackedObjects().empty()) qr_curr_msg_ = "";
//    if (!detected_objects.empty()) {
//      qr_skip_frames_ = 1;
//      return;
//    }
//  }
//  qr_curr_msg_ = "";
//  qr_skip_frames_ = skip_frames_;

  // Person part
  if (person_active_) {
    person_detector_->Predict(bgr, outputs);
    pdet_decoder_->Decode(outputs, detected_objects, bgr.size());

    OnlyNearbyDetections(detected_objects,
                         static_cast<int>(nearby_det_th_ * bgr.rows));

    person_tracker_.Update(detected_objects, person_id_);

    const int no_persons = person_tracker_.GetAcceptableObjectsCount();
    data_ready_ = no_persons != curr_no_persons_;

    total_no_persons_ = person_tracker_.GetRemovedCount() + no_persons;
    curr_no_persons_ = no_persons;

    auto now = chrono::system_clock::now();
    elapsed_time_seg_ =
        chrono::duration_cast<chrono::seconds>(now - beg_tpoint_).count();

    auto tt =
        chrono::duration_cast<chrono::seconds>(now - last_tpoint_).count();
    data_ready_5min_ = false;
    if (tt >= 5 * 60) {
      total_no_persons_5min_ = (total_no_persons_ - last_total_no_persons_);
      last_total_no_persons_ = total_no_persons_;
      last_tpoint_ = now;

      dt_5min_beg_ = dt_5min_end_;
      dt_5min_end_ = GetFormatedDateTime(now);

      data_ready_5min_ = true;
    }
  }
  ++frames_count_;
}

void PersonAnalytics::ProcessFrameFollowPerson(const cv::Mat &bgr, const cv::Mat gray){
    vector<cv::Mat> outputs;
    vector<DetectionDecoder::Object> detected_objects;
    person_detector_->Predict(bgr, outputs);
    pdet_decoder_->Decode(outputs, detected_objects, bgr.size());
    person_tracker_.Update(detected_objects, person_id_);
}

void PersonAnalytics::SetActive(bool qr, bool person) {
  qr_active_ = qr;
  if (person_active_ != person) {
    Reset();
    person_active_ = person;
  }
}

void PersonAnalytics::OnlyNearbyDetections(
    vector<DetectionDecoder::Object> &objs, int hth) {
  for (auto &&o : objs) {
    if (o.rect.y + o.rect.height < hth) o.label = -1;
  }
}
