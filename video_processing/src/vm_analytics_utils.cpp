#include "vm_analytics_utils.h"
#include "modules/ContainerFeedback.h"
#include "modules/mac_service.h"
#include "modules/sensors_service.h"

using namespace std;

namespace {
string EncodeMessageIntoJSon(const shared_ptr<PersonAnalytics> p_analytics) {
  nlohmann::json d;
  d["no_persons"] = p_analytics->GetCurrPersonCount();
  auto pTracker = p_analytics->GetPersonTracker();
  auto arr = nlohmann::json::array();
  for (const auto &obj : pTracker.GetTrackedObjects()) {
    if (obj.IsAcceptableDetection()) {
      int x = obj.rect.x;
      int y = obj.rect.y;
      int w = obj.rect.width;
      int h = obj.rect.height;
      arr.push_back(nlohmann::json::array({x, y, w, h}));
    }
  }
  d["bboxes"] = arr;
  return d.dump();
}

string EncodeMessageIntoJSonDB(const shared_ptr<PersonAnalytics> p_analytics) {
  static uchar beg = 1;

  nlohmann::json d;
  d["totalpersons"] = p_analytics->GetTotalPersonCount5Min();
  d["starttime"] = p_analytics->GetDateTime5MinBeg();
  d["endtime"] = p_analytics->GetDateTime5MinEnd();
  d["beg"] = beg;
  beg = 0;
  return d.dump();
}
}  // namespace

void  VMMessagesManager::Init(ros::NodeHandle *node, const nlohmann::json &config){

    node_ = node;
    speaker_pub_ = node_->advertise<modules::speaker_msg>("speaker_topic", 1);
    mqtt_pub_ =
        node_->advertise<modules::mqtt_publishers_msg>("mqtt_publishers", 1);
    sensor_srv_client_ =
        node_->serviceClient<modules::sensors_service>("sensors_service", false);

    container1_srv_client_ =
        node_->serviceClient<modules::ContainerFeedback>("feedback_srv_1", false);
    container2_srv_client_ =
        node_->serviceClient<modules::ContainerFeedback>("feedback_srv_2", false);

    joystick_pairing_ =
        node_->serviceClient<modules::mac_service>("joystick_mac_service", false);

    follow_person_pub_ =
        node_->advertise<modules::vp_whereiam_msg>("follow_person_topic", 1);

    enable_autopilot_ = node_->serviceClient<modules::enable_autopilot>(
        "followme_autopilot", false);

    // follow person
    const auto fc = config["follow_person"];
    no_attemps_th_ = fc.value("no_attemps",10);
    velTh0_ = fc.value("velTh0", 0.1F);
    ROS_INFO_STREAM(" velTh0 : " << velTh0_);

    velTh1_ = fc.value("velTh1", 0.1F);
    ROS_INFO_STREAM(" velTh1 : " << velTh1_);
    
    velTh2_ = fc.value("velTh2", 0.8F);
    ROS_INFO_STREAM(" velTh2 : " << velTh2_);
    
    cdirTh0_ = fc.value("cdirTh0", 0.2F);
    ROS_INFO_STREAM(" cdirTh0 : " << cdirTh0_);
    
    
    deltaVelTh_ = fc.value("deltaVel", 0.01F);
    ROS_INFO_STREAM(" deltaVel : " << deltaVelTh_);

    int nave = fc.value("nvel", 1);
    vel_smother_.Init(nave,0.0);

    nave = fc.value("ndir", 1);
    dir_smother_.Init(nave,0.0);
}

bool VMMessagesManager::ProcessQRMessage(
    const std::shared_ptr<PersonAnalytics> p_analytics) {
  static const string cmds[] = {"uc", "up", "bl", "ct1", "ct2", "joystick"};

  const string qr_str = p_analytics->GetQRMessage();
  // ROS_INFO_STREAM("QR" << qr_str);
  if (qr_str == last_qr_str_) return false;

  last_qr_str_ = qr_str;
  if (last_qr_str_.empty()) return true;

  const vector<string> subs = Split(qr_str, '/');
  for (size_t i = 0; i < sizeof(cmds) / sizeof(cmds[0]); ++i) {
    if (subs[0] == cmds[i]) {
      nlohmann::json d;

      // uc/sku/container_id/mac
      if (i == 0) {
        if (subs.size() != 4) {
          ROS_ERROR_STREAM(
              "UpdateContainer QR message is incorrectly formatted");
          return false;
        }
        LaunchQRDetectedAudio();
        ROS_INFO_STREAM("UpdateContainer QR message ... ");
        // mqtt message
        mqtt_msg_.mqtt_topic = "sensor/imei/updateContainer";
        d["mac"] = subs[3];  // mac
        d["sku"] = subs[1];  // sku
        mqtt_msg_.raw_msg = d.dump();
        mqtt_pub_.publish(mqtt_msg_);
        return true;
      }

      // up/{SKU}/{containerNumber}/{cantidadProducto}/{nombreDelProducto}
      if (i == 1) {
        if (subs.size() != 5) {
          ROS_ERROR_STREAM("UpdateProduct QR message is incorrectly formatted");
          return false;
        }
        LaunchQRDetectedAudio();
        ROS_INFO_STREAM("UpdateProduct QR message ... ");
        // mqtt message
        mqtt_msg_.mqtt_topic = "sensor/imei/updateProduct";
        d["name"] = subs[4];      // nombreDelProducto
        d["quantity"] = subs[3];  // cantidadProducto
        d["sku"] = subs[1];       // SKU
        mqtt_msg_.raw_msg = d.dump();
        mqtt_pub_.publish(mqtt_msg_);
        return true;
      }

      // battery status
      if (i == 2) {
        if (subs.size() != 1) {
          ROS_ERROR_STREAM("BatteryStatus QR message is incorrectly formatted");
          return false;
        }
        ROS_INFO_STREAM("BatteryStatus QR message ... ");
        LaunchQRDetectedAudio();
        speaker_msg_.mp3Id = GetAudioIdForBatteryStatus();
        if (!speaker_msg_.mp3Id.empty()) {
          speaker_pub_.publish(speaker_msg_);
        }
        return true;
      }

      // container 1&2 info
      if (i == 3 || i == 4) {
        if (subs.size() != 1) {
          ROS_ERROR_STREAM("Container QR message is incorrectly formatted");
          return false;
        }
        ROS_INFO_STREAM("Container QR message ... ");
        LaunchQRDetectedAudio();
        const auto audios_ids = GetAudioIdsForContainerStatus(i - 2);
        for (const auto &id : audios_ids) {
          speaker_msg_.mp3Id = id;
          if (!speaker_msg_.mp3Id.empty()) {
            speaker_pub_.publish(speaker_msg_);
          }
        }
        return true;
      }

      // joystick
      if (i == 5) {
        if (subs.size() != 2) {
          ROS_ERROR_STREAM(
              "Joystick pairing QR message is incorrectly formatted");
          return false;
        }
        ROS_INFO_STREAM("Joystick pairing QR message ... ");        
        LaunchQRDetectedAudio();
        SendJoystickMacAddress(subs[1]);
        return true;
      }
    }
  }
  return false;
}

void VMMessagesManager::ProcessPersonEvents(
    const std::shared_ptr<PersonAnalytics> p_analytics) {
  // personevent
  if (p_analytics->IsDataReady()) {
    mqtt_msg_.mqtt_topic = "sensor/imei/personevent";
    mqtt_msg_.qos = 0;
    mqtt_msg_.raw_msg = EncodeMessageIntoJSon(p_analytics);
    mqtt_pub_.publish(mqtt_msg_);
  }

  // totalpersons
  if (p_analytics->IsDataReady5Min()) {
    mqtt_msg_.mqtt_topic = "sensor/imei/totalpersons";
    mqtt_msg_.qos = 0;
    mqtt_msg_.raw_msg = EncodeMessageIntoJSonDB(p_analytics);
    mqtt_pub_.publish(mqtt_msg_);
  }
}

void VMMessagesManager::ResetFollowPerson(bool enable) {
    lperson_id_ = -1;
    last_vel_ = 0.0;
    current_state_ = enable ? kWaiting : kTargetLossed;
    no_attemps_ = 0;
    vel_smother_.Reset();
    //dir_smother_.Reset();
}

void VMMessagesManager::FollowPerson(const std::shared_ptr<PersonAnalytics> p_analytics, const cv::Size &frameSize)
{

    const ObjectTracker &pTracker = p_analytics->GetPersonTracker();
    switch (current_state_) {
    case kWaiting:        
        lperson_id_ = GetCandidateTarget(pTracker, frameSize);
        if (lperson_id_ != -1) {
            current_state_ = kTracking;
            no_attemps_ = 0;

            speaker_msg_.mp3Id = "105";
            speaker_pub_.publish(speaker_msg_);
            return;
        }
        ++no_attemps_;
        if (no_attemps_ >= no_attemps_th_) {
            current_state_ = kTargetLossed;
            no_attemps_ = 0;
            lperson_id_ = -1;
            StopAutopilot();
            ROS_INFO_STREAM("Failed selecting the target to follow after " << no_attemps_ << " attempts");
        }
        break;

    case kTracking:
        if (!SendCtr(pTracker, frameSize)) {
            current_state_ = kTargetLossed;
            lperson_id_ = -1;
            ROS_INFO_STREAM("Failed tracking: target is lossed");
        }
        break;

    case kTargetLossed:
        break;
    }
}

int VMMessagesManager::GetCandidateTarget(const ObjectTracker &pTracker,
                                        const cv::Size &frameSize) {
    const float fw = frameSize.width;
    const float fh = frameSize.height;
    const float fw2 = 0.5 * fw;

    float vel = 0;
    float cdir = 0;
    float best_cdir = cdirTh0_;
    long candidate_id = -1;
    for (auto &p : pTracker.GetTrackedObjects()) {
        if (p.InCurrentFrame() && p.IsAcceptableDetection()) {
            vel = 1.0 - (p.rect.y + p.rect.height) / fh;
            cdir = fabs(p.center.x - fw2) / fw2;

            if (vel < velTh0_ && cdir < best_cdir) {
                best_cdir = cdir;
                candidate_id = p.id;
            }
        }
    }
    return candidate_id;
}

bool VMMessagesManager::SendCtr(const ObjectTracker &pTracker,
                              const cv::Size &frameSize) {
    const float fh = frameSize.height;
    const float fw2 = 0.5 * frameSize.width;

    for (auto &p : pTracker.GetTrackedObjects()) {
        if (p.id == lperson_id_) {
            if (!p.InCurrentFrame()) return true;
            ctr_msg_.cdir = (p.center.x - fw2) / fw2;

                        
            const float vel = 1.0 - (p.rect.y + p.rect.height) / fh;
            //ROS_INFO_STREAM("RAW | Vel -> " << vel << " Dir -> " << ctr_msg_.cdir);
            
            if ( vel <= 0.65*velTh1_) {
                ctr_msg_.cdir = 0.0;                
                dir_smother_.Reset();
            }
            ctr_msg_.cdir = dir_smother_.Update(ctr_msg_.cdir);
            
            if (vel < velTh1_ || vel > velTh2_) {
                ctr_msg_.vel = 0.0; 
                vel_smother_.Reset();
            } else {
                ctr_msg_.vel = LinearMap(velTh1_, velTh2_, vel); 
                if (ctr_msg_.vel-last_vel_ > deltaVelTh_) {
                    ctr_msg_.vel = last_vel_ + deltaVelTh_;                    
                }
                ctr_msg_.vel = vel_smother_.Update(ctr_msg_.vel);
            }
            last_vel_ = ctr_msg_.vel;
            //ROS_INFO_STREAM("    | Vel -> " << ctr_msg_.vel << " Dir -> " << ctr_msg_.cdir);

            // send control
            follow_person_pub_.publish(ctr_msg_);
            return true;
        }
    }

    // send stop
    ctr_msg_.vel = ctr_msg_.cdir = 0.0;
    follow_person_pub_.publish(ctr_msg_);

    // disable autopilot
    StopAutopilot();
    return false;
}

void VMMessagesManager::StopAutopilot()
{
    // disable autopilot
    modules::enable_autopilot srv;
    srv.request.autopilot_enable=false;
    enable_autopilot_.call(srv);
    
}
    
std::string VMMessagesManager::GetAudioIdForBatteryStatus() {
  modules::sensors_service srv;
  if (sensor_srv_client_.call(srv)) {
    return BatteryLevelToAudioId(srv.response.battery);
  }
  return "btunknow";
}

std::vector<std::string> VMMessagesManager::GetAudioIdsForContainerStatus(
    int id) {
  auto client_ptr =
      (id == 1) ? &container1_srv_client_ : &container2_srv_client_;
  modules::ContainerFeedback srv;
  if (client_ptr->call(srv)) {
    auto data = nlohmann::json::parse(srv.response.feedback);
    bool is_connected = data.value<bool>("connection", false);
    int bl = data.value<int>("battery", -1);

    std::vector<std::string> result;
    if (id == 1) {
      result.push_back(is_connected ? "ct01" : "ct00");
    }
    if (id == 2) {
      result.push_back(is_connected ? "ct03" : "ct02");
    }

    if (is_connected) result.push_back(BatteryLevelToAudioId(bl));
    return result;
  }
  return {"ctunknow"};
}

std::string VMMessagesManager::BatteryLevelToAudioId(int bl) {
  bl = min(bl / 5, 19);
  char audio[32];
  sprintf(audio, "bl%02d", bl);
  return string(audio);
}

bool VMMessagesManager::SendJoystickMacAddress(const string &mac) {
  modules::mac_service srv;
  srv.request.mac_to_save = mac;
  if (joystick_pairing_.call(srv)) {
    if (srv.response.success) {
      return true;
    }
  }
  return false;
}

void VMMessagesManager::LaunchQRDetectedAudio() {
  speaker_msg_.mp3Id = "qrdetected";
  speaker_pub_.publish(speaker_msg_);
}
