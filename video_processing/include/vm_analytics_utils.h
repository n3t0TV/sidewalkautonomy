#ifndef VM_ANALYTICS_UTILS_H
#define VM_ANALYTICS_UTILS_H

#include "modules/autopilot_state_msg.h"
#include "modules/containers_msg.h"
#include "modules/enable_autopilot.h"
#include "modules/mqtt_publishers_msg.h"
#include "modules/speaker_msg.h"
#include "person_analytics.h"
#include "ros/ros.h"
#include "modules/vp_whereiam_msg.h"
#include "funct_utils.h"

class VMMessagesManager {
  ros::NodeHandle *node_;
  enum State { kWaiting, kTracking, kTargetLossed } current_state_{kTargetLossed};

 public:
  void Init(ros::NodeHandle *node, const nlohmann::json &config);
  bool ProcessQRMessage(const std::shared_ptr<PersonAnalytics> p_analytics);
  void ProcessPersonEvents(const std::shared_ptr<PersonAnalytics> p_analytics);
  void ResetFollowPerson(bool enable);
  long GetPersonTrackedId() const {return lperson_id_;}
  void FollowPerson(const std::shared_ptr<PersonAnalytics> p_analytics, const cv::Size &frameSize);

 private:
  std::string GetAudioIdForBatteryStatus();
  std::vector<std::string> GetAudioIdsForContainerStatus(int id);

  bool SendJoystickMacAddress(const std::string &mac);
  void LaunchQRDetectedAudio();
  std::string BatteryLevelToAudioId(int bl);


  ros::Publisher speaker_pub_;
  ros::Publisher mqtt_pub_;

  modules::speaker_msg speaker_msg_;
  modules::enable_autopilot autopilot_msg_;
  modules::mqtt_publishers_msg mqtt_msg_;

  ros::ServiceClient sensor_srv_client_;
  ros::ServiceClient container1_srv_client_;
  ros::ServiceClient container2_srv_client_;
  ros::ServiceClient joystick_pairing_;


  std::string last_qr_str_{""};


  // FollowTheLeader
  ros::Publisher follow_person_pub_;
  ros::ServiceClient enable_autopilot_;
  modules::vp_whereiam_msg ctr_msg_;
  float cdirTh0_{0.2};
  float velTh0_{0.1};
  float velTh1_{0.1};
  float velTh2_{0.1};
  float deltaVelTh_{0.1};

  int no_attemps_{0};
  int no_attemps_th_{0};
  long lperson_id_{-1};
  float last_vel_{0};

  int GetCandidateTarget(const ObjectTracker &pTracker, const cv::Size &frameSize);
  bool SendCtr(const ObjectTracker &pTracker, const cv::Size &frameSize);
  void StopAutopilot();
  SmootherAve vel_smother_;
  SmootherAve dir_smother_;    
};

#endif  // VM_ANALYTICS_UTILS_H
