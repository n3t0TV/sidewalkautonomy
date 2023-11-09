#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "funct_utils.h"
#include "modules/state_msg.h"
#include "person_analytics.h"
#include "ros/ros.h"
#include "vm_analytics_utils.h"

using namespace std;

#define TELEOP 5

static int wagon_status = TELEOP;
static bool follow_person_enable = false;

void StatusCallback(
    const modules::state_msg msg) { /*wagon_status = msg.status;*/
}

void AutoPilotStatusChangedCallback(const modules::autopilot_state_msg &msg) {
  follow_person_enable = msg.enable;
}
/*
bool Draw(cv::Mat roi, shared_ptr<PersonAnalytics> pa) {
  static int wk = -1;

  auto pTracker = pa->GetPersonTracker();
  auto qrTracker = pa->GetQRTracker();
  pTracker.Draw(roi, false);
  qrTracker.Draw(roi);

  if (qrTracker.GetTrackedObjects().size()) {
    cv::Rect r = qrTracker.GetTrackedObjects()[0].rect;
    float scale = 1.2;
    const int w = static_cast<int>(r.width * scale + 0.5);
    const int h = static_cast<int>(r.height * scale + 0.5);
    r.x = r.x + (r.width - w) / 2;
    r.y = r.y + (r.height - h) / 2;
    r.width = w;
    r.height = h;
    cv::rectangle(roi, r, {0, 255, 0}, 3);
  }
  cv::putText(roi, "no. persons: " + to_string(pa->GetCurrPersonCount()),
              {10, roi.rows - 40}, 3, 1, {0, 0, 255});
  cv::putText(roi, "total persons: " + to_string(pa->GetTotalPersonCount()),
              {10, roi.rows - 20}, 3, 1, {0, 0, 255});
  cv::imshow("Persons", roi);
  int key = cv::waitKey(wk);
  if (key == 27) return false;
  if (key == 'c') wk = wk == 1 ? -1 : 1;
  return true;
}

bool DrawFollowPerson(cv::Mat roi, shared_ptr<PersonAnalytics> pa, long pId) {
    static int wk = -1;

    auto pTracker = pa->GetPersonTracker();
    pTracker.Draw(roi, false);

    float dist=-1;
    float dir=0;
    for (auto &p : pTracker.GetTrackedObjects()){
        if (p.id == pId){
            cv::circle(roi, p.center,10,{0,0,255},2,-1);
            cv::line(roi, p.center-cv::Point2f{10,0}, p.center+cv::Point2f{10,0}, {0,0,255},2);
            cv::line(roi, p.center-cv::Point2f{0,10}, p.center+cv::Point2f{0,10}, {0,0,255},2);

            float w2 = roi.cols/2;
            float h = roi.rows;
            dist = 1.0-(p.rect.y + p.rect.height)/h;
            dir = (p.center.x-w2)/w2;
            break;

        }
    }

    if (pId!=-1){
    cv::putText(roi, "Distance  : " + to_string(dist), {10, roi.rows - 40}, 3, 1, {0, 0, 255});
    cv::putText(roi, "Direction : " + to_string(dir), {10, roi.rows - 20}, 3, 1, {0, 0, 255});
    }
    cv::imshow("Persons", roi);
    int key = cv::waitKey(wk);
    if (key == 27) return false;
    if (key == 'c') wk = wk == 1 ? -1 : 1;
    return true;
}
*/

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "VMAnalytics");
  ros::NodeHandle node;
  VMMessagesManager vm_msg_manager;
  shared_ptr<PersonAnalytics> p_analytics;
  string video_source;
  {
    const auto jobj = LoadJSon(argv[1]);
    p_analytics = PersonAnalytics::Create(jobj);
    video_source = jobj.value("video_source", "");
    vm_msg_manager.Init(&node, jobj);
  }



  // subcribers
  ros::Subscriber status_sub =
      node.subscribe("status_topic", 10, &StatusCallback);
  ros::Subscriber autopilot_status_sub_ = node.subscribe(
      "autopilot_status_topic", 1, &AutoPilotStatusChangedCallback);

  ros::Rate loop_rate(100);

  cv::Mat frame, roi;
  int current_status = TELEOP;
  bool last_follow_person_enable = follow_person_enable;

  string last_qr_str = "";
  ROS_INFO_STREAM("Capturing from video device: " << video_source);
  cv::VideoCapture cap(video_source);
  if (!cap.isOpened()) {
    ROS_INFO_STREAM("    Error from video device: " << video_source);
  }

  // disable person analytics
  p_analytics->SetActive(false, true);
  //cap.read(frame);
  //cv::VideoWriter ovideo("/home/tortoise/bboxes.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), static_cast<int>(15), frame.size(),true);

  cv::TickMeter tictac;
  long nframe = 0;
  while (ros::ok()) {
    if (!cap.read(frame)) {
      ROS_ERROR_STREAM("Error getting frame from device: " << video_source);
      ROS_INFO_STREAM("Traying to reopen device: '" << video_source
                                                    << "' again");
      std::this_thread::sleep_for(std::chrono::seconds(1));
      cap.open(video_source);
      if (cap.isOpened()) p_analytics->Reset();
      break;
      continue;
    }
    loop_rate.sleep();
    
    tictac.start();
    roi = Get4x3Roi(frame);
    if (current_status != wagon_status) {
      p_analytics->Reset();
      current_status = wagon_status;
    }

    if (current_status == TELEOP) {
      if (last_follow_person_enable != follow_person_enable) {
        ROS_INFO_STREAM("FOLLOW PERSON MODE: "
                        << (follow_person_enable ? "ENABLE" : "DISABLE"));
        last_follow_person_enable = follow_person_enable;
        vm_msg_manager.ResetFollowPerson(follow_person_enable);
        p_analytics->Reset();
      }

      if (follow_person_enable) {
        p_analytics->ProcessFrameFollowPerson(roi, {});
        vm_msg_manager.FollowPerson(p_analytics, roi.size());
      } else {
        p_analytics->ProcessFrame(roi, {});
        vm_msg_manager.ProcessPersonEvents(p_analytics);

        // disable QR detector
        //vm_msg_manager.ProcessQRMessage(p_analytics);
      }
    }
    tictac.stop();
    ++nframe;
    if (nframe%1000==0){
        ROS_INFO_STREAM("Frames count: " << nframe << " FPS (last 1000 frames): " << tictac.getFPS());            
        tictac.reset();
    }
    /*
    if (follow_person_enable==false){
        if (!Draw(roi, p_analytics)) break;
    } else {
        if (!DrawFollowPerson(roi, p_analytics, vm_msg_manager.GetPersonTrackedId())) break;
    }
    */

    //ovideo.write(frame);

    ros::spinOnce();
  }
  //ovideo.release();
  return 0;
}
