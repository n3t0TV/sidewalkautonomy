#include <iostream>
#include <opencv2/opencv.hpp>

#include "funct_utils.h"
#include "modules/state_msg.h"
#include "modules/vp_whereiam_msg.h"
#include "ros/ros.h"
#include "std_msgs/String.h"

using namespace std;
/*
bool Draw(cv::Mat &frame, float xl, float xr, float th = 0.2f) {
  static int wk = 1;

  const float blen = frame.cols / 1.0;
  const float o = (xl + xr) / 2.0 - 0.5;
  int h = 20;
  int xc = frame.cols / 2;

  cv::Scalar color =
      (std::abs(o) < th) ? cv::Scalar{0, 255, 0} : cv::Scalar{0, 0, 255};
  const int x0 = static_cast<int>(xc + o * blen);
  const int w = static_cast<int>(std::abs(o) * blen);

  if (o < 0.0) {
    cv::rectangle(frame, cv::Rect{x0, 0, w, h}, color, -1);
    if (std::abs(o) > th)
      cv::fillPoly(
          frame, std::vector<cv::Point>{{x0 - h / 2, h / 2}, {x0, 0}, {x0, h}},
          color, 0);
  } else {
    cv::rectangle(frame, cv::Rect{xc, 0, w, h}, color, -1);
    if (std::abs(o) > th)
      cv::fillPoly(frame,
                   std::vector<cv::Point>{
                       {xc + w + h / 2, h / 2}, {xc + w, 0}, {xc + w, h}},
                   color, 0);
  }

  xl *= frame.cols;
  xr *= frame.cols;
  float y = frame.size().height / 2.0;
  cv::rectangle(frame, cv::Rect(xl - 5, y - 5, 10, 10), {123, 4, 67}, -1);
  cv::rectangle(frame, cv::Rect(xr - 5, y - 5, 10, 10), {123, 124, 0}, -1);

  cv::imshow("DRAW", frame);
  int key = cv::waitKey(wk);
  if (key == 27) return false;
  if (key == 'c') wk = wk == 1 ? -1 : 1;
  return true;
}
*/

/*
INACTIVE = 0,
COMMAND = 1,
DRIVE = 2,
PARKED = 3,
VENDING = 4,
TELEOP = 5,
JOYSTICK = 6,
NOT_REGISTERED=7
*/

#define TELEOP 5
static int wagon_status = -1;

void StatusCallback(const modules::state_msg msg) { wagon_status = msg.status; }

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "WhereIAm");

  const auto jobj = LoadJSon(argv[1]);
  shared_ptr<Classifier> whereiam_cls;
  shared_ptr<Predictor> sidewalk_dir_predictor;
  for (const auto &o : jobj["models"]) {
    if (o["name"] == "WhereIam") {
      whereiam_cls = ClassifierFromJson(o);
      continue;
    }
    if (o["name"] == "SidewalkDir") {
      sidewalk_dir_predictor = PredictorFromJson(o);
      continue;
    }
  }

  const ulong skip_frames = jobj.value("skip_frames", 0) + 1;

  const string video_source = jobj.value("video_source", "");
  cv::VideoCapture cap(video_source);

  if (!cap.isOpened()) {
    cerr << "Error opening the device: " << video_source << endl;
    return -1;
  }

  cv::Mat frame;
  ros::NodeHandle nodeHandle;
  ros::Publisher whereiam_pub = nodeHandle.advertise<modules::vp_whereiam_msg>(
      "video_processing/whereiam", 1);

  ros::Subscriber statusSubscriber =
      nodeHandle.subscribe("status_topic", 10, &StatusCallback);

  ros::Rate loop_rate(100);

  modules::vp_whereiam_msg whereiam_msg;
  ulong nframes = 0;
  std::vector<cv::Mat> outputs;
  cv::Mat roi;
  while (ros::ok()) {
    if (!cap.read(frame)) {
      cerr << "Error reading from device: " << video_source << endl;
      break;
    }

    loop_rate.sleep();

    if (wagon_status == TELEOP && nframes % skip_frames == 0) {
      roi = Get4x3Roi(frame);
      if (whereiam_cls) {
        whereiam_cls->Predict(roi);
        auto p = whereiam_cls->GetTracker().GetIdScorePairRecent(0);
        whereiam_msg.cls_id = p.first;
        whereiam_msg.cls_confidence = p.second;
      }

      if (sidewalk_dir_predictor && whereiam_msg.cls_id == 2) {
        sidewalk_dir_predictor->Predict(roi, outputs);
        whereiam_msg.lpt = outputs[0].ptr<float>()[0];
        whereiam_msg.rpt = outputs[0].ptr<float>()[1];

        whereiam_msg.cdir =
            ((whereiam_msg.lpt + whereiam_msg.rpt) / 2.0 - 0.5) / 0.5;

        // debug code
        // if (!Draw(roi, whereiam_msg.lpt, whereiam_msg.rpt)) break;
      }
      whereiam_pub.publish(whereiam_msg);
    }
    ros::spinOnce();
    ++nframes;
  }
  return 0;
}
