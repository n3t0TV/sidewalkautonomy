#include "funct_utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "opencv_predictor.h"
#include "ssd_decoder.h"
#include "tensor_rt_predictor.h"
#include "yolox_decoder.h"

nlohmann::json LoadJSon(const std::string &fpath) {
  std::ifstream ifile(fpath);
  std::stringstream buffer;
  buffer << ifile.rdbuf();
  return nlohmann::json::parse(buffer.str());
}

std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf) {
  const string model_name = conf.value("name", "");
  const string model_type = conf.value("type", "");
  const string backend = conf.value("backend", "TRT");
  const string target = conf.value("target", "TRT");
  const string framework = conf.value("framework", "ONNX");

  const int input_w = conf.value("input_w", -1);
  const int input_h = conf.value("input_h", -1);
  const float scale = conf.value("scale", 1.0);
  const bool swap_ch = conf.value("swap_ch", false);

  const std::string model_path = conf.value("path", "");
  const std::string mpath_config = conf.value("path_config", "");

  const float mc1 = conf.value("mc1", 0.0f);
  const float mc2 = conf.value("mc2", 0.0f);
  const float mc3 = conf.value("mc3", 0.0f);

  std::cout << "Model name: " << model_name << std::endl;
  std::cout << "  Type         : " << model_type << std::endl;
  std::cout << "  Input Size   : [" << input_w << 'X' << input_h << ']'
            << std::endl;
  std::cout << "  Input Scale  :  " << scale << std::endl;
  std::cout << "  Input Mean   : [" << mc1 << ',' << mc2 << ',' << mc3 << ']'
            << std::endl;
  std::cout << "  Swap Channels:  " << swap_ch << std::endl;
  std::cout << "  Framework    :  " << framework << std::endl;
  std::cout << "  Target       :  " << target << std::endl;
  std::cout << "  Backend      :  " << backend << std::endl;
  std::cout << "  File Path    :  " << model_path << std::endl;

  std::shared_ptr<Predictor> result;
  if (backend == "TRT") {
    result = TensorRTPredictor::Create(model_path, framework);
  } else if (backend == "OCV") {
    result = OpenCVPredictor::Create(model_path, mpath_config, framework);
    auto p = dynamic_cast<OpenCVPredictor *>(result.get());
    if (target == "GPU") {
      //#ifdef HAVE_CUDA
      p->SetBackend(cv::dnn::DNN_BACKEND_CUDA);
      p->SetTargetDevice(cv::dnn::DNN_TARGET_CUDA);
      //#else
      //      std::cerr << "Your version of OpenCV does not support GPU
      //      inference. "
      //                   "Therefore, CPU inference will be used"
      //                << std::endl;
      //#endif
    } else {
      p->SetBackend(cv::dnn::DNN_BACKEND_OPENCV);
      p->SetTargetDevice(cv::dnn::DNN_TARGET_CPU);
    }
  }

  result->setExplictInputSize({input_w, input_h});
  result->SetInputParamsNorm(scale, {mc1, mc2, mc3}, swap_ch);
  return result;
}

std::shared_ptr<Classifier> ClassifierFromJson(const nlohmann::json &conf) {
  auto inf = PredictorFromJson(conf);

  const int ncls = conf.value("ncls", 10);
  const int nrecents = conf.value("nrecents", 1);
  const vector<string> labels = StringListFromJson(conf["labels"]);

  Classifier *ptr = new Classifier();
  ptr->Init(inf, ncls, nrecents, labels);
  return std::shared_ptr<Classifier>{ptr};
}

std::vector<std::string> StringListFromJson(const nlohmann::json &json_array) {
  const int n = json_array.size();
  std::vector<std::string> result(n);
  result.insert(result.begin(), json_array.begin(), json_array.end());
  return result;
}

std::shared_ptr<DetectionDecoder> DetectionDecoderFromJson(
    const nlohmann::json &conf) {
  const string decoder_type = conf.value("decoder", "");
  const float obj_th = conf.value("obj_th", 0.5f);
  const float nms_th = conf.value("nms_th", 0.3f);
  const int input_w = conf.value("input_w", -1);
  const int input_h = conf.value("input_h", -1);
  const int ncls = conf.value("ncls", -1);

  bool abort = false;
  if (input_w == -1) {
    cerr << "Missing `input_w` parameter" << endl;
    abort = true;
  }
  if (input_h == -1) {
    cerr << "Missing `input_h` parameter" << endl;
    abort = true;
  }
  if (ncls == -1) {
    cerr << "Missing `ncls` parameter" << endl;
    abort = true;
  }
  if (abort) return {};

  shared_ptr<DetectionDecoder> result{};
  if (decoder_type == "yolox") {
    return YoloXDecoder::Create({input_w, input_h}, ncls, obj_th, nms_th);
  } else if (decoder_type == "ssd") {
    return SSDDecoder::Create({input_w, input_h}, obj_th, nms_th);
  } else {
    cout << "Not valid detection decoder type " << endl;
    return {};
  }
}

void ObjectTrackerFromJson(const nlohmann::json &conf, ObjectTracker &tracker) {
  const int frames_to_count = conf.value("frames_to_count", 0);
  const int frames_to_discart = conf.value("frames_to_discart", 0);
  const float iou = conf.value("tracker_iou", 0.0);
  tracker.Init(frames_to_count, frames_to_discart, iou);
}

void SoftMax(cv::Mat &out) {
  float dem = 0.0;
  for (int i = 0; i < out.total(); ++i) {
    const float val = std::exp(out.ptr<float>(0)[i]);
    dem += val;
    out.ptr<float>(0)[i] = val;
  }
  for (int i = 0; i < out.total(); ++i) out.ptr<float>(0)[i] /= dem;
}

std::pair<int, float> ArgMax(const cv::Mat &out) {
  float max_val = out.ptr<float>(0)[0];
  int id = 0;
  for (int i = 1; i < out.total(); ++i) {
    if (out.ptr<float>(0)[i] > max_val) {
      id = i;
      max_val = out.ptr<float>(0)[i];
    }
  }
  return {id, max_val};
}

void ScaleRect(cv::Rect &r, float scale) {
  const int w = static_cast<int>(r.width * scale + 0.5);
  const int h = static_cast<int>(r.height * scale + 0.5);
  r.x = r.x + (r.width - w) / 2;
  r.y = r.y + (r.height - h) / 2;
  r.width = w;
  r.height = h;
}

cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame) {
  return rect & cv::Rect{0, 0, frame.cols, frame.rows};
}

cv::Mat Get4x3Roi(cv::Mat frame) {
  const int h = frame.rows;
  const int w = static_cast<int>((h / 3) * 4);

  int offset = (frame.cols - w) / 2;
  if (offset < 0) {
    return frame;
  }
  return frame({offset, 0, w, h});
}

vector<string> Split(const string &s, char delimiter) {
  vector<string> result;
  std::size_t pos, last_pos = 0;
  while ((pos = s.find(delimiter, last_pos)) != string::npos) {
    result.push_back(s.substr(last_pos, pos - last_pos));
    last_pos = pos + 1;
  }
  pos = s.size();
  result.push_back(s.substr(last_pos, pos - last_pos));
  return result;
};

float LinearMap(float a, float b, float x) {
    return (x-a)/(b-a);
}

void SmootherAve::Init(int nave, float initial_val){
    nave_  = nave;
    curr_val_ = val0_  = initial_val;
}

void SmootherAve::Reset(){
    Init(nave_, val0_);
}

float SmootherAve::Update(float x)
{
    curr_val_ = (curr_val_*(nave_-1) + x)/nave_;
    return curr_val_;
}
