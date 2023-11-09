#ifndef TENSOR_RT_PREDICTOR_H_
#define TENSOR_RT_PREDICTOR_H_

#include <NvInferRuntime.h>

#include <atomic>
#include <memory>
#include <vector>

#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "opencv2/opencv.hpp"
#include "predictor.h"

class TensorRTPredictorLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override;

 private:
  int32_t severity_level_{1};
};

using namespace std;

class TensorRTPredictor : public Predictor {
 public:
  friend class Inference;
  class Inference {
    cudaStream_t cuda_stream_;

#if NV_TENSORRT_MAJOR == 7
    nvinfer1::IExecutionContext *exec_context_{nullptr};
#else
    std::unique_ptr<nvinfer1::IExecutionContext> exec_context_{nullptr};
#endif

    std::vector<void *> gpu_buffers_{nullptr};
    std::vector<void *> host_buffers_{};

    void *user_param_{nullptr};
    bool CudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size);
    TensorRTPredictor *parent_{};

   public:
    Inference(TensorRTPredictor *parent);
    ~Inference();
    void *GetBuffer(int index) { return host_buffers_[index]; }
    void Launch();
    void WaitForResults();
    void SetUserParam(void *user_param) { user_param_ = user_param; }
    void *GetUserParam() { return user_param_; }
    bool IsDataReady() const;
  };

 public:
  static std::unique_ptr<Predictor> Create(
      const std::string &model_path, const std::string &framework = "TRT",
      nvinfer1::BuilderFlag precision = nvinfer1::BuilderFlag::kTF32,
      nvinfer1::DeviceType device = nvinfer1::DeviceType::kGPU,
      int async_count = 0);

  static std::unique_ptr<Predictor> Create(
      const char *model_buffer, const size_t model_size,
      const std::string &framework = "TRT",
      nvinfer1::BuilderFlag precision = nvinfer1::BuilderFlag::kTF32,
      nvinfer1::DeviceType device = nvinfer1::DeviceType::kGPU,
      int async_count = 0);

  void Predict(const cv::Mat &img, std::vector<cv::Mat> &outputs,
               const std::vector<std::string> &output_names = {}) override;
  bool EnqueueToInput(const cv::Mat &img, void *user_param = nullptr) override;
  bool ReadFromOutput(std::vector<cv::Mat> &outputs,
                      std::vector<std::string> onames = {},
                      void **user_param = nullptr) override;

  bool OkToRead() override;
  bool OkToWrite() override;
  int ActiveInferences() const override {
    return running_inferences_count_.load();
  }

  ~TensorRTPredictor();

 private:
  TensorRTPredictor() = default;
  bool Init(const std::string &model_path,
            const std::string &framework = "ENGINE",
            nvinfer1::BuilderFlag precision = nvinfer1::BuilderFlag::kTF32,
            nvinfer1::DeviceType device = nvinfer1::DeviceType::kGPU,
            int async_count = 0);

  bool Init(const char *model_buffer, const size_t model_size,
            const std::string &framework = "ENGINE",
            nvinfer1::BuilderFlag precision = nvinfer1::BuilderFlag::kTF32,
            nvinfer1::DeviceType device = nvinfer1::DeviceType::kGPU,
            int async_count = 0);

 private:
  // util functions
  string GenModelName(const char *mpath, size_t n, int precision, int device);
  void StringDataToFile(const std::string &filePath, const char *data,
                        size_t n);
  void LaunchInference(const cv::Mat &frame, Inference *inference);
  bool ConfigureBuilder(nvinfer1::IBuilder *builder,
                        nvinfer1::IBuilderConfig *config,
                        nvinfer1::BuilderFlag precision,
                        nvinfer1::DeviceType device, bool allowGPUFallback);
  void GetInferenceResult(Inference *inference, std::vector<cv::Mat> &result,
                          const std::vector<std::string> &output_names = {});

#if NV_TENSORRT_MAJOR == 7
  nvinfer1::ICudaEngine *BuildCudaEngine(const char *model_buffer,
                                         const size_t n);
  nvinfer1::ICudaEngine *BuildCudaEngineFromONNX(
      const char *model_buffer, const size_t n, nvinfer1::BuilderFlag precision,
      nvinfer1::DeviceType device, bool allowGPUFallback);
#else
  std::unique_ptr<nvinfer1::ICudaEngine> BuildCudaEngine(
      const char *model_buffer, const size_t n);

  std::unique_ptr<nvinfer1::ICudaEngine> BuildCudaEngineFromONNX(
      const char *model_buffer, const size_t n, nvinfer1::BuilderFlag precision,
      nvinfer1::DeviceType device, bool allowGPUFallback);
#endif

  TensorRTPredictorLogger logger_;
  std::vector<Inference *> inferences_{};

#if NV_TENSORRT_MAJOR == 7
  nvinfer1::ICudaEngine *cuda_engine_{nullptr};
#else
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine_{nullptr};
#endif

  std::vector<size_t> buffer_sizes_{};
  std::vector<std::vector<int32_t>> buffer_dims_{};
  std::atomic_size_t running_inferences_count_{0};
  size_t index_in_{0};
  size_t index_out_{0};
  int buffer_input_index_{-1};
};

#endif
