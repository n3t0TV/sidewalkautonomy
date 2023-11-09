#include "tensor_rt_predictor.h"

#include <fstream>
#include <iostream>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

#define LOG_TRT "[TRT]    "
#define CACHE_MODEL "/home/jetson/TRT_CACHE/"

void TensorRTPredictorLogger::log(Severity severity, const char *msg) noexcept {
  static const std::string levels[] = {"INTERNAL_ERROR", "ERROR", "WARNING",
                                       "INFO", "DEBUG_INFO"};
  auto iseverity = static_cast<int32_t>(severity);
  if (iseverity <= severity_level_) {
    std::cerr << "[" << levels[iseverity] << "]: " << msg << std::endl;
  }
}

std::unique_ptr<Predictor> TensorRTPredictor::Create(
    const std::string &model_path, const std::string &framework,
    nvinfer1::BuilderFlag precision, nvinfer1::DeviceType device,
    int async_count) {
  auto ptr = new TensorRTPredictor();
  ptr->Init(model_path, framework, precision, device, async_count);
  return std::unique_ptr<TensorRTPredictor>(ptr);
}

std::unique_ptr<Predictor> TensorRTPredictor::Create(
    const char *model_buffer, const size_t model_size,
    const std::string &framework, nvinfer1::BuilderFlag precision,
    nvinfer1::DeviceType device, int async_count) {
  auto ptr = new TensorRTPredictor();
  ptr->Init(model_buffer, model_size, framework, precision, device,
            async_count);
  return std::unique_ptr<TensorRTPredictor>(ptr);
}

bool TensorRTPredictor::Init(const std::string &model_path,
                             const std::string &framework,
                             nvinfer1::BuilderFlag precision,
                             nvinfer1::DeviceType device, int async_count) {
  std::string model_buffer;
  Predictor::GetStringData(model_path, &model_buffer);
  return Init(model_buffer.data(), model_buffer.size(), framework, precision,
              device, async_count);
}

bool TensorRTPredictor::Init(const char *model_buffer, const size_t model_size,
                             const std::string &framework,
                             nvinfer1::BuilderFlag precision,
                             nvinfer1::DeviceType device, int async_count) {
  static bool loadedPlugins = false;
  if (!loadedPlugins) {
    std::cout << LOG_TRT "loading NVIDIA plugins..." << std::endl;
    loadedPlugins = initLibNvInferPlugins(&logger_, "");
    if (!loadedPlugins) {
      std::cout << LOG_TRT "failed to load NVIDIA plugins." << std::endl;
    } else {
      std::cout << LOG_TRT "completed loading NVIDIA plugins." << std::endl;
    }
  }
  // cudaSetDevice(device);

  if (framework == "TRT") {  // recomended
    cuda_engine_ = BuildCudaEngine(model_buffer, model_size);
  } else if (framework == "ONNX") {
    const string m_name =
        GenModelName(model_buffer, model_size, static_cast<int>(precision),
                     static_cast<int>(device));
    string data;
    GetStringData(CACHE_MODEL + m_name, &data);
    if (data.empty()) {
      cuda_engine_ = BuildCudaEngineFromONNX(model_buffer, model_size,
                                             precision, device, true);
    } else {
      cuda_engine_ = BuildCudaEngine(data.data(), data.size());
    }
  } else {
    std::cerr << " Framework: ``" << framework << "`` is not supported"
              << std::endl;
    return false;
  }

  const int batch_size = cuda_engine_->getMaxBatchSize();
  if (batch_size != 1) {
    std::cerr << "BatchSize must be equal to 1" << std::endl;
    return false;
  }
  const auto nb = cuda_engine_->getNbBindings();
  int input_count = 0;
  // infer the input dimension from the model graph
  for (int i = 0; i < nb; ++i) {
    const auto dims = cuda_engine_->getBindingDimensions(i);
    buffer_dims_.push_back(vector<int32_t>{dims.d, dims.d + dims.nbDims});
    assert(cuda_engine_->getBindingDataType(i) == nvinfer1::DataType::kFLOAT);
    size_t buffer_size = sizeof(float);
    for (int k = 0; k < dims.nbDims; ++k) buffer_size *= dims.d[k];
    buffer_sizes_.push_back(buffer_size);

    if (cuda_engine_->bindingIsInput(i)) {
      input_ch_ = dims.d[1];
      input_size_.height = dims.d[2];
      input_size_.width = dims.d[3];
      ++input_count;
      buffer_input_index_ = i;
    }
  }

  // only single input models allowed
  if (input_count != 1) {
    std::cerr << "Model must have only one input" << std::endl;
    return false;
  }

  if (async_count <= 1) {
    inferences_.resize(1);
    inferences_[0] = new Inference(this);
  } else {
    inferences_.resize(async_count);
    for (int i = 0; i < inferences_.size(); ++i) {
      inferences_[i] = new Inference(this);
    }
  }
  return true;
}

TensorRTPredictor::~TensorRTPredictor() {
  for (Inference *p : inferences_) {
    delete p;
  }
}

void TensorRTPredictor::Predict(const cv::Mat &img,
                                std::vector<cv::Mat> &outputs,
                                const std::vector<std::string> &output_names) {
  LaunchInference(img, inferences_[0]);
  GetInferenceResult(inferences_[0], outputs, output_names);
}

bool TensorRTPredictor::EnqueueToInput(const cv::Mat &img, void *user_param) {
  if (inferences_.size() == 0) {
    std::cerr << "Inference queue size should be greater than 0" << std::endl;
    exit(0);
  }
  if (running_inferences_count_.load() < inferences_.size()) {
    LaunchInference(img, inferences_[index_in_]);
    inferences_[index_in_]->SetUserParam(user_param);
    index_in_ = (index_in_ + 1) % inferences_.size();
    running_inferences_count_.fetch_add(1);
    return true;
  }
  return false;
}

bool TensorRTPredictor::ReadFromOutput(vector<cv::Mat> &outputs,
                                       vector<string> onames,
                                       void **user_param) {
  if (running_inferences_count_.load() > 0) {
    GetInferenceResult(inferences_[index_out_], outputs, onames);
    if (user_param) {
      *user_param = inferences_[index_out_]->GetUserParam();
    }
    index_out_ = (index_out_ + 1) % inferences_.size();
    running_inferences_count_.fetch_sub(1);
    return true;
  }
  return false;
}

bool TensorRTPredictor::OkToRead() {
  return (running_inferences_count_.load() > 0) &&
         (inferences_[index_out_]->IsDataReady());
}

bool TensorRTPredictor::OkToWrite() {
  return running_inferences_count_.load() < inferences_.size();
}

void TensorRTPredictor::StringDataToFile(const std::string &filePath,
                                         const char *data, size_t n) {
  std::ofstream ofs(filePath, std::ios::out | std::ios::binary);
  ofs.write(data, n);
  ofs.close();
}

bool TensorRTPredictor::Inference::CudaAllocMapped(void **cpuPtr, void **gpuPtr,
                                                   size_t size) {
  if (!cpuPtr || !gpuPtr || size == 0) return false;

  // CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess) {
    std::cerr << "Error allocating mapped memory" << std::endl;
    return false;
  }
  if (cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) != cudaSuccess) {
    std::cerr << "Error getting mapped pointers" << std::endl;
    return false;
  }
  memset(*cpuPtr, 0, size);
  return true;
}

void TensorRTPredictor::LaunchInference(const cv::Mat &frame, Inference *inf) {
  void *buff = inf->GetBuffer(buffer_input_index_);
  if (normalization_fnt_) {
    normalization_fnt_(frame, (float *)buff);
  } else {
    const cv::Mat blob = cv::dnn::blobFromImage(frame, scale_, input_size_,
                                                mean_, swap_ch_, false, CV_32F);
    size_t n = buffer_sizes_[buffer_input_index_];
    std::memcpy(buff, blob.ptr<float>(0), n);
  }
  inf->Launch();
}

void TensorRTPredictor::GetInferenceResult(
    Inference *inf, std::vector<cv::Mat> &result,
    const std::vector<std::string> &output_names) {
  inf->WaitForResults();
  result.clear();

  if (output_names.empty()) {
    result.resize(buffer_sizes_.size() - 1);
    int j = 0;
    for (int i = 0; i < buffer_sizes_.size(); ++i) {
      if (cuda_engine_->bindingIsInput(i) == false) {
        cv::Mat(buffer_dims_[i], CV_32F, inf->GetBuffer(i)).copyTo(result[j]);
        ++j;
      }
    }
  } else {
    result.resize(output_names.size());
    int j = 0;
    for (const auto &name : output_names) {
      const int i = cuda_engine_->getBindingIndex(name.c_str());
      cv::Mat(buffer_dims_[i], CV_32F, inf->GetBuffer(i)).copyTo(result[j]);
      ++j;
    }
  }
}

#if NV_TENSORRT_MAJOR == 7
nvinfer1::ICudaEngine *TensorRTPredictor::BuildCudaEngine(
    const char *model_buffer, const size_t n) {
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger_);
  // runtime->setDLACore(0);

  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(model_buffer, n);
  runtime->destroy();
  return engine;
}

nvinfer1::ICudaEngine *TensorRTPredictor::BuildCudaEngineFromONNX(
    const char *model_buffer, const size_t n, nvinfer1::BuilderFlag precision,
    nvinfer1::DeviceType device, bool allowGPUFallback) {
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger_);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(
      1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  if (!network) {
    std::cerr << LOG_TRT "IBuilder::createNetworkV2(EXPLICIT_BATCH) failed"
              << std::endl;
    return {};
  }

  nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger_);
  if (!parser) {
    std::cerr << LOG_TRT "failed to create nvonnxparser::IParser instance"
              << std::endl;
    return {};
  }

  if (!parser->parse(model_buffer, n)) {
    std::cerr << LOG_TRT "failed to parse ONNX model." << std::endl;
    return {};
  }

  nvinfer1::IBuilderConfig *builderConfig = builder->createBuilderConfig();
  if (!ConfigureBuilder(builder, builderConfig, precision, device,
                        allowGPUFallback)) {
    std::cerr << LOG_TRT "failed to configure builder" << std::endl;
    return {};
  }

  nvinfer1::ICudaEngine *eng =
      builder->buildEngineWithConfig(*network, *builderConfig);
  if (!eng) {
    std::cerr << LOG_TRT "failed to build CUDA engine" << std::endl;
    return nullptr;
  }
  nvinfer1::IHostMemory *serMem = eng->serialize();

  // cache the engine so as not to repeat the conversion process again
  const string chache_model_path =
      CACHE_MODEL + GenModelName(model_buffer, n, static_cast<int>(precision),
                                 static_cast<int>(device));
  StringDataToFile(chache_model_path, reinterpret_cast<char *>(serMem->data()),
                   serMem->size());
  serMem->destroy();
  builder->destroy();
  network->destroy();
  builderConfig->destroy();
  parser->destroy();

  return eng;
}
#else
std::unique_ptr<nvinfer1::ICudaEngine> TensorRTPredictor::BuildCudaEngine(
    const char *model_buffer, const size_t n) {
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger_);
  // runtime->setDLACore(0);

  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(model_buffer, n);
  delete runtime;
  return std::unique_ptr<nvinfer1::ICudaEngine>{engine};
}

std::unique_ptr<nvinfer1::ICudaEngine>
TensorRTPredictor::BuildCudaEngineFromONNX(const char *model_buffer,
                                           const size_t n,
                                           nvinfer1::BuilderFlag precision,
                                           nvinfer1::DeviceType device,
                                           bool allowGPUFallback) {
  std::unique_ptr<nvinfer1::IBuilder> builder{
      nvinfer1::createInferBuilder(logger_)};
  std::unique_ptr<nvinfer1::INetworkDefinition> network{
      builder->createNetworkV2(
          1U << (uint32_t)
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

  if (!network) {
    std::cerr << LOG_TRT "IBuilder::createNetworkV2(EXPLICIT_BATCH) failed"
              << std::endl;
    return {};
  }

  std::unique_ptr<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, logger_)};
  if (!parser) {
    std::cerr << LOG_TRT "failed to create nvonnxparser::IParser instance"
              << std::endl;
    return {};
  }

  const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kVERBOSE;

  if (!parser->parse(model_buffer, n)) {
    std::cerr << LOG_TRT "failed to parse ONNX model." << std::endl;
    return {};
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig{
      builder->createBuilderConfig()};
  if (!ConfigureBuilder(builder.get(), builderConfig.get(), precision, device,
                        allowGPUFallback)) {
    std::cerr << LOG_TRT "failed to configure builder" << std::endl;
    return {};
  }

  std::unique_ptr<nvinfer1::IHostMemory> serMem{
      builder->buildSerializedNetwork(*network, *builderConfig)};

  if (!serMem) {
    std::cerr << LOG_TRT "failed to build CUDA engine" << std::endl;
    return nullptr;
  }

  // cache the engine so as not to repeat the conversion process again
  const string chache_model_path =
      CACHE_MODEL + GenModelName(model_buffer, n, static_cast<int>(precision),
                                 static_cast<int>(device));
  StringDataToFile(chache_model_path, reinterpret_cast<char *>(serMem->data()),
                   serMem->size());
  return BuildCudaEngine(reinterpret_cast<const char *>(serMem->data()),
                         serMem->size());
}
#endif

string TensorRTPredictor::GenModelName(const char *mpath, size_t n,
                                       int precision, int device) {
  std::hash<string> hasher;
  auto s = std::min(size_t(1024 * 1024), n);
  size_t unique_no = hasher({mpath, s});
  char out_name[128];

  std::sprintf(out_name, "model_%lu_precision_%d_device_%d_trt_%d_%d",
               unique_no, precision, device, NV_TENSORRT_MAJOR,
               NV_TENSORRT_MINOR);
  return {out_name};
}

bool TensorRTPredictor::ConfigureBuilder(nvinfer1::IBuilder *builder,
                                         nvinfer1::IBuilderConfig *config,
                                         nvinfer1::BuilderFlag precision,
                                         nvinfer1::DeviceType device,
                                         bool allowGPUFallback) {
  if (!builder) return false;

  std::cerr << LOG_TRT "Configuring network builder" << std::endl;

  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(20 << 16);

  // set up the builder for the desired precision
  if (precision == nvinfer1::BuilderFlag::kINT8) {
    std::cerr << LOG_TRT "INT8 requested but calibrator is NULL" << std::endl;
    return false;
  }

  config->setFlag(precision);
  // set the default device type
  config->setDefaultDeviceType(device);
  if (allowGPUFallback) config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  return true;
}

TensorRTPredictor::Inference::Inference(TensorRTPredictor *parent) {
  parent_ = parent;
  auto context = parent_->cuda_engine_->createExecutionContext();
  assert(context != nullptr);

#if NV_TENSORRT_MAJOR == 7
  exec_context_ = context;
#else
  exec_context_.reset(context);
#endif

  const auto nb = parent_->cuda_engine_->getNbBindings();
  gpu_buffers_.resize(nb);
  host_buffers_.resize(nb);

  for (int i = 0; i < nb; ++i) {
    /*int ok = */ CudaAllocMapped(&(host_buffers_[i]), &(gpu_buffers_[i]),
                                  parent->buffer_sizes_[i]);
  }
  // Create stream
  cudaStreamCreate(&cuda_stream_);
}

void TensorRTPredictor::Inference::Launch() {
  exec_context_->enqueue(1, gpu_buffers_.data(), cuda_stream_, 0);
}

void TensorRTPredictor::Inference::WaitForResults() {
  cudaStreamSynchronize(cuda_stream_);
}

TensorRTPredictor::Inference::~Inference() {
  cudaStreamDestroy(cuda_stream_);
  for (size_t i = 0; i < gpu_buffers_.size(); ++i) {
    cudaFreeHost(host_buffers_[i]);
  }
#if NV_TENSORRT_MAJOR == 7
  exec_context_->destroy();
#endif
}

bool TensorRTPredictor::Inference::IsDataReady() const {
  return cudaStreamQuery(cuda_stream_) == cudaSuccess;
}
