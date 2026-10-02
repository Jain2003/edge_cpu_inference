#include "inference_engine.hpp"
#include <chrono>
#include <stdexcept>

namespace edge_vision {

InferenceEngine::InferenceEngine(const fs::path& model_path, int intra_op_threads)
    : model_path_(model_path),
      env_(ORT_LOGGING_LEVEL_WARNING, "EdgeVisionAssistant") {
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path.string());
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(intra_op_threads);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
}

void InferenceEngine::warmup(
    const std::vector<float>& input_tensor_values,
    const std::array<int64_t, 4>& input_shape
) {
    const std::array<const char*, 1> input_names = { input_name_.c_str() };
    const std::array<const char*, 1> output_names = { output_name_.c_str() };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_values.data()),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
}

std::vector<Ort::Value> InferenceEngine::run(
    const std::vector<float>& input_tensor_values,
    const std::array<int64_t, 4>& input_shape,
    double& out_inference_ms
) {
    const std::array<const char*, 1> input_names = { input_name_.c_str() };
    const std::array<const char*, 1> output_names = { output_name_.c_str() };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_values.data()),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const auto t_start = Clock::now();
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );
    const auto t_end = Clock::now();

    out_inference_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return output_tensors;
}

double InferenceEngine::get_model_size_mb() const {
    if (fs::exists(model_path_)) {
        return static_cast<double>(fs::file_size(model_path_)) / (1024.0 * 1024.0);
    }
    return 0.0;
}

} // namespace edge_vision
