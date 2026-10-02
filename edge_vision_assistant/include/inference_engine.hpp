#pragma once

#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <memory>

namespace edge_vision {

/**
 * InferenceEngine
 * ---------------
 * Manages the ONNX Runtime session, memory buffers, AVX2 SIMD execution,
 * warm-up runs, and single/batch inference execution on the CPU.
 */
class InferenceEngine {
public:
    explicit InferenceEngine(const fs::path& model_path, int intra_op_threads = 0);
    ~InferenceEngine() = default;

    // Disallow copy, allow move
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept = default;
    InferenceEngine& operator=(InferenceEngine&&) noexcept = default;

    // Execute warm-up inference pass to populate CPU caches & vector pipelines
    void warmup(const std::vector<float>& input_tensor_values, const std::array<int64_t, 4>& input_shape);

    // Run inference, populate out_tensors, and measure kernel execution latency in ms
    std::vector<Ort::Value> run(
        const std::vector<float>& input_tensor_values,
        const std::array<int64_t, 4>& input_shape,
        double& out_inference_ms
    );

    const fs::path& get_model_path() const { return model_path_; }
    const std::string& get_input_name() const { return input_name_; }
    const std::string& get_output_name() const { return output_name_; }
    double get_model_size_mb() const;

private:
    fs::path model_path_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

} // namespace edge_vision
