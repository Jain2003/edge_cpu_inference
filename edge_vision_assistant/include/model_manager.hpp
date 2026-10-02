#pragma once

#include "types.hpp"
#include "inference_engine.hpp"
#include "postprocessor.hpp"
#include <string>
#include <filesystem>

namespace edge_vision {

/**
 * ModelManager
 * ------------
 * Manages model paths, verification, auto-triggering quantization,
 * model profile evaluation, and side-by-side scorecard generation.
 */
class ModelManager {
public:
    explicit ModelManager(
        fs::path fp32_path = "models/resnet18-v1-7.onnx",
        fs::path int8_path = "models/resnet18-v1-7-int8.onnx",
        fs::path labels_path = "models/imagenet_classes.txt"
    );

    bool validate_fp32_model(std::string& error_msg) const;
    bool ensure_int8_model(std::string& error_msg) const;

    const fs::path& get_fp32_path() const { return fp32_path_; }
    const fs::path& get_int8_path() const { return int8_path_; }
    const fs::path& get_labels_path() const { return labels_path_; }

    static double get_file_size_mb(const fs::path& path);

    // Profile single model evaluation (warmup + measured run + postprocess + RSS memory)
    static ModelRunProfile evaluate_profile(
        InferenceEngine& engine,
        const Postprocessor& postprocessor,
        const std::vector<float>& input_tensor,
        const std::array<int64_t, 4>& input_shape,
        const std::string& tag,
        const std::string& precision,
        bool verbose = false
    );

    // Print side-by-side comparison table (Scorecard)
    static void print_comparison_table(
        const ModelRunProfile& fp32,
        const ModelRunProfile& int8,
        double preprocess_ms
    );

private:
    fs::path fp32_path_;
    fs::path int8_path_;
    fs::path labels_path_;
};

} // namespace edge_vision
