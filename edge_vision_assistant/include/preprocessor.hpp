#pragma once

#include "types.hpp"
#include <vector>
#include <array>
#include <string>
#include <filesystem>

namespace edge_vision {

struct PreprocessConfig {
    int target_w = 224;
    int target_h = 224;
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std_dev = {0.229f, 0.224f, 0.225f};
};

/**
 * Preprocessor
 * ------------
 * Ingests images from disk, applies high-quality bilinear resizing,
 * ImageNet normalization, and HWC to planar NCHW layout conversion.
 */
class Preprocessor {
public:
    explicit Preprocessor(PreprocessConfig config = PreprocessConfig{});

    // Ingest image from disk, resize and normalize into planar float tensor
    bool process(
        const fs::path& image_path,
        std::vector<float>& output_tensor,
        std::string& error_msg,
        bool verbose = false
    ) const;

    const PreprocessConfig& get_config() const { return config_; }
    std::array<int64_t, 4> get_input_shape() const {
        return {1, 3, config_.target_h, config_.target_w};
    }

private:
    PreprocessConfig config_;
};

} // namespace edge_vision
