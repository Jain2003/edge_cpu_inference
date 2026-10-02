#include "preprocessor.hpp"

// STB Image: Lightweight, self-contained single-header image loading & resizing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <iostream>
#include <iomanip>

namespace edge_vision {

// Custom RAII deleter for STB image memory
struct StbImageDeleter {
    void operator()(unsigned char* ptr) const noexcept {
        if (ptr != nullptr) {
            stbi_image_free(ptr);
        }
    }
};
using UniqueStbImage = std::unique_ptr<unsigned char[], StbImageDeleter>;

Preprocessor::Preprocessor(PreprocessConfig config)
    : config_(config) {}

bool Preprocessor::process(
    const fs::path& image_path,
    std::vector<float>& output_tensor,
    std::string& error_msg,
    bool verbose
) const {
    int orig_w = 0;
    int orig_h = 0;
    int channels = 0;

    // 1. Ingest image snapshot from disk (force 3 channels RGB)
    UniqueStbImage raw_rgb(
        stbi_load(image_path.string().c_str(), &orig_w, &orig_h, &channels, 3),
        StbImageDeleter{}
    );

    if (!raw_rgb) {
        error_msg = "Failed to load image from disk: " + image_path.string() + " (" + stbi_failure_reason() + ")";
        return false;
    }

    if (verbose) {
        std::cout << "\n=================================================================================\n";
        std::cout << "🔍 [STEP 1] Ingested raw image into memory (raw_rgb)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  File Path          : " << image_path.string() << "\n";
        std::cout << "  Heap Pointer Address: " << static_cast<void*>(raw_rgb.get()) << "\n";
        std::cout << "  Original Dimensions: " << orig_w << " x " << orig_h << " (Decoded as 3 Channels: RGB)\n";
        std::cout << "  Total Pixels       : " << (orig_w * orig_h) << " pixels\n";
        std::cout << "  Total Buffer Size  : " << (orig_w * orig_h * 3) << " bytes (~"
                  << std::fixed << std::setprecision(2) << ((orig_w * orig_h * 3) / (1024.0 * 1024.0)) << " MB)\n";
        std::cout << "  Memory Layout      : Interleaved HWC [R0, G0, B0,  R1, G1, B1, ...]\n";
        std::cout << "  First 3 Pixels in Memory:\n";
        for (int i = 0; i < 3 && i < orig_w; ++i) {
            std::cout << "    Pixel (" << i << ", 0): Red=" << static_cast<int>(raw_rgb[i * 3 + 0])
                      << ", Green=" << static_cast<int>(raw_rgb[i * 3 + 1])
                      << ", Blue=" << static_cast<int>(raw_rgb[i * 3 + 2]) << "\n";
        }
    }

    // 2. Resize to model input dimensions (target_w x target_h) using high-quality bilinear interpolation
    const int target_w = config_.target_w;
    const int target_h = config_.target_h;
    std::vector<unsigned char> resized_rgb(static_cast<size_t>(target_w * target_h * 3));
    stbir_resize_uint8_linear(
        raw_rgb.get(), orig_w, orig_h, 0,
        resized_rgb.data(), target_w, target_h, 0,
        STBIR_RGB
    );

    if (verbose) {
        std::cout << "\n=================================================================================\n";
        std::cout << "🔍 [STEP 2] Resized image to model dimensions (resized_rgb)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  Target Dimensions  : " << target_w << " x " << target_h << " (3 Channels RGB)\n";
        std::cout << "  Buffer Size        : " << resized_rgb.size() << " bytes (224 x 224 x 3 = 150,528 bytes)\n";
        std::cout << "  Vector Pointer     : " << static_cast<void*>(resized_rgb.data()) << "\n";
        std::cout << "  Memory Layout      : Interleaved HWC [R0, G0, B0,  R1, G1, B1, ...]\n";
        std::cout << "  Key Pixel Samples in resized_rgb:\n";
        std::cout << "    * Top-Left  (0, 0)      (index 0)     : Red=" << static_cast<int>(resized_rgb[0])
                  << ", Green=" << static_cast<int>(resized_rgb[1])
                  << ", Blue=" << static_cast<int>(resized_rgb[2]) << "\n";

        size_t center_idx = static_cast<size_t>((112 * target_w + 112) * 3);
        std::cout << "    * Center    (112, 112)  (index " << center_idx << ") : Red=" << static_cast<int>(resized_rgb[center_idx + 0])
                  << ", Green=" << static_cast<int>(resized_rgb[center_idx + 1])
                  << ", Blue=" << static_cast<int>(resized_rgb[center_idx + 2]) << "\n";

        size_t br_idx = static_cast<size_t>((223 * target_w + 223) * 3);
        std::cout << "    * Bottom-Rt (223, 223)  (index " << br_idx << "): Red=" << static_cast<int>(resized_rgb[br_idx + 0])
                  << ", Green=" << static_cast<int>(resized_rgb[br_idx + 1])
                  << ", Blue=" << static_cast<int>(resized_rgb[br_idx + 2]) << "\n";
    }

    raw_rgb.reset();

    // 3. Normalization and Planar HWC -> CHW layout conversion
    const auto& mean = config_.mean;
    const auto& std_dev = config_.std_dev;

    const size_t channel_stride = static_cast<size_t>(target_w * target_h);
    output_tensor.resize(3 * channel_stride);

    if (verbose) {
        std::cout << "\n=================================================================================\n";
        std::cout << "🔍 [STEP 3] Normalization Math & Planar Memory Mapping (HWC -> NCHW)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  ImageNet Constants :\n";
        std::cout << "    * Red Channel   : Mean = " << mean[0] << ", StdDev = " << std_dev[0] << "\n";
        std::cout << "    * Green Channel : Mean = " << mean[1] << ", StdDev = " << std_dev[1] << "\n";
        std::cout << "    * Blue Channel  : Mean = " << mean[2] << ", StdDev = " << std_dev[2] << "\n";
        std::cout << "  Channel Stride     : " << channel_stride << " floats per color plane (224 x 224)\n";
        std::cout << "  Output Tensor Planar Partitions:\n";
        std::cout << "    * Channel 0 (All Red)   : Indices [0       .. " << (channel_stride - 1) << "]\n";
        std::cout << "    * Channel 1 (All Green) : Indices [" << channel_stride << "   .. " << (2 * channel_stride - 1) << "]\n";
        std::cout << "    * Channel 2 (All Blue)  : Indices [" << (2 * channel_stride) << "  .. " << (3 * channel_stride - 1) << "]\n";
        std::cout << "\n  --- Step-by-Step Arithmetic for Selected Sample Pixels ---\n";
    }

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            const size_t src_idx = static_cast<size_t>((y * target_w + x) * 3);
            const size_t dst_spatial_idx = static_cast<size_t>(y * target_w + x);

            const float r = static_cast<float>(resized_rgb[src_idx + 0]) / 255.0f;
            const float g = static_cast<float>(resized_rgb[src_idx + 1]) / 255.0f;
            const float b = static_cast<float>(resized_rgb[src_idx + 2]) / 255.0f;

            const float norm_r = (r - mean[0]) / std_dev[0];
            const float norm_g = (g - mean[1]) / std_dev[1];
            const float norm_b = (b - mean[2]) / std_dev[2];

            const size_t out_r_idx = 0 * channel_stride + dst_spatial_idx;
            const size_t out_g_idx = 1 * channel_stride + dst_spatial_idx;
            const size_t out_b_idx = 2 * channel_stride + dst_spatial_idx;

            output_tensor[out_r_idx] = norm_r;
            output_tensor[out_g_idx] = norm_g;
            output_tensor[out_b_idx] = norm_b;

            if (verbose && ((x == 0 && y == 0) || (x == 112 && y == 112))) {
                std::cout << "  Calculating Pixel (" << x << ", " << y << "):\n";
                std::cout << "    Source in resized_rgb at src_idx=" << src_idx << ": "
                          << "RedByte=" << static_cast<int>(resized_rgb[src_idx + 0])
                          << ", GreenByte=" << static_cast<int>(resized_rgb[src_idx + 1])
                          << ", BlueByte=" << static_cast<int>(resized_rgb[src_idx + 2]) << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "    [Red]   : (" << r << " - " << mean[0] << ") / " << std_dev[0] << " = "
                          << (norm_r >= 0 ? "+" : "") << norm_r << "  --> stored at output_tensor[" << out_r_idx << "]\n";
                std::cout << "    [Green] : (" << g << " - " << mean[1] << ") / " << std_dev[1] << " = "
                          << (norm_g >= 0 ? "+" : "") << norm_g << "  --> stored at output_tensor[" << out_g_idx << "]\n";
                std::cout << "    [Blue]  : (" << b << " - " << mean[2] << ") / " << std_dev[2] << " = "
                          << (norm_g >= 0 ? "+" : "") << norm_b << "  --> stored at output_tensor[" << out_b_idx << "]\n\n";
            }
        }
    }

    if (verbose) {
        std::cout << "=================================================================================\n";
        std::cout << "🔍 [STEP 4] Preprocessed Image Tensor in RAM (output_tensor)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  Total Float Elements: " << output_tensor.size() << " floats (~"
                  << (output_tensor.size() * sizeof(float) / 1024.0) << " KB)\n";
        std::cout << "  Tensor Dimensions   : [1, 3, " << target_h << ", " << target_w << "] (Batch=1, Channels=3, H=" << target_h << ", W=" << target_w << ")\n";
        std::cout << "  Buffer Pointer      : " << static_cast<void*>(output_tensor.data()) << "\n";
        std::cout << "  Memory Structure    : 3 Planar Channels of " << target_w << " x " << target_h << " = " << channel_stride << " floats each\n";

        const char* ch_names[3] = {"Channel 0 (Red)", "Channel 1 (Green)", "Channel 2 (Blue)"};
        for (int c = 0; c < 3; ++c) {
            size_t start = c * channel_stride;
            std::cout << "\n  --- " << ch_names[c] << " [Indices " << start << " .. " << (start + channel_stride - 1) << "] ---\n";
            std::cout << "  Top-Left 8 x 8 Spatial Matrix Slice (Rows y=0..7, Cols x=0..7):\n";
            for (int y = 0; y < 8; ++y) {
                std::cout << "    [y=" << std::setw(2) << y << "] ";
                for (int x = 0; x < 8; ++x) {
                    float val = output_tensor[start + y * target_w + x];
                    std::cout << (val >= 0 ? " +" : " ") << std::fixed << std::setprecision(3) << val << " ";
                }
                std::cout << "\n";
            }
            std::cout << "    ... (216 more rows of 224 pixels in this plane) ...\n";
        }
        std::cout << "=================================================================================\n\n";
    }

    return true;
}

} // namespace edge_vision
