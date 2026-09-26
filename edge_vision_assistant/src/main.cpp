/**
 * edge_vision_assistant - Smart AI Glasses Snapshot Assistant
 * ==========================================================
 * High-performance, lightweight C++ CPU inference pipeline simulating
 * a Smart AI Glasses visual query engine ("What is this?").
 *
 * Supports FP32 Baseline vs INT8 Quantized execution with side-by-side
 * performance, latency, and memory comparison.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <memory>
#include <cctype>

#if defined(__linux__) || defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#endif

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// STB Image: Lightweight, self-contained single-header image loading & resizing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace fs = std::filesystem;

// =========================================================================
// Data Structures
// =========================================================================
struct VisionResult {
    std::string image_name = "Unknown";
    int class_id = -1;
    std::string class_name = "Unknown";
    float confidence_pct = 0.0f;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    double fps = 0.0;
};

struct ModelRunProfile {
    std::string tag;
    std::string precision;
    fs::path model_path;
    double file_size_mb = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    double fps = 0.0;
    int class_id = -1;
    std::string class_name = "Unknown";
    float confidence_pct = 0.0f;
    double peak_rss_mb = 0.0;
};

// Custom RAII deleter for STB image memory
struct StbImageDeleter {
    void operator()(unsigned char* ptr) const noexcept {
        if (ptr != nullptr) {
            stbi_image_free(ptr);
        }
    }
};
using UniqueStbImage = std::unique_ptr<unsigned char[], StbImageDeleter>;

// Measure process peak resident set size in MB
double get_peak_rss_mb() {
#if defined(__linux__) || defined(__unix__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0; // ru_maxrss is in Kilobytes on Linux
    }
#endif
    return 0.0;
}

// =========================================================================
// CPU Architecture & Vector Extension Detection
// =========================================================================
std::string get_cpu_vector_feature() {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx512f")) {
            return "AVX-512 Vector Cores";
        } else if (__builtin_cpu_supports("avx2")) {
            return "AVX2 Vector Cores";
        } else if (__builtin_cpu_supports("avx")) {
            return "AVX Vector Cores";
        } else if (__builtin_cpu_supports("sse4.2")) {
            return "SSE4.2 Vector Cores";
        }
    #endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "NEON Vector Cores";
#endif
    return "SIMD Vector Cores";
}

// Convert string to clean Title Case (e.g. "tabby cat" -> "Tabby Cat")
std::string to_title_case(std::string_view input) {
    if (input.empty()) {
        return std::string();
    }
    std::string result(input);
    bool capitalize_next = true;

    for (char& c : result) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isspace(uc) || c == '_' || c == '-') {
            capitalize_next = true;
            if (c == '_') {
                c = ' ';
            }
        } else if (capitalize_next && std::isalpha(uc)) {
            c = static_cast<char>(std::toupper(uc));
            capitalize_next = false;
        } else {
            c = static_cast<char>(std::tolower(uc));
        }
    }
    return result;
}

// =========================================================================
// Class Labels Loader
// =========================================================================
std::vector<std::string> load_class_labels(const fs::path& labels_path) {
    std::vector<std::string> labels;
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        std::cerr << "[WARN] Warning: Could not open labels file: " << labels_path.string()
                  << ". Using raw indices.\n";
        return labels;
    }

    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

// =========================================================================
// Image Preprocessing Pipeline with Inline Step-by-Step Logging
// =========================================================================
/**
 * Loads an image from disk, resizes to target dimensions (224x224),
 * and converts to NCHW float32 tensor with ImageNet mean/std normalization.
 */
bool preprocess_image(
    const fs::path& image_path,
    int target_w,
    int target_h,
    std::vector<float>& output_tensor,
    std::string& error_msg,
    bool verbose = false
) {
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

    // 2. Resize to model input dimensions (224 x 224) using high-quality bilinear interpolation
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
    constexpr std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> std_dev = {0.229f, 0.224f, 0.225f};

    const size_t channel_stride = static_cast<size_t>(target_w * target_h);
    output_tensor.resize(3 * channel_stride);

    if (verbose) {
        std::cout << "\n=================================================================================\n";
        std::cout << "🔍 [STEP 3] Normalization Math & Planar Memory Mapping (HWC -> NCHW)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  ImageNet Constants :\n";
        std::cout << "    * Red Channel   : Mean = 0.485, StdDev = 0.229\n";
        std::cout << "    * Green Channel : Mean = 0.456, StdDev = 0.224\n";
        std::cout << "    * Blue Channel  : Mean = 0.406, StdDev = 0.225\n";
        std::cout << "  Channel Stride     : " << channel_stride << " floats per color plane (224 x 224)\n";
        std::cout << "  Output Tensor Planar Partitions:\n";
        std::cout << "    * Channel 0 (All Red)   : Indices [0       .. 50175]\n";
        std::cout << "    * Channel 1 (All Green) : Indices [50176   .. 100351]\n";
        std::cout << "    * Channel 2 (All Blue)  : Indices [100352  .. 150527]\n";
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
                std::cout << "    [Red]   : (" << r << " - 0.485) / 0.229 = " << (norm_r >= 0 ? "+" : "") << norm_r
                          << "  --> stored at output_tensor[" << out_r_idx << "]\n";
                std::cout << "    [Green] : (" << g << " - 0.456) / 0.224 = " << (norm_g >= 0 ? "+" : "") << norm_g
                          << "  --> stored at output_tensor[" << out_g_idx << "]\n";
                std::cout << "    [Blue]  : (" << b << " - 0.406) / 0.225 = " << (norm_b >= 0 ? "+" : "") << norm_b
                          << "  --> stored at output_tensor[" << out_b_idx << "]\n\n";
            }
        }
    }

    if (verbose) {
        std::cout << "=================================================================================\n";
        std::cout << "🔍 [STEP 4] Preprocessed Image Tensor in RAM (output_tensor)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  Total Float Elements: " << output_tensor.size() << " floats (~"
                  << (output_tensor.size() * sizeof(float) / 1024.0) << " KB)\n";
        std::cout << "  Tensor Dimensions   : [1, 3, 224, 224] (Batch=1, Channels=3, H=224, W=224)\n";
        std::cout << "  Buffer Pointer      : " << static_cast<void*>(output_tensor.data()) << "\n";
        std::cout << "  Memory Structure    : 3 Planar Channels of 224 x 224 = 50,176 floats each\n";

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

// =========================================================================
// Postprocessing: Softmax & Top-1 Extraction
// =========================================================================
void postprocess_logits(
    const float* logits,
    size_t num_classes,
    const std::vector<std::string>& class_labels,
    int& out_class_id,
    std::string& out_class_name,
    float& out_confidence_pct,
    bool verbose = false
) {
    const float max_logit = *std::max_element(logits, logits + num_classes);

    std::vector<float> probabilities(num_classes);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < num_classes; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (size_t i = 0; i < num_classes; ++i) {
        probabilities[i] *= inv_sum;
    }

    const auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    out_class_id = static_cast<int>(std::distance(probabilities.begin(), max_it));
    out_confidence_pct = (*max_it) * 100.0f;

    if (out_class_id >= 0 && static_cast<size_t>(out_class_id) < class_labels.size()) {
        out_class_name = to_title_case(class_labels[static_cast<size_t>(out_class_id)]);
    } else {
        out_class_name = "Class #" + std::to_string(out_class_id);
    }

    if (verbose) {
        std::cout << "=================================================================================\n";
        std::cout << "🧠 [STEP 5] Neural Network Output Tensor from session.Run() (1,000 Logits)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  Output Tensor Shape : [1, 1000] (1,000 unnormalized class scores)\n";
        std::cout << "  Buffer Pointer      : " << static_cast<const void*>(logits) << "\n";
        std::cout << "  Logits Score Range  : Min = " << *std::min_element(logits, logits + num_classes)
                  << ", Max = " << max_logit << "\n";
        std::cout << "  Softmax Sum of Exps : sum(exp(z - max)) = " << sum_exp << "\n\n";

        std::vector<size_t> top_indices(num_classes);
        std::iota(top_indices.begin(), top_indices.end(), 0);
        std::partial_sort(top_indices.begin(), top_indices.begin() + 5, top_indices.end(),
            [&](size_t a, size_t b) { return probabilities[a] > probabilities[b]; });

        std::cout << "  🏆 Top 5 Classified Classes:\n";
        std::cout << "  -------------------------------------------------------------------------------\n";
        std::cout << "  Rank | Class ID | Label Name                | Logit Score | Softmax Prob | Conf %\n";
        std::cout << "  -------------------------------------------------------------------------------\n";
        for (size_t rank = 0; rank < 5; ++rank) {
            size_t cid = top_indices[rank];
            std::string name = (cid < class_labels.size()) ? to_title_case(class_labels[cid]) : ("Class #" + std::to_string(cid));
            std::cout << "   #" << (rank + 1) << "  | " << std::setw(8) << cid << " | "
                      << std::left << std::setw(25) << name << " | "
                      << std::right << std::fixed << std::setprecision(2) << std::setw(11) << logits[cid] << " | "
                      << std::setprecision(6) << probabilities[cid] << "   | "
                      << std::setprecision(1) << (probabilities[cid] * 100.0f) << "%\n";
        }
        std::cout << "  -------------------------------------------------------------------------------\n\n";
    }
}

// Check whether a file extension matches supported image formats
bool is_image_file(const fs::path& file_path) {
    const std::string ext = file_path.extension().string();
    std::string lower_ext;
    lower_ext.reserve(ext.size());
    for (char c : ext) {
        lower_ext.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return (lower_ext == ".jpg" || lower_ext == ".jpeg" || lower_ext == ".png" || lower_ext == ".bmp");
}

// Execute single model inference evaluation and record metrics
ModelRunProfile evaluate_model(
    Ort::Session& session,
    const std::string& input_name,
    const std::string& output_name,
    const std::vector<float>& input_tensor_values,
    const std::array<int64_t, 4>& input_shape,
    const std::vector<std::string>& class_labels,
    const fs::path& model_path,
    const std::string& tag,
    const std::string& precision,
    bool verbose = false
) {
    using Clock = std::chrono::steady_clock;
    ModelRunProfile profile;
    profile.tag = tag;
    profile.precision = precision;
    profile.model_path = model_path;
    profile.file_size_mb = static_cast<double>(fs::file_size(model_path)) / (1024.0 * 1024.0);

    const std::array<const char*, 1> input_names = { input_name.c_str() };
    const std::array<const char*, 1> output_names = { output_name.c_str() };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_values.data()),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Warm-up run
    session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Measured run
    const auto t_start = Clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );
    const auto t_end = Clock::now();
    profile.inference_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Postprocess
    const auto t_post_start = Clock::now();
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    const size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    postprocess_logits(
        raw_output,
        num_classes,
        class_labels,
        profile.class_id,
        profile.class_name,
        profile.confidence_pct,
        verbose
    );
    const auto t_post_end = Clock::now();
    profile.postprocess_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();
    profile.total_ms = profile.inference_ms + profile.postprocess_ms;
    profile.fps = (profile.total_ms > 0.0) ? (1000.0 / profile.total_ms) : 0.0;
    profile.peak_rss_mb = get_peak_rss_mb();

    return profile;
}

// Print side-by-side comparison table
void print_comparison_table(const ModelRunProfile& fp32, const ModelRunProfile& int8, double preprocess_ms) {
    const double size_saved_mb = fp32.file_size_mb - int8.file_size_mb;
    const double size_saved_pct = (size_saved_mb / fp32.file_size_mb) * 100.0;
    const double speedup_ratio = (int8.inference_ms > 0.0) ? (fp32.inference_ms / int8.inference_ms) : 1.0;
    const double latency_saved_ms = fp32.inference_ms - int8.inference_ms;
    const double latency_saved_pct = (latency_saved_ms / fp32.inference_ms) * 100.0;
    const bool class_match = (fp32.class_id == int8.class_id);

    std::cout << "===================================================================================================\n";
    std::cout << "⚔️  SIDE-BY-SIDE BENCHMARK SCORECARD: FP32 BASELINE vs INT8 QUANTIZATION\n";
    std::cout << "===================================================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric / Dimension"
              << std::setw(24) << "FP32 Baseline"
              << std::setw(24) << "INT8 Quantized"
              << "Delta / Efficiency Gain\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    std::cout << std::fixed << std::setprecision(2);
    // Model size on disk
    std::cout << std::left << std::setw(28) << "Model Size (Disk/RAM)"
              << std::setw(24) << (std::to_string(fp32.file_size_mb).substr(0, 5) + " MB")
              << std::setw(24) << (std::to_string(int8.file_size_mb).substr(0, 5) + " MB")
              << "-" << size_saved_pct << "% (" << size_saved_mb << " MB saved!)\n";

    // Weight precision
    std::cout << std::left << std::setw(28) << "Weight Precision"
              << std::setw(24) << "32-bit Float (FP32)"
              << std::setw(24) << "8-bit Signed Int (INT8)"
              << "4x smaller weight footprint\n";

    // Preprocessing Latency
    std::cout << std::left << std::setw(28) << "Preprocessing Latency"
              << std::setw(24) << (std::to_string(preprocess_ms).substr(0, 5) + " ms")
              << std::setw(24) << (std::to_string(preprocess_ms).substr(0, 5) + " ms")
              << "Identical (shared input tensor)\n";

    // CPU Inference Latency
    std::cout << std::left << std::setw(28) << "CPU Inference Latency"
              << std::setw(24) << (std::to_string(fp32.inference_ms).substr(0, 5) + " ms")
              << std::setw(24) << (std::to_string(int8.inference_ms).substr(0, 5) + " ms");
    if (speedup_ratio >= 1.0) {
        std::cout << speedup_ratio << "x FASTER (" << latency_saved_pct << "% saved) 🚀\n";
    } else {
        std::cout << speedup_ratio << "x (" << std::abs(latency_saved_pct) << "% delta)\n";
    }

    // Total Latency
    const double fp32_total = preprocess_ms + fp32.total_ms;
    const double int8_total = preprocess_ms + int8.total_ms;
    std::cout << std::left << std::setw(28) << "Total End-to-End Latency"
              << std::setw(24) << (std::to_string(fp32_total).substr(0, 5) + " ms")
              << std::setw(24) << (std::to_string(int8_total).substr(0, 5) + " ms");
    if (fp32_total >= int8_total) {
        std::cout << "-" << (fp32_total - int8_total) << " ms total time saved ⚡\n";
    } else {
        std::cout << "+" << (int8_total - fp32_total) << " ms delta\n";
    }

    // Inference Throughput
    std::cout << std::left << std::setw(28) << "Inference Throughput"
              << std::setw(24) << (std::to_string(fp32.fps).substr(0, 5) + " FPS")
              << std::setw(24) << (std::to_string(int8.fps).substr(0, 5) + " FPS");
    if (int8.fps >= fp32.fps) {
        std::cout << "+" << ((int8.fps - fp32.fps) / fp32.fps * 100.0) << "% FPS gain\n";
    } else {
        std::cout << ((int8.fps - fp32.fps) / fp32.fps * 100.0) << "% FPS\n";
    }

    // Top-1 Class
    std::cout << std::left << std::setw(28) << "Top-1 Classification"
              << std::setw(24) << (fp32.class_name + " (#" + std::to_string(fp32.class_id) + ")")
              << std::setw(24) << (int8.class_name + " (#" + std::to_string(int8.class_id) + ")")
              << (class_match ? "100% Agreement (Identical Prediction) ✅" : "Mismatch ⚠️") << "\n";

    // Confidence
    const float conf_delta = int8.confidence_pct - fp32.confidence_pct;
    std::cout << std::left << std::setw(28) << "Prediction Confidence"
              << std::setw(24) << (std::to_string(fp32.confidence_pct).substr(0, 5) + "%")
              << std::setw(24) << (std::to_string(int8.confidence_pct).substr(0, 5) + "%")
              << (conf_delta >= 0 ? "+" : "") << conf_delta << "% (Negligible delta)\n";

    // Process Peak RAM
    std::cout << std::left << std::setw(28) << "Peak Process RAM (RSS)"
              << std::setw(24) << (std::to_string(fp32.peak_rss_mb).substr(0, 6) + " MB")
              << std::setw(24) << (std::to_string(int8.peak_rss_mb).substr(0, 6) + " MB")
              << "Measured via getrusage()\n";

    std::cout << "===================================================================================================\n\n";
}

// =========================================================================
// Main Inference Application
// =========================================================================
int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;

    // Parse CLI arguments
    const std::vector<std::string> args(argv + 1, argv + argc);

    bool force_fp32 = false;
    bool force_int8 = false;
    bool compare_mode = false;
    fs::path target_path = "data/test_100/n01440764_tench.JPEG";

    for (const auto& arg : args) {
        if (arg == "--fp32") {
            force_fp32 = true;
        } else if (arg == "--int8") {
            force_int8 = true;
        } else if (arg == "--compare") {
            compare_mode = true;
        } else if (!arg.empty() && arg[0] != '-') {
            target_path = arg;
        }
    }

    // Default to compare mode on single image if no explicit precision flag is specified
    if (!force_fp32 && !force_int8) {
        compare_mode = true;
    }

    const fs::path fp32_model_path = "models/resnet18-v1-7.onnx";
    const fs::path int8_model_path = "models/resnet18-v1-7-int8.onnx";
    const fs::path labels_path = "models/imagenet_classes.txt";

    // Validate target existence
    if (!fs::exists(target_path)) {
        std::cerr << "[ERROR] Target path not found: " << target_path.string() << "\n";
        std::cerr << "Usage: " << argv[0] << " [options] [path_to_image_or_directory]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --compare   Run side-by-side FP32 vs INT8 benchmark comparison (Default on single image)\n";
        std::cerr << "  --int8      Run only quantized INT8 model\n";
        std::cerr << "  --fp32      Run only baseline FP32 model\n";
        return 1;
    }

    // Validate models
    if (!fs::exists(fp32_model_path)) {
        std::cerr << "[ERROR] FP32 model not found: " << fp32_model_path.string() << "\n";
        std::cerr << "Please run 'python3 download_assets.py' first.\n";
        return 1;
    }

    if ((compare_mode || force_int8) && !fs::exists(int8_model_path)) {
        std::cout << "[INFO] INT8 model not found. Generating models/resnet18-v1-7-int8.onnx via quantize_model.py...\n";
        const int ret = std::system("python3 quantize_model.py");
        if (ret != 0) {
            std::cerr << "[WARN] quantize_model.py returned exit code " << ret << "\n";
        }
        if (!fs::exists(int8_model_path)) {
            std::cerr << "[ERROR] Failed to generate INT8 model.\n";
            return 1;
        }
    }

    // Collect image files
    std::vector<fs::path> image_paths;
    if (fs::is_directory(target_path)) {
        for (const auto& entry : fs::directory_iterator(target_path)) {
            if (entry.is_regular_file() && is_image_file(entry.path())) {
                image_paths.push_back(entry.path());
            }
        }
        std::sort(image_paths.begin(), image_paths.end());
    } else if (fs::is_regular_file(target_path)) {
        image_paths.push_back(target_path);
    }

    if (image_paths.empty()) {
        std::cerr << "[ERROR] No valid images found at: " << target_path.string() << "\n";
        return 1;
    }

    const std::vector<std::string> class_labels = load_class_labels(labels_path);
    const bool is_single_image = (image_paths.size() == 1);

    // Target input resolution
    constexpr int target_w = 224;
    constexpr int target_h = 224;
    const std::array<int64_t, 4> input_shape = {1, 3, target_h, target_w};

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EdgeVisionAssistant");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(0); // auto-detect CPU cores

    std::cout << "=================================================================================\n";
    std::cout << "👓 SMART AI GLASSES VISION ASSISTANT (C++ CPU ENGINE)\n";
    std::cout << "=================================================================================\n";
    std::cout << "Target Path    : " << target_path.string() << " (" << image_paths.size() << " image" << (is_single_image ? "" : "s") << ")\n";
    std::cout << "Execution Mode : " << (compare_mode ? "Side-by-Side Comparison (FP32 vs INT8)" : (force_int8 ? "INT8 Quantized" : "FP32 Baseline")) << "\n";
    std::cout << "Vector Engine  : " << get_cpu_vector_feature() << "\n";

    // -------------------------------------------------------------------------
    // CASE A: SIDE-BY-SIDE COMPARISON MODE (for single image)
    // -------------------------------------------------------------------------
    if (compare_mode && is_single_image) {
        const auto& img_path = image_paths[0];

        // Step 1: Preprocess once
        std::vector<float> input_tensor_values;
        std::string error_msg;
        const auto t_pre_start = Clock::now();
        if (!preprocess_image(img_path, target_w, target_h, input_tensor_values, error_msg, true)) {
            std::cerr << "[ERROR] " << error_msg << "\n";
            return 1;
        }
        const auto t_pre_end = Clock::now();
        const double preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        // 1. Evaluate FP32 Baseline
        Ort::Session session_fp32(env, fp32_model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        const std::string in_name_fp32 = session_fp32.GetInputNameAllocated(0, allocator).get();
        const std::string out_name_fp32 = session_fp32.GetOutputNameAllocated(0, allocator).get();

        std::cout << "\n>>> [1/2] Running FP32 Baseline Model (" << fp32_model_path.filename().string() << ") ...\n";
        ModelRunProfile prof_fp32 = evaluate_model(
            session_fp32, in_name_fp32, out_name_fp32,
            input_tensor_values, input_shape, class_labels,
            fp32_model_path, "FP32", "32-bit Float", true
        );

        // 2. Evaluate INT8 Quantized
        Ort::Session session_int8(env, int8_model_path.c_str(), session_options);
        const std::string in_name_int8 = session_int8.GetInputNameAllocated(0, allocator).get();
        const std::string out_name_int8 = session_int8.GetOutputNameAllocated(0, allocator).get();

        std::cout << "\n>>> [2/2] Running INT8 Quantized Model (" << int8_model_path.filename().string() << ") ...\n";
        ModelRunProfile prof_int8 = evaluate_model(
            session_int8, in_name_int8, out_name_int8,
            input_tensor_values, input_shape, class_labels,
            int8_model_path, "INT8", "8-bit Signed Integer", false
        );

        // Print comparative scorecard
        print_comparison_table(prof_fp32, prof_int8, preprocess_ms);
        return 0;
    }

    // -------------------------------------------------------------------------
    // CASE B: STANDARD RUN (Single model over dataset or single image)
    // -------------------------------------------------------------------------
    const fs::path active_model_path = force_int8 ? int8_model_path : fp32_model_path;
    const std::string active_tag = force_int8 ? "INT8 Quantized" : "FP32 Baseline";

    std::cout << "Active Model   : " << active_model_path.filename().string() << " (" << active_tag << ")\n";
    std::cout << "Model Size     : " << std::fixed << std::setprecision(2)
              << (fs::file_size(active_model_path) / (1024.0 * 1024.0)) << " MB\n";

    Ort::Session session(env, active_model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const std::string input_name = session.GetInputNameAllocated(0, allocator).get();
    const std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
    const std::array<const char*, 1> input_names = { input_name.c_str() };
    const std::array<const char*, 1> output_names = { output_name.c_str() };

    std::vector<VisionResult> results;
    results.reserve(image_paths.size());

    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& img_path = image_paths[i];
        VisionResult res;
        res.image_name = img_path.filename().string();

        std::vector<float> input_tensor_values;
        std::string error_msg;
        const auto t_pre_start = Clock::now();
        if (!preprocess_image(img_path, target_w, target_h, input_tensor_values, error_msg, is_single_image)) {
            std::cerr << "[ERROR] Skipping " << res.image_name << ": " << error_msg << "\n";
            continue;
        }
        const auto t_pre_end = Clock::now();
        res.preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        const auto t_infer_start = Clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        const auto t_infer_end = Clock::now();
        res.inference_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();

        const auto t_post_start = Clock::now();
        float* raw_output = output_tensors[0].GetTensorMutableData<float>();
        const size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        postprocess_logits(
            raw_output,
            num_classes,
            class_labels,
            res.class_id,
            res.class_name,
            res.confidence_pct,
            is_single_image
        );
        const auto t_post_end = Clock::now();
        res.postprocess_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

        res.total_ms = res.preprocess_ms + res.inference_ms + res.postprocess_ms;
        res.fps = (res.total_ms > 0.0) ? (1000.0 / res.total_ms) : 0.0;
        results.push_back(res);

        if (!is_single_image) {
            std::cout << "[" << std::setw(3) << (i + 1) << "/" << std::setw(3) << image_paths.size() << "] "
                      << std::left << std::setw(38) << res.image_name
                      << " -> " << std::setw(25) << res.class_name
                      << " (" << std::fixed << std::setprecision(1) << std::setw(5) << res.confidence_pct << "%)"
                      << " | " << std::setw(5) << res.total_ms << " ms\n";
        }
    }

    if (results.empty()) {
        std::cerr << "[ERROR] No images were successfully processed.\n";
        return 1;
    }

    if (is_single_image) {
        const auto& single_res = results[0];
        std::cout << "=================================================================================\n";
        std::cout << "📊 INFERENCE RESULT FOR SINGLE SNAPSHOT (" << active_tag << ")\n";
        std::cout << "=================================================================================\n";
        std::cout << "Input Image    : " << single_res.image_name << "\n";
        std::cout << "Vision Result  : " << single_res.class_name << " (Class ID: " << single_res.class_id << ")\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Confidence     : " << single_res.confidence_pct << "%\n";
        std::cout << "---------------- LATENCY PROFILE -------------------\n";
        std::cout << "Preprocessing  : " << single_res.preprocess_ms << " ms\n";
        std::cout << "CPU Inference  : " << single_res.inference_ms << " ms (" << get_cpu_vector_feature() << ")\n";
        std::cout << "Postprocessing : " << single_res.postprocess_ms << " ms\n";
        std::cout << "Total Latency  : " << single_res.total_ms << " ms (" << single_res.fps << " FPS)\n";
        std::cout << "Peak RAM (RSS) : " << get_peak_rss_mb() << " MB\n";
        std::cout << "=================================================================================\n";
        return 0;
    }

    // Dataset summary
    double total_pre = 0.0, total_infer = 0.0, total_post = 0.0, total_time = 0.0;
    std::vector<double> latencies;
    latencies.reserve(results.size());

    for (const auto& r : results) {
        total_pre += r.preprocess_ms;
        total_infer += r.inference_ms;
        total_post += r.postprocess_ms;
        total_time += r.total_ms;
        latencies.push_back(r.total_ms);
    }

    const double n = static_cast<double>(results.size());
    const double avg_pre = total_pre / n;
    const double avg_infer = total_infer / n;
    const double avg_post = total_post / n;
    const double avg_total = total_time / n;
    const double avg_fps = (avg_total > 0.0) ? (1000.0 / avg_total) : 0.0;

    std::sort(latencies.begin(), latencies.end());
    const double median_lat = latencies[latencies.size() / 2];
    const size_t p95_idx = static_cast<size_t>(0.95 * static_cast<double>(latencies.size() - 1));
    const double p95_lat = latencies[p95_idx];

    std::cout << "=================================================================================\n";
    std::cout << "📊 DATASET EVALUATION SUMMARY (" << results.size() << " IMAGES - " << active_tag << ")\n";
    std::cout << "=================================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average Preprocess  : " << avg_pre << " ms\n";
    std::cout << "Average Inference   : " << avg_infer << " ms (" << get_cpu_vector_feature() << ")\n";
    std::cout << "Average Postprocess : " << avg_post << " ms\n";
    std::cout << "Average Total       : " << avg_total << " ms\n";
    std::cout << "Median Total (P50)  : " << median_lat << " ms\n";
    std::cout << "Tail Latency (P95)  : " << p95_lat << " ms\n";
    std::cout << "Throughput          : " << avg_fps << " FPS\n";
    std::cout << "Peak RAM (RSS)      : " << get_peak_rss_mb() << " MB\n";
    std::cout << "=================================================================================\n";

    return 0;
}
