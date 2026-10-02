#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <filesystem>
#include <chrono>
#include <memory>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <algorithm>

#if defined(__linux__) || defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace edge_vision {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Results for an individual image inference pass
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

// Top-K prediction item
struct ClassificationPrediction {
    size_t rank = 0;
    int class_id = -1;
    std::string class_name = "Unknown";
    float logit = 0.0f;
    float probability = 0.0f;
    float confidence_pct = 0.0f;
};

// Execution profile metrics for model comparison (FP32 vs INT8)
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

// Feature Flags for extensible capability selection (-f f1, -f f2, etc.)
enum class FeatureMode {
    Auto,           // Default behavior (compare on single image, F1 on dataset)
    Compare,        // Feature comparison: F1 (FP32) vs F2 (INT8)
    F1_FP32,        // Feature 1: FP32 Baseline ONNX Runtime CPU Pipeline
    F2_INT8,        // Feature 2: INT8 Dynamic Quantization
    F3_SIMD,        // Feature 3: SIMD AVX2 Vector Preprocessing (Planned)
    F4_ASYNC,       // Feature 4: Asynchronous Multi-threaded 4-Core Pipeline (Planned)
    F5_STATIC_INT8  // Feature 5: Static Calibration INT8 Quantization (Planned)
};

// Measure process peak resident set size (RSS) in MB
inline double get_peak_rss_mb() {
#if defined(__linux__) || defined(__unix__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0; // ru_maxrss is in Kilobytes on Linux
    }
#endif
    return 0.0;
}

// Detect CPU SIMD vector feature
inline std::string get_cpu_vector_feature() {
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
inline std::string to_title_case(std::string_view input) {
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

// Convert string to lower case
inline std::string to_lower_str(std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return result;
}

// Check whether a file extension matches supported image formats
inline bool is_image_file(const fs::path& file_path) {
    const std::string ext = file_path.extension().string();
    std::string lower_ext;
    lower_ext.reserve(ext.size());
    for (char c : ext) {
        lower_ext.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return (lower_ext == ".jpg" || lower_ext == ".jpeg" || lower_ext == ".png" || lower_ext == ".bmp");
}

} // namespace edge_vision
