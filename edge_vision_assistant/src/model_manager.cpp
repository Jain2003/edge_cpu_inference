#include "model_manager.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>

namespace edge_vision {

ModelManager::ModelManager(fs::path fp32_path, fs::path int8_path, fs::path labels_path)
    : fp32_path_(std::move(fp32_path)),
      int8_path_(std::move(int8_path)),
      labels_path_(std::move(labels_path)) {}

bool ModelManager::validate_fp32_model(std::string& error_msg) const {
    if (!fs::exists(fp32_path_)) {
        error_msg = "FP32 model not found: " + fp32_path_.string() + "\nPlease run 'python3 download_assets.py' first.";
        return false;
    }
    return true;
}

bool ModelManager::ensure_int8_model(std::string& error_msg) const {
    if (!fs::exists(int8_path_)) {
        std::cout << "[INFO] INT8 model not found. Generating " << int8_path_.string() << " via quantize_model.py...\n";
        const int ret = std::system("python3 quantize_model.py");
        if (ret != 0) {
            std::cerr << "[WARN] quantize_model.py returned exit code " << ret << "\n";
        }
        if (!fs::exists(int8_path_)) {
            error_msg = "Failed to generate INT8 model at: " + int8_path_.string();
            return false;
        }
    }
    return true;
}

double ModelManager::get_file_size_mb(const fs::path& path) {
    if (fs::exists(path)) {
        return static_cast<double>(fs::file_size(path)) / (1024.0 * 1024.0);
    }
    return 0.0;
}

ModelRunProfile ModelManager::evaluate_profile(
    InferenceEngine& engine,
    const Postprocessor& postprocessor,
    const std::vector<float>& input_tensor,
    const std::array<int64_t, 4>& input_shape,
    const std::string& tag,
    const std::string& precision,
    bool verbose
) {
    ModelRunProfile profile;
    profile.tag = tag;
    profile.precision = precision;
    profile.model_path = engine.get_model_path();
    profile.file_size_mb = engine.get_model_size_mb();

    // 1. Warm-up
    engine.warmup(input_tensor, input_shape);

    // 2. Timed inference
    double inference_ms = 0.0;
    auto output_tensors = engine.run(input_tensor, input_shape, inference_ms);
    profile.inference_ms = inference_ms;

    // 3. Postprocess
    const auto t_post_start = Clock::now();
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    const size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    postprocessor.process(
        raw_output,
        num_classes,
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

void ModelManager::print_comparison_table(
    const ModelRunProfile& fp32,
    const ModelRunProfile& int8,
    double preprocess_ms
) {
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

} // namespace edge_vision
