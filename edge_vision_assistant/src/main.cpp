/**
 * edge_vision_assistant - Smart AI Glasses Snapshot Assistant
 * ==========================================================
 * High-performance, modular C++ CPU inference pipeline simulating
 * a Smart AI Glasses visual query engine ("What is this?").
 *
 * Supports extensible feature flags:
 *  - -f f1, --fp32    : Feature 1 (FP32 Baseline CPU Inference)
 *  - -f f2, --int8    : Feature 2 (INT8 Dynamic Quantization)
 *  - -f compare       : Side-by-Side Comparison Scorecard (F1 vs F2)
 *  - -f f3 (planned)  : Feature 3 (SIMD AVX2 Vector Preprocessing)
 *  - -f f4 (planned)  : Feature 4 (Async Multi-threaded 4-Core Pipeline)
 *  - -f f5 (planned)  : Feature 5 (Static Calibration Quantization)
 */

#include "types.hpp"
#include "preprocessor.hpp"
#include "inference_engine.hpp"
#include "postprocessor.hpp"
#include "model_manager.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <algorithm>

using namespace edge_vision;

void print_help(const char* prog_name) {
    std::cout << "=================================================================================\n";
    std::cout << "👓 SMART AI GLASSES VISION ASSISTANT (C++ CPU ENGINE)\n";
    std::cout << "=================================================================================\n";
    std::cout << "Usage: " << prog_name << " [options] [path_to_image_or_directory]\n\n";
    std::cout << "Feature Flags:\n";
    std::cout << "  -f, --feature <name>  Select pipeline feature mode to execute:\n";
    std::cout << "                          f1, fp32       : Feature 1 (FP32 Baseline CPU Inference)\n";
    std::cout << "                          f2, int8       : Feature 2 (INT8 Dynamic Quantization, ~75% smaller)\n";
    std::cout << "                          compare, comp  : Side-by-side benchmark comparison (F1 vs F2)\n";
    std::cout << "                          f3, simd       : Feature 3 (SIMD AVX2 Preprocessing - Planned)\n";
    std::cout << "                          f4, async      : Feature 4 (Async 4-Core Pipeline - Planned)\n";
    std::cout << "                          f5, static_int8: Feature 5 (Static Quantization - Planned)\n\n";
    std::cout << "Convenience Aliases:\n";
    std::cout << "  --compare             Alias for -f compare (Default on single image)\n";
    std::cout << "  --fp32                Alias for -f f1\n";
    std::cout << "  --int8                Alias for -f f2\n";
    std::cout << "  -h, --help            Show this help reference and exit\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " -f f1 data/test_100/n01440764_tench.JPEG\n";
    std::cout << "  " << prog_name << " -f f2 data/test_100/n01440764_tench.JPEG\n";
    std::cout << "  " << prog_name << " -f compare data/test_100/n01440764_tench.JPEG\n";
    std::cout << "  " << prog_name << " -f f2 data/test_100\n";
    std::cout << "=================================================================================\n";
}

int main(int argc, char* argv[]) {
    // 1. Parse CLI arguments
    const std::vector<std::string> args(argv + 1, argv + argc);

    FeatureMode feature_mode = FeatureMode::Auto;
    fs::path target_path = "data/test_100/n01440764_tench.JPEG";

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "-f" || arg == "--feature") {
            if (i + 1 < args.size()) {
                const std::string val = to_lower_str(args[++i]);
                if (val == "f1" || val == "fp32" || val == "1") {
                    feature_mode = FeatureMode::F1_FP32;
                } else if (val == "f2" || val == "int8" || val == "2") {
                    feature_mode = FeatureMode::F2_INT8;
                } else if (val == "compare" || val == "comp" || val == "both" || val == "f1_vs_f2") {
                    feature_mode = FeatureMode::Compare;
                } else if (val == "f3" || val == "simd") {
                    feature_mode = FeatureMode::F3_SIMD;
                } else if (val == "f4" || val == "async") {
                    feature_mode = FeatureMode::F4_ASYNC;
                } else if (val == "f5" || val == "static_int8") {
                    feature_mode = FeatureMode::F5_STATIC_INT8;
                } else {
                    std::cerr << "[ERROR] Unknown feature flag: '" << val << "'\n";
                    std::cerr << "Run '" << argv[0] << " --help' to view available features.\n";
                    return 1;
                }
            } else {
                std::cerr << "[ERROR] Missing argument for " << arg << "\n";
                return 1;
            }
        } else if (arg.rfind("-f=", 0) == 0 || arg.rfind("--feature=", 0) == 0) {
            const size_t eq_pos = arg.find('=');
            const std::string val = to_lower_str(arg.substr(eq_pos + 1));
            if (val == "f1" || val == "fp32" || val == "1") {
                feature_mode = FeatureMode::F1_FP32;
            } else if (val == "f2" || val == "int8" || val == "2") {
                feature_mode = FeatureMode::F2_INT8;
            } else if (val == "compare" || val == "comp" || val == "both" || val == "f1_vs_f2") {
                feature_mode = FeatureMode::Compare;
            } else if (val == "f3" || val == "simd") {
                feature_mode = FeatureMode::F3_SIMD;
            } else if (val == "f4" || val == "async") {
                feature_mode = FeatureMode::F4_ASYNC;
            } else if (val == "f5" || val == "static_int8") {
                feature_mode = FeatureMode::F5_STATIC_INT8;
            } else {
                std::cerr << "[ERROR] Unknown feature flag: '" << val << "'\n";
                std::cerr << "Run '" << argv[0] << " --help' to view available features.\n";
                return 1;
            }
        } else if (arg == "--fp32") {
            feature_mode = FeatureMode::F1_FP32;
        } else if (arg == "--int8") {
            feature_mode = FeatureMode::F2_INT8;
        } else if (arg == "--compare") {
            feature_mode = FeatureMode::Compare;
        } else if (!arg.empty() && arg[0] != '-') {
            target_path = arg;
        }
    }

    // Planned feature guards
    if (feature_mode == FeatureMode::F3_SIMD) {
        std::cout << "[INFO] Feature 3 (SIMD AVX2 Vector Preprocessing) is currently in development.\n";
        std::cout << "Currently active features: -f f1 (FP32), -f f2 (INT8), -f compare.\n";
        return 0;
    } else if (feature_mode == FeatureMode::F4_ASYNC) {
        std::cout << "[INFO] Feature 4 (Asynchronous Multi-threaded 4-Core Pipeline) is currently in development.\n";
        std::cout << "Currently active features: -f f1 (FP32), -f f2 (INT8), -f compare.\n";
        return 0;
    } else if (feature_mode == FeatureMode::F5_STATIC_INT8) {
        std::cout << "[INFO] Feature 5 (Static Calibration INT8 Quantization) is currently in development.\n";
        std::cout << "Currently active features: -f f1 (FP32), -f f2 (INT8), -f compare.\n";
        return 0;
    }

    // 2. Validate Target Path
    if (!fs::exists(target_path)) {
        std::cerr << "[ERROR] Target path not found: " << target_path.string() << "\n";
        std::cerr << "Run '" << argv[0] << " --help' for usage instructions.\n";
        return 1;
    }

    // Collect image paths
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

    const bool is_single_image = (image_paths.size() == 1);

    // Resolve Auto mode: single image defaults to Compare, dataset defaults to F1 FP32
    if (feature_mode == FeatureMode::Auto) {
        feature_mode = is_single_image ? FeatureMode::Compare : FeatureMode::F1_FP32;
    }

    const bool compare_mode = (feature_mode == FeatureMode::Compare);
    const bool is_int8_mode = (feature_mode == FeatureMode::F2_INT8);

    // 3. Initialize Model & Pipeline Managers
    ModelManager model_mgr;
    std::string error_msg;

    if (!model_mgr.validate_fp32_model(error_msg)) {
        std::cerr << "[ERROR] " << error_msg << "\n";
        return 1;
    }

    if ((compare_mode || is_int8_mode) && !model_mgr.ensure_int8_model(error_msg)) {
        std::cerr << "[ERROR] " << error_msg << "\n";
        return 1;
    }

    Preprocessor preprocessor;
    Postprocessor postprocessor(model_mgr.get_labels_path());
    const auto input_shape = preprocessor.get_input_shape();

    // 4. Banner Output
    std::string feature_title;
    if (compare_mode) {
        feature_title = "Side-by-Side Comparison (Feature 1 FP32 vs Feature 2 INT8)";
    } else if (is_int8_mode) {
        feature_title = "Feature 2: INT8 Dynamic Quantization";
    } else {
        feature_title = "Feature 1: FP32 Baseline Pipeline";
    }

    std::cout << "=================================================================================\n";
    std::cout << "👓 SMART AI GLASSES VISION ASSISTANT (C++ CPU ENGINE)\n";
    std::cout << "=================================================================================\n";
    std::cout << "Target Path    : " << target_path.string() << " (" << image_paths.size() << " image" << (is_single_image ? "" : "s") << ")\n";
    std::cout << "Active Feature : " << feature_title << "\n";
    std::cout << "Vector Engine  : " << get_cpu_vector_feature() << "\n";

    // -------------------------------------------------------------------------
    // CASE A: SIDE-BY-SIDE COMPARISON MODE (for single image)
    // -------------------------------------------------------------------------
    if (compare_mode && is_single_image) {
        const auto& img_path = image_paths[0];

        // Step 1: Preprocess once
        std::vector<float> input_tensor_values;
        const auto t_pre_start = Clock::now();
        if (!preprocessor.process(img_path, input_tensor_values, error_msg, true)) {
            std::cerr << "[ERROR] " << error_msg << "\n";
            return 1;
        }
        const auto t_pre_end = Clock::now();
        const double preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        // 1. Evaluate FP32 Baseline
        InferenceEngine engine_fp32(model_mgr.get_fp32_path());
        std::cout << "\n>>> [1/2] Running FP32 Baseline Model (" << model_mgr.get_fp32_path().filename().string() << ") ...\n";
        ModelRunProfile prof_fp32 = ModelManager::evaluate_profile(
            engine_fp32, postprocessor,
            input_tensor_values, input_shape,
            "FP32", "32-bit Float", true
        );

        // 2. Evaluate INT8 Quantized
        InferenceEngine engine_int8(model_mgr.get_int8_path());
        std::cout << "\n>>> [2/2] Running INT8 Quantized Model (" << model_mgr.get_int8_path().filename().string() << ") ...\n";
        ModelRunProfile prof_int8 = ModelManager::evaluate_profile(
            engine_int8, postprocessor,
            input_tensor_values, input_shape,
            "INT8", "8-bit Signed Integer", false
        );

        // Print comparative scorecard
        ModelManager::print_comparison_table(prof_fp32, prof_int8, preprocess_ms);
        return 0;
    }

    // -------------------------------------------------------------------------
    // CASE B: STANDARD RUN (Single model over dataset or single image)
    // -------------------------------------------------------------------------
    const fs::path active_model_path = is_int8_mode ? model_mgr.get_int8_path() : model_mgr.get_fp32_path();
    const std::string active_tag = is_int8_mode ? "Feature 2 (INT8 Quantized)" : "Feature 1 (FP32 Baseline)";

    std::cout << "Active Model   : " << active_model_path.filename().string() << " (" << active_tag << ")\n";
    std::cout << "Model Size     : " << std::fixed << std::setprecision(2)
              << ModelManager::get_file_size_mb(active_model_path) << " MB\n";

    InferenceEngine engine(active_model_path);
    std::vector<VisionResult> results;
    results.reserve(image_paths.size());

    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& img_path = image_paths[i];
        VisionResult res;
        res.image_name = img_path.filename().string();

        std::vector<float> input_tensor_values;
        const auto t_pre_start = Clock::now();
        if (!preprocessor.process(img_path, input_tensor_values, error_msg, is_single_image)) {
            std::cerr << "[ERROR] Skipping " << res.image_name << ": " << error_msg << "\n";
            continue;
        }
        const auto t_pre_end = Clock::now();
        res.preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        // Run inference
        double inference_ms = 0.0;
        auto output_tensors = engine.run(input_tensor_values, input_shape, inference_ms);
        res.inference_ms = inference_ms;

        // Postprocess logits
        const auto t_post_start = Clock::now();
        float* raw_output = output_tensors[0].GetTensorMutableData<float>();
        const size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        postprocessor.process(
            raw_output,
            num_classes,
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
