#include "postprocessor.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace edge_vision {

Postprocessor::Postprocessor(const fs::path& labels_path)
    : labels_path_(labels_path) {
    load_labels();
}

void Postprocessor::load_labels() {
    labels_.clear();
    std::ifstream file(labels_path_);
    if (!file.is_open()) {
        std::cerr << "[WARN] Warning: Could not open labels file: " << labels_path_.string()
                  << ". Using raw indices.\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }
}

std::vector<ClassificationPrediction> Postprocessor::get_top_k(
    const float* logits,
    size_t num_classes,
    size_t k
) const {
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

    std::vector<size_t> top_indices(num_classes);
    std::iota(top_indices.begin(), top_indices.end(), 0);
    const size_t sort_count = std::min(k, num_classes);
    std::partial_sort(
        top_indices.begin(),
        top_indices.begin() + sort_count,
        top_indices.end(),
        [&](size_t a, size_t b) { return probabilities[a] > probabilities[b]; }
    );

    std::vector<ClassificationPrediction> results;
    results.reserve(sort_count);

    for (size_t rank = 0; rank < sort_count; ++rank) {
        const size_t cid = top_indices[rank];
        ClassificationPrediction pred;
        pred.rank = rank + 1;
        pred.class_id = static_cast<int>(cid);
        if (cid < labels_.size()) {
            pred.class_name = to_title_case(labels_[cid]);
        } else {
            pred.class_name = "Class #" + std::to_string(cid);
        }
        pred.logit = logits[cid];
        pred.probability = probabilities[cid];
        pred.confidence_pct = probabilities[cid] * 100.0f;
        results.push_back(pred);
    }

    return results;
}

void Postprocessor::process(
    const float* logits,
    size_t num_classes,
    int& out_class_id,
    std::string& out_class_name,
    float& out_confidence_pct,
    bool verbose
) const {
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

    if (out_class_id >= 0 && static_cast<size_t>(out_class_id) < labels_.size()) {
        out_class_name = to_title_case(labels_[static_cast<size_t>(out_class_id)]);
    } else {
        out_class_name = "Class #" + std::to_string(out_class_id);
    }

    if (verbose) {
        std::cout << "=================================================================================\n";
        std::cout << "🧠 [STEP 5] Neural Network Output Tensor from session.Run() (1,000 Logits)\n";
        std::cout << "=================================================================================\n";
        std::cout << "  Output Tensor Shape : [1, " << num_classes << "] (" << num_classes << " unnormalized class scores)\n";
        std::cout << "  Buffer Pointer      : " << static_cast<const void*>(logits) << "\n";
        std::cout << "  Logits Score Range  : Min = " << *std::min_element(logits, logits + num_classes)
                  << ", Max = " << max_logit << "\n";
        std::cout << "  Softmax Sum of Exps : sum(exp(z - max)) = " << sum_exp << "\n\n";

        const auto top5 = get_top_k(logits, num_classes, 5);

        std::cout << "  🏆 Top 5 Classified Classes:\n";
        std::cout << "  -------------------------------------------------------------------------------\n";
        std::cout << "  Rank | Class ID | Label Name                | Logit Score | Softmax Prob | Conf %\n";
        std::cout << "  -------------------------------------------------------------------------------\n";
        for (const auto& pred : top5) {
            std::cout << "   #" << pred.rank << "  | " << std::setw(8) << pred.class_id << " | "
                      << std::left << std::setw(25) << pred.class_name << " | "
                      << std::right << std::fixed << std::setprecision(2) << std::setw(11) << pred.logit << " | "
                      << std::setprecision(6) << pred.probability << "   | "
                      << std::setprecision(1) << pred.confidence_pct << "%\n";
        }
        std::cout << "  -------------------------------------------------------------------------------\n\n";
    }
}

} // namespace edge_vision
