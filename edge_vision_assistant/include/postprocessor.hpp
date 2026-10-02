#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <filesystem>

namespace edge_vision {

/**
 * Postprocessor
 * -------------
 * Ingests raw output logits from the model, computes numerically stable
 * Softmax probabilities, selects the Argmax Top-1 class, and maps to
 * human-readable ImageNet class synset labels.
 */
class Postprocessor {
public:
    explicit Postprocessor(const fs::path& labels_path);

    // Compute Top-1 prediction and populate class_id, class_name, and confidence_pct
    void process(
        const float* logits,
        size_t num_classes,
        int& out_class_id,
        std::string& out_class_name,
        float& out_confidence_pct,
        bool verbose = false
    ) const;

    // Extract Top-K predictions sorted by confidence
    std::vector<ClassificationPrediction> get_top_k(
        const float* logits,
        size_t num_classes,
        size_t k = 5
    ) const;

    const std::vector<std::string>& get_labels() const { return labels_; }
    size_t get_num_labels() const { return labels_.size(); }

private:
    fs::path labels_path_;
    std::vector<std::string> labels_;

    void load_labels();
};

} // namespace edge_vision
