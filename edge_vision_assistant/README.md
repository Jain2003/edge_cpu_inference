# 👓 Edge Vision Assistant (C++ CPU Inference)

A lightweight, high-performance C++ CPU inference engine simulating a **Smart AI Glasses Snapshot Assistant**.

When the user takes a photo snapshot with their AI glasses and asks *"What is this?"*, the executable ingests the image path, executes single-image ONNX CPU inference via the ONNX Runtime C++ API, and outputs human-readable classification results with detailed latency profiling.

Supports both **FP32 Baseline** and **INT8 Dynamic Quantization** with automated side-by-side performance, latency, and memory comparison.

---

## 📁 Directory Structure

```text
edge_vision_assistant/
├── CMakeLists.txt              # Cross-platform CMake build configuration (auto-fetches ONNX Runtime)
├── download_assets.py          # Python script to download ResNet-18, ImageNet labels, and sample images
├── quantize_model.py           # Generates 8-bit quantized ONNX model (shrinks weights by ~75%)
├── benchmark_100.py            # Automated 100-image ImageNet benchmarking & report generator
├── results/                    # Generated benchmark evaluation reports
│   ├── benchmark_results.xlsx  # Detailed styled Excel report with accuracy & latency metrics
│   └── benchmark_results.csv   # Raw CSV export of all benchmarked runs
├── include/                    # Header-only dependencies (stb_image, stb_image_resize2)
├── data/                       # ImageNet benchmark test dataset
│   └── test_100/               # 100 diverse ImageNet benchmark images
├── models/                     # Pre-trained ONNX models & ImageNet classes
│   ├── resnet18-v1-7.onnx      # Baseline FP32 ResNet-18 model (~44.7 MB)
│   ├── resnet18-v1-7-int8.onnx # Dynamically quantized INT8 ResNet-18 model (~11.3 MB)
│   └── imagenet_classes.txt    # 1,000 ImageNet synset category labels
├── src/
│   └── main.cpp                # Modern C++ dataset & snapshot inference engine (FP32 vs INT8)
└── README.md                   # Build, usage, and architectural documentation
```

---

## ⚡ Quick Start

### 1. Download Model & Assets

Run the asset download script once to fetch the pre-trained `resnet18-v1-7.onnx` model and the ImageNet 1,000 class labels:

```bash
python3 download_assets.py
```

### 2. Generate INT8 Quantized Model (Optional)

Generate the 8-bit quantized model using dynamic quantization:

```bash
python3 quantize_model.py
```
*(If the INT8 model is missing when running in comparison mode, the C++ application automatically runs this script for you).*

### 3. Configure & Build with CMake

```bash
cmake -B build
cmake --build build --config Release
```

*Note: CMake automatically downloads and configures the official pre-built ONNX Runtime C++ binaries for your platform (Linux x86_64, aarch64, macOS, Windows) if not already installed!*

### 4. Run Side-by-Side Comparison (FP32 vs INT8)

Run directly on any snapshot image to trigger the side-by-side benchmark scorecard:

```bash
./build/edge_vision_assistant data/test_100/n01440764_tench.JPEG
```

You can also explicitly select the execution precision:

```bash
# Run only INT8 quantized model
./build/edge_vision_assistant --int8 data/test_100/n01440764_tench.JPEG

# Run only FP32 baseline model
./build/edge_vision_assistant --fp32 data/test_100/n01440764_tench.JPEG

# Evaluate entire 100-image dataset
./build/edge_vision_assistant data/test_100
```

---

## ⚔️ Side-by-Side Benchmark Scorecard

When evaluating a snapshot in `--compare` mode, the C++ engine measures preprocessing, CPU inference latency across both models, prediction fidelity, and peak resident memory (RSS via `getrusage`):

```text
===================================================================================================
⚔️  SIDE-BY-SIDE BENCHMARK SCORECARD: FP32 BASELINE vs INT8 QUANTIZATION
===================================================================================================
Metric / Dimension          FP32 Baseline           INT8 Quantized          Delta / Efficiency Gain
---------------------------------------------------------------------------------------------------
Model Size (Disk/RAM)       44.65 MB                11.25 MB                -74.78% (33.39 MB saved!)
Weight Precision            32-bit Float (FP32)     8-bit Signed Int (INT8) 4x smaller weight footprint
Preprocessing Latency       18.22 ms                18.22 ms                Identical (shared input tensor)
CPU Inference Latency       22.28 ms                23.86 ms                0.93x (7.12% delta)
Total End-to-End Latency    40.80 ms                42.11 ms                +1.31 ms delta
Inference Throughput        44.28 FPS               41.85 FPS               -5.50% FPS
Top-1 Classification        Tench (#0)              Tench (#0)              100% Agreement (Identical Prediction) ✅
Prediction Confidence       99.16%                  99.39%                  +0.22% (Negligible delta)
Peak Process RAM (RSS)      400.16 MB               400.16 MB               Measured via getrusage()
===================================================================================================
```

---

## 📊 100-Image ImageNet Benchmark Suite

The repository includes a comprehensive evaluation and profiling suite ([`benchmark_100.py`](benchmark_100.py)) that validates model accuracy and real-world latency distribution over 100 diverse ImageNet test images.

### Running the Benchmark

```bash
python3 benchmark_100.py
```

### What the Benchmark Does:
1. **Automated Dataset Sourcing**: Downloads 100 diverse images sampled across all 1,000 ImageNet categories into `data/test_100/`.
2. **Ground Truth Validation**: Maps synset identifiers to official human-readable class names.
3. **C++ Subprocess Execution**: Evaluates `./build/edge_vision_assistant` against each image snapshot.
4. **Statistical Aggregation**: Computes mean, median, P90, and P95 latency percentiles for preprocessing, CPU inference, and end-to-end execution.
5. **Formatted Exports**: Saves results to `results/benchmark_results.csv` and a styled multi-sheet Excel workbook `results/benchmark_results.xlsx`.

---

## 🔍 Pipeline Architecture (`src/main.cpp`)

1. **CLI Ingestion**: Parses arguments (`--compare`, `--int8`, `--fp32`, path) using modern C++ `std::vector<std::string>`.
2. **Preprocessing**:
   - Ingests image from disk via STB Image (`stbi_load`).
   - High-quality bilinear resize to model input dimensions (224 × 224) via `stbir_resize_uint8_linear`.
   - Normalizes pixels using ImageNet mean & standard deviation ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - Converts interleaved RGB (HWC) to planar NCHW layout `[1, 3, 224, 224]`.
3. **Dual-Model Execution**: Evaluates `Ort::Session::Run()` for both FP32 and INT8 models with warm-up cycles.
4. **Postprocessing**: Computes numerically stable Softmax, selects Top-1 Argmax class, and looks up human-readable ImageNet label with title formatting.
5. **Profiling**: Measures microsecond timestamps via `std::chrono::steady_clock` across Preprocessing, CPU Inference, and aggregate execution, along with peak resident memory (`getrusage()`).

---

## 💡 C++ Design & Best Practices

### 1. Modern C++ Argument Parsing & Memory Profiling
In [`src/main.cpp`](src/main.cpp):
- Uses `std::vector<std::string>` and `std::filesystem::path` to eliminate raw pointer arithmetic and C-style char arrays.
- Employs `getrusage(RUSAGE_SELF, ...)` on Linux to inspect resident process memory (RSS) in megabytes.
- Manages STB Image buffers via `std::unique_ptr<unsigned char[], StbImageDeleter>` RAII handles.

### 2. Why Explicit `std::` Instead of `using namespace std;`?
This codebase follows modern C++ industry guidelines (including C++ Core Guidelines SF.7):
- **Avoid Namespace Pollution**: The `std` namespace defines hundreds of generic identifiers (`size`, `count`, `distance`, `move`, `min`, `max`). Using `using namespace std;` brings all of them into global scope, risking name collisions.
- **Explicit Provenance & Readability**: Explicitly writing `std::vector`, `std::string`, `std::chrono` makes it immediately clear which types belong to the standard library vs third-party APIs (`Ort::`) or project structures (`VisionResult`).
- **Targeted Scope Aliases**: Where abbreviation improves readability without namespace pollution, targeted local aliases are used instead (e.g., `using Clock = std::chrono::steady_clock;` inside functions).

---

## 🚀 Optimization & Customization Points

- **Zero-Allocation Arenas**: Pre-allocate continuous tensor buffers to eliminate heap allocations across consecutive frames.
- **SIMD Vector Intrinsics**: Replace pixel normalization loops with explicit AVX2/AVX-512 (`_mm256_fmadd_ps`) or ARM NEON (`vfmaq_f32`) vector intrinsics.
- **VNNI Acceleration**: On CPUs supporting Intel DL Boost / VNNI (`vpdpbusd`), INT8 quantized matrix multiplications execute with double the throughput of standard AVX2.
