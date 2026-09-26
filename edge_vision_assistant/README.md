# 👓 Edge Vision Assistant (C++ CPU Inference)

A lightweight, high-performance C++ CPU inference engine simulating a **Smart AI Glasses Snapshot Assistant**.

When the user takes a photo snapshot with their AI glasses and asks *"What is this?"*, the executable ingests the image path, executes single-image ONNX CPU inference via the ONNX Runtime C++ API, and outputs human-readable classification results with detailed latency profiling.

---

## 📁 Directory Structure

```text
edge_vision_assistant/
├── CMakeLists.txt              # Cross-platform CMake build configuration (auto-fetches ONNX Runtime)
├── download_assets.py          # Python script to download ResNet-18, ImageNet labels, and sample images
├── benchmark_100.py            # Automated 100-image ImageNet benchmarking & report generator
├── results/                    # Generated benchmark evaluation reports
│   ├── benchmark_results.xlsx  # Detailed styled Excel report with accuracy & latency metrics
│   └── benchmark_results.csv   # Raw CSV export of all benchmarked runs
├── include/                    # Header-only dependencies (stb_image, stb_image_resize2)
├── data/                       # ImageNet benchmark test dataset
│   └── test_100/               # 100 diverse ImageNet benchmark images
├── models/                     # Pre-trained ONNX model & ImageNet classes
│   ├── resnet18-v1-7.onnx      # Pre-trained ResNet-18 model
│   └── imagenet_classes.txt    # 1,000 ImageNet synset category labels
├── src/
│   └── main.cpp                # Modern C++ dataset & snapshot inference engine
└── README.md                   # Build, usage, and architectural documentation
```

---

## ⚡ Quick Start

### 1. Download Model & Assets

Run the asset download script once to fetch the pre-trained `resnet18-v1-7.onnx` model and the ImageNet 1,000 class labels:

```bash
python3 download_assets.py
```

### 2. Configure & Build with CMake

```bash
cmake -B build
cmake --build build --config Release
```

*Note: CMake automatically downloads and configures the official pre-built ONNX Runtime C++ binaries for your platform (Linux x86_64, aarch64, macOS, Windows) if not already installed!*

### 3. Run Inference on the ImageNet Dataset

Run without arguments to evaluate all 100 images in `data/test_100/` and view the dataset summary:

```bash
./build/edge_vision_assistant
```

Or pass any directory or individual image path directly via CLI:

```bash
./build/edge_vision_assistant data/test_100
./build/edge_vision_assistant data/test_100/n01440764_tench.JPEG
./build/edge_vision_assistant /path/to/any/custom_image.jpg
```

---

## 🖥️ Example CLI Output

```text
====================================================
👓 SMART AI GLASSES VISION ASSISTANT (C++ CPU ENGINE)
====================================================
Input Image    : data/cat.jpg
Vision Result  : Tabby Cat (Class ID: 281)
Confidence     : 94.2%
---------------- LATENCY PROFILE -------------------
Preprocessing  : 1.8 ms
CPU Inference  : 11.4 ms (AVX2 Vector Cores)
Total Latency  : 13.2 ms (75.8 FPS)
====================================================
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

### Summary Benchmark Metrics

| Metric | Measured Value |
| :--- | :--- |
| **Total Test Images** | 100 diverse categories |
| **Top-1 Accuracy** | High agreement with ground-truth synset labels |
| **Inference Hardware** | Edge CPU with SIMD / AVX2 Vector Kernels |
| **P95 Tail Latency** | Profiled under burst snapshot loads |
| **Reports Produced** | `results/benchmark_results.xlsx`, `results/benchmark_results.csv` |

---

## 🔍 Pipeline Architecture (`src/main.cpp`)

1. **CLI Ingestion**: Accepts a single image file path representing a snapshot capture event (with fallback default).
2. **Preprocessing**:
   - Ingests image from disk via STB Image (`stbi_load`).
   - High-quality bilinear resize to model input dimensions (224 × 224) via `stbir_resize_uint8_linear`.
   - Normalizes pixels using ImageNet mean & standard deviation ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - Converts interleaved RGB (HWC) to planar NCHW layout `[1, 3, 224, 224]`.
3. **Execution**: Evaluates `Ort::Session::Run()` using ONNX Runtime's optimized vector execution provider.
4. **Postprocessing**: Computes numerically stable Softmax, selects Top-1 Argmax class, and looks up human-readable ImageNet label with title formatting.
5. **Profiling**: Measures microsecond timestamps via `std::chrono::steady_clock` across Preprocessing, CPU Inference, and aggregate execution.

---

## 💡 C++ Design & Best Practices

### 1. Modern C++ Argument Parsing & Dataset Fallback
In [`src/main.cpp`](src/main.cpp):
```cpp
// Parse CLI arguments using standard C++ strings (defaults to dataset "data/test_100")
const std::vector<std::string> args(argv + 1, argv + argc);
const fs::path target_path = args.empty() ? fs::path("data/test_100") : fs::path(args[0]);
```
- **Why?** Using `std::vector<std::string>` and `std::filesystem::path` completely avoids C-style raw pointer arithmetic and raw character array handling.
- **Flexibility**: If no arguments are passed, it defaults to evaluating the entire `data/test_100` benchmark suite. If an individual image or custom folder is passed, it dynamically detects and processes it.

### 2. Why Explicit `std::` Instead of `using namespace std;`?
This codebase follows modern C++ industry guidelines (including C++ Core Guidelines SF.7):
- **Avoid Namespace Pollution**: The `std` namespace defines hundreds of generic identifiers (such as `size`, `count`, `distance`, `move`, `min`, `max`). Using `using namespace std;` brings all of them into global scope, risking name collisions with third-party headers (ONNX Runtime, STB Image) and project types.
- **Explicit Provenance & Readability**: Explicitly writing `std::vector`, `std::string`, `std::chrono` makes it immediately apparent which types belong to the standard library vs third-party APIs (`Ort::`) or project structures (`VisionResult`).
- **Targeted Scope Aliases**: Where abbreviation improves readability without namespace pollution, targeted local aliases are used instead (e.g., `using Clock = std::chrono::steady_clock;` inside `main()`, or `namespace fs = std::filesystem;`).

---

## 🚀 Optimization & Customization Points

The pipeline is intentionally structured into clean, modular blocks with marked optimization extension points:
- **Zero-Allocation Arenas**: Pre-allocate continuous tensor buffers to eliminate heap allocations across consecutive frames.
- **SIMD Vector Intrinsics**: Replace pixel normalization loops with explicit AVX2/AVX-512 (`_mm256_fmadd_ps`) or ARM NEON (`vfmaq_f32`) vector intrinsics.
- **Multi-Threading**: Tune `session_options.SetIntraOpNumThreads()` for your specific edge CPU core configuration.
