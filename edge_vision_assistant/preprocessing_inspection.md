# 🔬 Deep Preprocessing Inspection Report

**Target Image**: `data/test_100/n01440764_tench.JPEG`  
**Purpose**: Step-by-step mathematical & memory audit of raw image ingestion, resizing, ImageNet normalization, and planar (HWC -> NCHW) tensor conversion.

---

## 📸 1. Raw Image Ingestion (`raw_rgb`)

When `stbi_load()` decodes the JPEG file, it allocates a continuous block of unsigned 8-bit integers (`unsigned char`) on the heap.

| Property | Value | Explanation |
| :--- | :--- | :--- |
| **Heap Memory Address** | `0x5da970a99e60` | Pointer returned by `stbi_load` wrapped in `UniqueStbImage` |
| **Original Resolution** | `500 × 375` | Width × Height in pixels |
| **Original File Channels** | `3` | Channels detected in source JPEG |
| **Decoded Channels** | `3 (RGB)` | Forced to 3 by passing `3` to `stbi_load` |
| **Total Raw Pixels** | `187500` | Total spatial pixels |
| **Total Memory Size** | `562500 bytes` (~0.54 MB) | `orig_w × orig_h × 3` |
| **Memory Layout** | Interleaved `[R0, G0, B0, R1, G1, B1, ...]` | Standard HWC (Height, Width, Channel) format |

### First 8 Raw Pixels in Memory (`raw_rgb`)

| Pixel # | Spatial (X, Y) | Red (Dec / Hex) | Green (Dec / Hex) | Blue (Dec / Hex) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | (0, 0) | `71` (`0x47`) | `82` (`0x52`) | `76` (`0x4C`) |
| 1 | (1, 0) | `70` (`0x46`) | `79` (`0x4F`) | `76` (`0x4C`) |
| 2 | (2, 0) | `73` (`0x49`) | `83` (`0x53`) | `82` (`0x52`) |
| 3 | (3, 0) | `71` (`0x47`) | `90` (`0x5A`) | `88` (`0x58`) |
| 4 | (4, 0) | `69` (`0x45`) | `88` (`0x58`) | `86` (`0x56`) |
| 5 | (5, 0) | `73` (`0x49`) | `85` (`0x55`) | `85` (`0x55`) |
| 6 | (6, 0) | `74` (`0x4A`) | `86` (`0x56`) | `84` (`0x54`) |
| 7 | (7, 0) | `68` (`0x44`) | `89` (`0x59`) | `82` (`0x52`) |

---

## 📐 2. Bilinear Resizing (`resized_rgb`)

The neural network expects an input resolution of exactly `224 × 224`. `stbir_resize_uint8_linear()` samples adjacent pixels and computes weighted averages to produce smooth, anti-aliased downscaling.

| Property | Value | Explanation |
| :--- | :--- | :--- |
| **Target Resolution** | `224 × 224` | Required ResNet-18 spatial dimension |
| **Vector Buffer Type** | `std::vector<unsigned char>` | Heap-managed C++ dynamic array |
| **Vector Size** | `150528 bytes` (~147 KB) | `224 × 224 × 3` bytes |
| **Vector Memory Pointer** | `0x5DA97051DD70` | Pointer returned by `resized_rgb.data()` |

### Resized Pixels at 5 Key Anchor Locations

| Anchor Location | (X, Y) | Memory Index (`(Y*224+X)*3`) | Red (`uint8`) | Green (`uint8`) | Blue (`uint8`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-Left** | (0, 0) | `0` | `70` | `81` | `77` |
| **Top-Right** | (223, 0) | `669` | `4` | `4` | `4` |
| **Center** | (112, 112) | `75600` | `224` | `184` | `69` |
| **Bottom-Left** | (0, 223) | `149856` | `60` | `69` | `69` |
| **Bottom-Right** | (223, 223) | `150525` | `83` | `91` | `91` |

---

## 🧮 3. Normalization & Planar Layout Conversion

This stage transforms interleaved integers into floating-point planar tensors using the ImageNet standard formula:

$$\text{Pixel}_{\text{scaled}} = \frac{\text{Byte}}{255.0}$$
$$\text{Tensor Value} = \frac{\text{Pixel}_{\text{scaled}} - \mu}{\sigma}$$

Where ImageNet statistics are:
* **Red Channel ($\mu_0, \sigma_0$)**: $\mu = 0.485$, $\sigma = 0.229$
* **Green Channel ($\mu_1, \sigma_1$)**: $\mu = 0.456$, $\sigma = 0.224$
* **Blue Channel ($\mu_2, \sigma_2$)**: $\mu = 0.406$, $\sigma = 0.225$

### Step-by-Step Calculation Walkthrough for Sample Pixels

| Pixel | Channel | Raw `uint8` | Scaled (`/ 255.0`) | Minus Mean (`- μ`) | Divided by Std (`/ σ`) | Output Tensor Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-Left (0,0)** | Red | `70` | `0.2745` | `-0.2105` | **`-0.9192`** | `[0]` (`0 × 50176 + 0`) |
| **Top-Left (0,0)** | Green | `81` | `0.3176` | `-0.1384` | **`-0.6176`** | `[50176]` (`1 × 50176 + 0`) |
| **Top-Left (0,0)** | Blue | `77` | `0.3020` | `-0.1040` | **`-0.4624`** | `[100352]` (`2 × 50176 + 0`) |
| **Top-Right (223,0)** | Red | `4` | `0.0157` | `-0.4693` | **`-2.0494`** | `[223]` (`0 × 50176 + 223`) |
| **Top-Right (223,0)** | Green | `4` | `0.0157` | `-0.4403` | **`-1.9657`** | `[50399]` (`1 × 50176 + 223`) |
| **Top-Right (223,0)** | Blue | `4` | `0.0157` | `-0.3903` | **`-1.7347`** | `[100575]` (`2 × 50176 + 223`) |
| **Center (112,112)** | Red | `224` | `0.8784` | `+0.3934` | **`+1.7180`** | `[25200]` (`0 × 50176 + 25200`) |
| **Center (112,112)** | Green | `184` | `0.7216` | `+0.2656` | **`+1.1856`** | `[75376]` (`1 × 50176 + 25200`) |
| **Center (112,112)** | Blue | `69` | `0.2706` | `-0.1354` | **`-0.6018`** | `[125552]` (`2 × 50176 + 25200`) |
| **Bottom-Left (0,223)** | Red | `60` | `0.2353` | `-0.2497` | **`-1.0904`** | `[49952]` (`0 × 50176 + 49952`) |
| **Bottom-Left (0,223)** | Green | `69` | `0.2706` | `-0.1854` | **`-0.8277`** | `[100128]` (`1 × 50176 + 49952`) |
| **Bottom-Left (0,223)** | Blue | `69` | `0.2706` | `-0.1354` | **`-0.6018`** | `[150304]` (`2 × 50176 + 49952`) |
| **Bottom-Right (223,223)** | Red | `83` | `0.3255` | `-0.1595` | **`-0.6965`** | `[50175]` (`0 × 50176 + 50175`) |
| **Bottom-Right (223,223)** | Green | `91` | `0.3569` | `-0.0991` | **`-0.4426`** | `[100351]` (`1 × 50176 + 50175`) |
| **Bottom-Right (223,223)** | Blue | `91` | `0.3569` | `-0.0491` | **`-0.2184`** | `[150527]` (`2 × 50176 + 50175`) |

---

## 📦 4. Output Tensor Anatomy (`output_tensor`)

The output tensor is a contiguous 1D `std::vector<float>` of size **150,528 elements** (602,112 bytes) formatted in **NCHW Planar** layout:

```text
Index 0                                    Index 50,176                               Index 100,352                          Index 150,527
┌──────────────────────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┐
│       CHANNEL 0: RED (224 × 224)         │       CHANNEL 1: GREEN (224 × 224)       │       CHANNEL 2: BLUE (224 × 224)        │
│             50,176 floats                │             50,176 floats                │             50,176 floats                │
└──────────────────────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘
```

### Channel-by-Channel Tensor Statistics

| Channel | Index Range | Min Value | Max Value | Mean Value | Expected Visual Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Channel 0 (Red)** | `[0 .. 50175]` | `-2.067` | `2.215` | `-0.295` | Normalized red color intensity plane |
| **Channel 1 (Green)** | `[50176 .. 100351]` | `-1.966` | `2.394` | `-0.088` | Normalized green color intensity plane |
| **Channel 2 (Blue)** | `[100352 .. 150527]` | `-1.752` | `2.501` | `-0.153` | Normalized blue color intensity plane |

### Top-Left 5 × 5 Tensor Grid Previews

#### Channel 0 (Red) (Top-Left 5 × 5 Spatial Grid)

```text
 -0.919  -0.902  -0.885  -0.936  -0.902 
 -0.953  -0.919  -0.868  -0.936  -0.919 
 -1.022  -0.988  -0.919  -0.936  -0.953 
 -1.056  -1.056  -0.988  -0.953  -0.936 
 -1.056  -1.039  -0.988  -0.953  -0.936 
```

#### Channel 1 (Green) (Top-Left 5 × 5 Spatial Grid)

```text
 -0.618  -0.530  -0.513  -0.548  -0.583 
 -0.600  -0.513  -0.513  -0.565  -0.530 
 -0.618  -0.548  -0.548  -0.565  -0.530 
 -0.670  -0.635  -0.600  -0.565  -0.548 
 -0.688  -0.635  -0.583  -0.565  -0.565 
```

#### Channel 2 (Blue) (Top-Left 5 × 5 Spatial Grid)

```text
 -0.462  -0.323  -0.323  -0.393  -0.393 
 -0.428  -0.358  -0.340  -0.393  -0.375 
 -0.428  -0.428  -0.393  -0.393  -0.375 
 -0.515  -0.480  -0.445  -0.410  -0.393 
 -0.567  -0.462  -0.428  -0.393  -0.393 
```

---

## 🚀 5. Why This Preprocessing Is Common to Edge & Cloud AI

1. **Neural Network Invariance**: A PyTorch/ONNX ResNet-18 model trained in the cloud requires the identical numerical distribution whether running on an AWS cloud server or on smart AI glasses.
2. **Zero-Copy Tensor Creation**: `Ort::Value::CreateTensor()` takes `output_tensor.data()` directly without any memory copies, giving optimal microsecond CPU performance.
