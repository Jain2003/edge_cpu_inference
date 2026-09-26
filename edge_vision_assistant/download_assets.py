#!/usr/bin/env python3
"""
download_assets.py
------------------
Downloads required pre-trained ONNX models, class labels, sample test images,
and header libraries for the Edge Vision Assistant.

Targets:
  - models/resnet18-v1-7.onnx
  - models/imagenet_classes.txt
  - data/cat.jpg, data/dog.jpg, data/car.jpg
  - include/stb_image.h, include/stb_image_resize2.h
"""

import os
import sys
import json
import urllib.request
from pathlib import Path


def download_file(url: str, destination: Path, description: str):
    """Download a file with progress reporting."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[OK] {description} already exists: {destination} ({destination.stat().st_size:,} bytes)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FETCHING] {description} from {url} ...")

    # Custom opener with User-Agent to avoid 403 Forbidden from some CDNs
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (EdgeVisionAssistant/1.0)"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 64 * 1024  # 64 KB

            with open(destination, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  -> Progress: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  -> Downloaded: {mb_done:.1f} MB")
                        sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"[SAVED] {destination} ({downloaded:,} bytes)")
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_imagenet_classes(destination: Path):
    """Download and format ImageNet 1000 class names."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[OK] ImageNet labels already exist: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FETCHING] ImageNet 1000 class labels ...")

    # Primary source: anishathalye simple labels JSON
    url_json = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        req = urllib.request.Request(url_json, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            labels = json.loads(response.read().decode("utf-8"))
            with open(destination, "w", encoding="utf-8") as f:
                for label in labels:
                    f.write(f"{label}\n")
            print(f"[SAVED] {len(labels)} class labels saved to {destination}")
            return
    except Exception as e:
        print(f"[WARN] Failed fetching JSON labels ({e}), falling back to PyTorch Hub text list...")

    # Fallback source: PyTorch Hub imagenet_classes.txt
    url_txt = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    download_file(url_txt, destination, "ImageNet 1000 labels (text fallback)")


def main():
    # Resolve directory relative to this script
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    data_dir = script_dir / "data"
    include_dir = script_dir / "include"

    print("=" * 60)
    print("👓 Edge Vision Assistant - Asset Downloader")
    print("=" * 60)

    # 1. Download Pre-trained ONNX Model (ResNet-18)
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
    download_file(model_url, models_dir / "resnet18-v1-7.onnx", "ResNet-18 ONNX Model")

    # 2. Download ImageNet Class Labels
    download_imagenet_classes(models_dir / "imagenet_classes.txt")


    # 4. Download STB Image Headers (header-only image loading & resizing)
    stb_headers = {
        "stb_image.h": "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h",
        "stb_image_resize2.h": "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h",
    }
    for filename, url in stb_headers.items():
        download_file(url, include_dir / filename, f"STB Header: {filename}")

    print("=" * 60)
    print(" All assets downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
