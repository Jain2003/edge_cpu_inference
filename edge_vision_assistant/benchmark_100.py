#!/usr/bin/env python3
"""
benchmark_100.py
----------------
Downloads 100 diverse ImageNet test images, executes the C++ Edge Vision
Assistant executable on each image, measures classification accuracy and latency,
and exports the comprehensive results into an Excel workbook (and CSV).

Outputs:
  - data/test_100/             (100 diverse sample images)
  - benchmark_results.xlsx     (Styled multi-tab Excel workbook)
  - benchmark_results.csv      (Plain CSV format)
"""

import os
import re
import sys
import json
import time
import urllib.request
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_BIN = SCRIPT_DIR / "build" / "edge_vision_assistant"
DATA_100_DIR = SCRIPT_DIR / "data" / "test_100"
RESULTS_DIR = SCRIPT_DIR / "results"
EXCEL_OUTPUT = RESULTS_DIR / "benchmark_results.xlsx"
CSV_OUTPUT = RESULTS_DIR / "benchmark_results.csv"


def download_synset_mapping():
    """Fetch the ImageNet synset to integer class ID mapping."""
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # Format: {"281": ["n02123045", "tabby"], ...}
    synset_to_info = {}
    for class_id_str, val in data.items():
        synset = val[0]
        class_name = val[1].replace("_", " ").title()
        synset_to_info[synset] = (int(class_id_str), class_name)
    return synset_to_info


def fetch_image_list():
    """Select 100 diverse images from EliSchwartz/imagenet-sample-images."""
    url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read().decode("utf-8"))

    image_files = [x for x in items if x["name"].lower().endswith((".jpeg", ".jpg", ".png"))]

    # Select 100 evenly distributed across the 1000 categories (step = 10)
    selected = [image_files[i] for i in range(0, len(image_files), 10)][:100]
    return selected


def download_single_image(item, destination_dir: Path):
    """Worker to download a single image."""
    dest = destination_dir / item["name"]
    if dest.exists() and dest.stat().st_size > 0:
        return item["name"], True

    url = item["download_url"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return item["name"], True
    except Exception as e:
        return item["name"], False


def download_100_images():
    """Download 100 images concurrently."""
    DATA_100_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching repository catalog for 100 diverse ImageNet test samples...")
    selected_items = fetch_image_list()
    print(f"Downloading {len(selected_items)} images into {DATA_100_DIR}...")

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_image, item, DATA_100_DIR): item for item in selected_items}
        for future in as_completed(futures):
            name, success = future.result()
            completed += 1
            sys.stdout.write(f"\r  -> Progress: {completed}/{len(selected_items)} images downloaded")
            sys.stdout.flush()

    print("\n Download of 100 test images complete!")
    return [item["name"] for item in selected_items]


def run_cpp_inference(image_path: Path):
    """Execute the C++ binary and parse the formatted output."""
    cmd = [str(BUILD_BIN), str(image_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    out = result.stdout

    # Parse output using regular expressions
    res_match = re.search(r"Vision Result\s*:\s*(.*?)\s*\(Class ID:\s*(\d+)\)", out)
    conf_match = re.search(r"Confidence\s*:\s*([0-9.]+)%", out)
    pre_match = re.search(r"Preprocessing\s*:\s*([0-9.]+)\s*ms", out)
    inf_match = re.search(r"CPU Inference\s*:\s*([0-9.]+)\s*ms", out)
    tot_match = re.search(r"Total Latency\s*:\s*([0-9.]+)\s*ms\s*\(([0-9.]+)\s*FPS\)", out)

    if not (res_match and conf_match and pre_match and inf_match and tot_match):
        raise ValueError(f"Failed to parse C++ output for {image_path.name}:\n{out}")

    pred_name = res_match.group(1).strip()
    pred_id = int(res_match.group(2))
    confidence_pct = float(conf_match.group(1))
    pre_ms = float(pre_match.group(1))
    inf_ms = float(inf_match.group(1))
    total_ms = float(tot_match.group(1))
    fps = float(tot_match.group(2))

    return {
        "pred_name": pred_name,
        "pred_id": pred_id,
        "confidence_pct": confidence_pct,
        "pre_ms": pre_ms,
        "inf_ms": inf_ms,
        "total_ms": total_ms,
        "fps": fps,
    }


def create_styled_excel(records, summary):
    """Generate a clean, styled Excel workbook using openpyxl."""
    wb = openpyxl.Workbook()

    # Define color palette & styles
    font_family = "Segoe UI"
    c_primary = "1F4E79"       # Deep blue
    c_accent = "2E75B6"        # Slate blue
    c_card_bg = "EDF2F8"       # Light gray-blue
    c_zebra = "F9FAFC"         # Light zebra stripe
    c_correct_bg = "E2EFDA"    # Soft green
    c_correct_fg = "375623"    # Dark green text
    c_incorrect_bg = "FCE4D6"  # Soft peach/red
    c_incorrect_fg = "C65911"  # Dark red/orange text

    border_thin = Side(border_style="thin", color="D9D9D9")
    box_border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)

    # -------------------------------------------------------------------------
    # TAB 1: SUMMARY DASHBOARD
    # -------------------------------------------------------------------------
    ws_dash = wb.active
    ws_dash.title = "Executive Summary"
    ws_dash.views.sheetView[0].showGridLines = True

    # Title Banner
    ws_dash.merge_cells("B2:G2")
    title_cell = ws_dash["B2"]
    title_cell.value = "👓 EDGE VISION ASSISTANT - 100-IMAGE BENCHMARK REPORT"
    title_cell.font = Font(name=font_family, size=14, bold=True, color="FFFFFF")
    title_cell.fill = PatternFill(start_color=c_primary, end_color=c_primary, fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[2].height = 36

    # Subtitle / Metadata
    ws_dash.merge_cells("B3:G3")
    sub_cell = ws_dash["B3"]
    sub_cell.value = f"Platform: Linux C++ CPU Engine (AVX2 Vector SIMD) | Model: ResNet-18 (ONNX Runtime) | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    sub_cell.font = Font(name=font_family, size=9, italic=True, color="595959")
    sub_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[3].height = 20

    # Summary KPI Cards
    kpis = [
        ("Total Images Tested", summary["total"], "images"),
        ("Top-1 Match Accuracy", f"{summary['accuracy_pct']:.1f}%", f"{summary['correct']}/{summary['total']} exact matches"),
        ("Avg Preprocessing", f"{summary['avg_pre_ms']:.2f} ms", "STB resize + normalization"),
        ("Avg CPU Inference", f"{summary['avg_inf_ms']:.2f} ms", "AVX2 SIMD Vector Cores"),
        ("Avg Total Latency", f"{summary['avg_total_ms']:.2f} ms", f"{summary['avg_fps']:.1f} FPS throughput"),
        ("95th Percentile Latency", f"{summary['p95_total_ms']:.2f} ms", "P95 tail latency"),
    ]

    row_start = 5
    for i, (title, val, note) in enumerate(kpis):
        r = row_start + (i // 2) * 3
        c_idx = 2 if (i % 2 == 0) else 5
        col1 = get_column_letter(c_idx)
        col2 = get_column_letter(c_idx + 2)

        ws_dash.merge_cells(f"{col1}{r}:{col2}{r}")
        cell_t = ws_dash[f"{col1}{r}"]
        cell_t.value = title.upper()
        cell_t.font = Font(name=font_family, size=9, bold=True, color="595959")
        cell_t.fill = PatternFill(start_color=c_card_bg, end_color=c_card_bg, fill_type="solid")
        cell_t.alignment = Alignment(horizontal="center", vertical="center")

        ws_dash.merge_cells(f"{col1}{r+1}:{col2}{r+1}")
        cell_v = ws_dash[f"{col1}{r+1}"]
        cell_v.value = val
        cell_v.font = Font(name=font_family, size=16, bold=True, color=c_primary)
        cell_v.fill = PatternFill(start_color=c_card_bg, end_color=c_card_bg, fill_type="solid")
        cell_v.alignment = Alignment(horizontal="center", vertical="center")

        ws_dash.merge_cells(f"{col1}{r+2}:{col2}{r+2}")
        cell_n = ws_dash[f"{col1}{r+2}"]
        cell_n.value = note
        cell_n.font = Font(name=font_family, size=8, italic=True, color="7F7F7F")
        cell_n.fill = PatternFill(start_color=c_card_bg, end_color=c_card_bg, fill_type="solid")
        cell_n.alignment = Alignment(horizontal="center", vertical="center")

    # Latency Percentiles Table
    ptable_row = 15
    ws_dash.cell(row=ptable_row, column=2, value="LATENCY DISTRIBUTION BREAKDOWN (ms)").font = Font(name=font_family, size=10, bold=True, color=c_primary)

    headers_p = ["Metric", "Preprocessing", "CPU Inference", "Total Latency"]
    for col_idx, h in enumerate(headers_p, start=2):
        c = ws_dash.cell(row=ptable_row+1, column=col_idx, value=h)
        c.font = Font(name=font_family, size=9, bold=True, color="FFFFFF")
        c.fill = PatternFill(start_color=c_accent, end_color=c_accent, fill_type="solid")
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = box_border

    p_rows = [
        ("Minimum", f"{summary['min_pre']:.2f} ms", f"{summary['min_inf']:.2f} ms", f"{summary['min_total']:.2f} ms"),
        ("Median (P50)", f"{summary['med_pre']:.2f} ms", f"{summary['med_inf']:.2f} ms", f"{summary['med_total']:.2f} ms"),
        ("90th Percentile (P90)", f"{summary['p90_pre']:.2f} ms", f"{summary['p90_inf']:.2f} ms", f"{summary['p90_total']:.2f} ms"),
        ("95th Percentile (P95)", f"{summary['p95_pre']:.2f} ms", f"{summary['p95_inf']:.2f} ms", f"{summary['p95_total']:.2f} ms"),
        ("Maximum", f"{summary['max_pre']:.2f} ms", f"{summary['max_inf']:.2f} ms", f"{summary['max_total']:.2f} ms"),
    ]

    for offset, row_data in enumerate(p_rows, start=ptable_row+2):
        for c_offset, val in enumerate(row_data, start=2):
            cell = ws_dash.cell(row=offset, column=c_offset, value=val)
            cell.font = Font(name=font_family, size=9)
            cell.alignment = Alignment(horizontal="center" if c_offset > 2 else "left", vertical="center")
            cell.border = box_border
            if offset % 2 == 1:
                cell.fill = PatternFill(start_color=c_zebra, end_color=c_zebra, fill_type="solid")

    # Column widths for Dashboard
    ws_dash.column_dimensions["A"].width = 3
    for col_l in ["B", "C", "D", "E", "F", "G"]:
        ws_dash.column_dimensions[col_l].width = 18

    # -------------------------------------------------------------------------
    # TAB 2: DETAILED INFERENCE RECORDS
    # -------------------------------------------------------------------------
    ws_details = wb.create_sheet(title="All 100 Image Results")
    ws_details.views.sheetView[0].showGridLines = True

    detail_headers = [
        "Index",
        "Image Filename",
        "Ground Truth Class",
        "GT ID",
        "Predicted Class",
        "Pred ID",
        "Top-1 Match",
        "Confidence (%)",
        "Preprocessing (ms)",
        "Inference (ms)",
        "Total Latency (ms)",
        "FPS"
    ]

    # Header row
    for col_idx, h in enumerate(detail_headers, start=1):
        cell = ws_details.cell(row=1, column=col_idx, value=h)
        cell.font = Font(name=font_family, size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color=c_primary, end_color=c_primary, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = box_border
    ws_details.row_dimensions[1].height = 28

    # Data rows
    for row_idx, r in enumerate(records, start=2):
        ws_details.row_dimensions[row_idx].height = 20
        is_even = (row_idx % 2 == 0)
        row_bg = c_zebra if is_even else "FFFFFF"

        is_match = (r["gt_id"] == r["pred_id"])
        match_str = "MATCH" if is_match else "MISMATCH"

        vals = [
            r["index"],
            r["filename"],
            r["gt_name"],
            r["gt_id"],
            r["pred_name"],
            r["pred_id"],
            match_str,
            r["confidence_pct"],
            r["pre_ms"],
            r["inf_ms"],
            r["total_ms"],
            r["fps"]
        ]

        for col_idx, val in enumerate(vals, start=1):
            c = ws_details.cell(row=row_idx, column=col_idx, value=val)
            c.font = Font(name=font_family, size=9)
            c.border = box_border

            # Default row background
            c.fill = PatternFill(start_color=row_bg, end_color=row_bg, fill_type="solid")

            # Column-specific formatting and alignment
            if col_idx in (1, 4, 6):  # IDs and Index
                c.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx in (2, 3, 5):  # Names
                c.alignment = Alignment(horizontal="left", vertical="center")
            elif col_idx == 7:  # Match column styling
                c.alignment = Alignment(horizontal="center", vertical="center")
                if is_match:
                    c.fill = PatternFill(start_color=c_correct_bg, end_color=c_correct_bg, fill_type="solid")
                    c.font = Font(name=font_family, size=9, bold=True, color=c_correct_fg)
                else:
                    c.fill = PatternFill(start_color=c_incorrect_bg, end_color=c_incorrect_bg, fill_type="solid")
                    c.font = Font(name=font_family, size=9, bold=True, color=c_incorrect_fg)
            elif col_idx == 8:  # Confidence
                c.alignment = Alignment(horizontal="right", vertical="center")
                c.number_format = "0.0\"%\""
            elif col_idx in (9, 10, 11):  # Latency
                c.alignment = Alignment(horizontal="right", vertical="center")
                c.number_format = "0.00"
            elif col_idx == 12:  # FPS
                c.alignment = Alignment(horizontal="right", vertical="center")
                c.number_format = "0.0"

    # Auto-adjust column widths for readability
    for col in ws_details.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws_details.column_dimensions[col_letter].width = max(max_len + 3, 11)

    wb.save(EXCEL_OUTPUT)
    print(f" Excel workbook successfully created: {EXCEL_OUTPUT}")


def export_csv(records):
    """Write records to CSV file."""
    import csv
    fieldnames = [
        "Index", "Filename", "Ground_Truth_Class", "Ground_Truth_ID",
        "Predicted_Class", "Predicted_ID", "Top1_Match", "Confidence_Pct",
        "Preprocessing_ms", "CPU_Inference_ms", "Total_Latency_ms", "FPS"
    ]
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "Index": r["index"],
                "Filename": r["filename"],
                "Ground_Truth_Class": r["gt_name"],
                "Ground_Truth_ID": r["gt_id"],
                "Predicted_Class": r["pred_name"],
                "Predicted_ID": r["pred_id"],
                "Top1_Match": "YES" if r["gt_id"] == r["pred_id"] else "NO",
                "Confidence_Pct": f"{r['confidence_pct']:.1f}",
                "Preprocessing_ms": f"{r['pre_ms']:.2f}",
                "CPU_Inference_ms": f"{r['inf_ms']:.2f}",
                "Total_Latency_ms": f"{r['total_ms']:.2f}",
                "FPS": f"{r['fps']:.1f}",
            })
    print(f" CSV file successfully created: {CSV_OUTPUT}")


def main():
    if not BUILD_BIN.exists():
        print(f"[ERROR] C++ executable not found at {BUILD_BIN}. Please run cmake --build build first.")
        sys.exit(1)

    print("=" * 65)
    print("👓 EDGE VISION ASSISTANT: 100-IMAGE AUTOMATED BENCHMARK SUITE")
    print("=" * 65)

    # 1. Fetch synset mapping
    print("Fetching ImageNet synset dictionary...")
    synset_mapping = download_synset_mapping()

    # 2. Download 100 test images
    image_names = download_100_images()

    # 3. Execute C++ inference on each image
    print("\nRunning C++ CPU inference across all 100 test images...")
    records = []
    correct_count = 0

    pre_times = []
    inf_times = []
    tot_times = []

    for idx, fname in enumerate(image_names, start=1):
        img_path = DATA_100_DIR / fname

        # Extract Ground Truth from filename (e.g. n01514859_hen.JPEG)
        synset = fname.split("_")[0]
        gt_info = synset_mapping.get(synset, (-1, fname.split("_")[1].split(".")[0].title()))
        gt_id, gt_name = gt_info

        # Run inference
        res = run_cpp_inference(img_path)

        is_match = (gt_id == res["pred_id"])
        if is_match:
            correct_count += 1

        pre_times.append(res["pre_ms"])
        inf_times.append(res["inf_ms"])
        tot_times.append(res["total_ms"])

        records.append({
            "index": idx,
            "filename": fname,
            "gt_id": gt_id,
            "gt_name": gt_name,
            "pred_id": res["pred_id"],
            "pred_name": res["pred_name"],
            "confidence_pct": res["confidence_pct"],
            "pre_ms": res["pre_ms"],
            "inf_ms": res["inf_ms"],
            "total_ms": res["total_ms"],
            "fps": res["fps"],
        })

        sys.stdout.write(f"\r  -> Processed {idx}/100 images | Current Accuracy: {correct_count/idx*100:5.1f}%")
        sys.stdout.flush()

    sys.stdout.write("\n")

    # 4. Calculate statistical metrics
    import numpy as np
    tot_arr = np.array(tot_times)
    pre_arr = np.array(pre_times)
    inf_arr = np.array(inf_times)

    summary = {
        "total": len(records),
        "correct": correct_count,
        "accuracy_pct": (correct_count / len(records)) * 100.0,
        "avg_pre_ms": float(np.mean(pre_arr)),
        "avg_inf_ms": float(np.mean(inf_arr)),
        "avg_total_ms": float(np.mean(tot_arr)),
        "avg_fps": float(1000.0 / np.mean(tot_arr)),
        "min_pre": float(np.min(pre_arr)),
        "med_pre": float(np.median(pre_arr)),
        "p90_pre": float(np.percentile(pre_arr, 90)),
        "p95_pre": float(np.percentile(pre_arr, 95)),
        "max_pre": float(np.max(pre_arr)),
        "min_inf": float(np.min(inf_arr)),
        "med_inf": float(np.median(inf_arr)),
        "p90_inf": float(np.percentile(inf_arr, 90)),
        "p95_inf": float(np.percentile(inf_arr, 95)),
        "max_inf": float(np.max(inf_arr)),
        "min_total": float(np.min(tot_arr)),
        "med_total": float(np.median(tot_arr)),
        "p90_total": float(np.percentile(tot_arr, 90)),
        "p95_total": float(np.percentile(tot_arr, 95)),
        "p95_total_ms": float(np.percentile(tot_arr, 95)),
        "max_total": float(np.max(tot_arr)),
    }

    # 5. Export to Excel and CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    create_styled_excel(records, summary)
    export_csv(records)

    print("\n" + "=" * 65)
    print(" BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"  • Total Images Tested    : {summary['total']}")
    print(f"  • Top-1 Match Accuracy   : {summary['accuracy_pct']:.1f}% ({summary['correct']}/{summary['total']})")
    print(f"  • Mean Preprocessing     : {summary['avg_pre_ms']:.2f} ms")
    print(f"  • Mean CPU Inference     : {summary['avg_inf_ms']:.2f} ms (AVX2 Vector SIMD)")
    print(f"  • Mean Total Latency     : {summary['avg_total_ms']:.2f} ms ({summary['avg_fps']:.1f} FPS)")
    print(f"  • P95 Tail Latency       : {summary['p95_total_ms']:.2f} ms")
    print(f"  • Excel Workbook Path    : {EXCEL_OUTPUT}")
    print(f"  • CSV Results Path       : {CSV_OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
