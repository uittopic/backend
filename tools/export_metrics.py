#!/usr/bin/env python3
"""
Xuất metrics ra file JSON từ file CSV của eval_new_model.py

Cách dùng:
  python tools/export_metrics.py
  python tools/export_metrics.py --input outputs/metrics_cleaned_v1.csv
  python tools/export_metrics.py --input outputs/metrics_80_20.csv --output outputs/full_eval/eval_80_20.json
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description="Export metrics CSV → JSON")
parser.add_argument(
    "--input", "-i",
    default=None,
    help="File CSV đầu vào. Mặc định: tự tìm file metrics_*.csv mới nhất trong outputs/",
)
parser.add_argument(
    "--output", "-o",
    default="outputs/metrics.json",
    help="File JSON đầu ra. Default: outputs/metrics.json",
)
parser.add_argument(
    "--batch-json",
    default=None,
    help="File batch_test JSON (tùy chọn, để lấy metadata quality). "
         "Nếu không truyền sẽ bỏ qua phần quality metadata.",
)
_args = parser.parse_args()

# Resolve input CSV
if _args.input:
    METRICS_CSV = BASE / _args.input
else:
    metrics_files = sorted((BASE / "outputs").glob("metrics_*.csv"), reverse=True)
    if metrics_files:
        METRICS_CSV = metrics_files[0]
        print(f"ℹ️  Tự tìm thấy: {METRICS_CSV.name}")
    else:
        print("❌ Không tìm thấy file metrics_*.csv trong outputs/")
        print("   Chạy trước: python tools/eval_new_model.py --model-path models/blip_vietnamese_cleaned_v1")
        exit(1)

BATCH_TEST_JSON = Path(_args.batch_json) if _args.batch_json else None
OUTPUT_JSON = (BASE / _args.output) if _args.output else BASE / "outputs" / "metrics.json"

print("📊 Đang xuất metrics ra JSON...")
print(f"   Input:  {METRICS_CSV}")
print(f"   Output: {OUTPUT_JSON}")

# Đọc batch test metadata (tùy chọn)
meta = {}
if BATCH_TEST_JSON and BATCH_TEST_JSON.exists():
    with open(BATCH_TEST_JSON) as f:
        batch_data = json.load(f)
    meta = batch_data.get("metadata", {}).get("stats", {})

# Đọc metrics BLEU — hỗ trợ 2 format CSV:
#   Format mới (eval_new_model.py): bleux_no_accent, bleux_with_accent, caption_with_accent
#   Format cũ (eval_detailed.py):     bleu1, bleu2, bleu3, bleu4, bleu, prediction
bleu1_list, bleu2_list, bleu3_list, bleu4_list, bleu_list = [], [], [], [], []
samples = []
with open(METRICS_CSV) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    is_new_format = "bleu1_no_accent" in headers or "bleu_no_accent" in headers

    for row in reader:
        if is_new_format:
            # Format mới: eval_new_model.py
            # Pipeline: BLIP → Accent Restoration → output có dấu → ưu tiên with_accent
            bleu1 = float(row.get("bleu1_with_accent") or row.get("bleu1_no_accent") or 0)
            bleu2 = float(row.get("bleu2_with_accent") or row.get("bleu2_no_accent") or 0)
            bleu3 = float(row.get("bleu3_with_accent") or row.get("bleu3_no_accent") or 0)
            bleu4 = float(row.get("bleu4_with_accent") or row.get("bleu4_no_accent") or 0)
            bleu = float(row.get("bleu_with_accent") or row.get("bleu_no_accent") or 0)
            prediction = row.get("caption_with_accent") or row.get("caption_no_accent") or ""
        else:
            # Format cũ: eval_detailed.py
            bleu1 = float(row.get("bleu1") or 0)
            bleu2 = float(row.get("bleu2") or 0)
            bleu3 = float(row.get("bleu3") or 0)
            bleu4 = float(row.get("bleu4") or 0)
            bleu = float(row.get("bleu") or 0)
            prediction = row.get("prediction") or ""

        bleu1_list.append(bleu1)
        bleu2_list.append(bleu2)
        bleu3_list.append(bleu3)
        bleu4_list.append(bleu4)
        bleu_list.append(bleu)
        samples.append({
            "image": row.get("image", ""),
            "ground_truth": row.get("ground_truth", ""),
            "prediction": prediction,
            "bleu": bleu,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
        })

n = len(bleu_list)

# Tính BLEU trung bình
avg_bleu = {
    "bleu1": round(sum(bleu1_list) / n, 4),
    "bleu2": round(sum(bleu2_list) / n, 4),
    "bleu3": round(sum(bleu3_list) / n, 4),
    "bleu4": round(sum(bleu4_list) / n, 4),
    "bleu": round(sum(bleu_list) / n, 4)
}

# BLEU max, min
bleu_max = round(max(bleu_list), 4)
bleu_min = round(min(bleu_list), 4)

# Đếm samples theo BLEU ranges
bleu_ranges = {
    "bleu_0": sum(1 for b in bleu_list if b == 0),
    "bleu_0_01": sum(1 for b in bleu_list if 0 < b <= 0.01),
    "bleu_01_02": sum(1 for b in bleu_list if 0.01 < b <= 0.02),
    "bleu_02_05": sum(1 for b in bleu_list if 0.02 < b <= 0.05),
    "bleu_05_1": sum(1 for b in bleu_list if 0.05 < b <= 0.1),
    "bleu_1_plus": sum(1 for b in bleu_list if b > 0.1)
}

# Đọc processing time từ batch test (tùy chọn)
processing_times = []
avg_processing_time = 0
if BATCH_TEST_JSON and BATCH_TEST_JSON.exists():
    with open(BATCH_TEST_JSON) as f:
        batch_data = json.load(f)
    processing_times = [r["processing_time"] for r in batch_data["results"] if r.get("success")]
    avg_processing_time = round(sum(processing_times) / len(processing_times), 2) if processing_times else 0

# Build JSON output
batch_section = {}
if meta:
    batch_section = {
        "total_images": meta.get("total_images", 0),
        "success": meta.get("success", 0),
        "failed": meta.get("failed", 0),
        "error_count": meta.get("error_count", 0),
        "success_rate": meta.get("success_rate", 0),
        "quality": {
            "good_captions": meta.get("good_captions", 0),
            "medium_captions": meta.get("medium_captions", 0),
            "bad_captions": meta.get("bad_captions", 0),
            "no_caption": meta.get("no_caption", 0),
        },
        "processing_time": {
            "avg_seconds_per_image": avg_processing_time,
            "total_seconds": round(sum(processing_times), 2) if processing_times else 0,
            "unit": "seconds",
        },
    }

output = {
    "metadata": {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "export_metrics.py",
        "sources": {
            "metrics_csv": str(METRICS_CSV.name),
            "batch_test": str(BATCH_TEST_JSON.name) if BATCH_TEST_JSON else None,
        }
    },
    "batch_test": batch_section,
    "bleu_scores": {
        "average": avg_bleu,
        "max": bleu_max,
        "min": bleu_min,
        "sample_count": n,
        "distribution": bleu_ranges,
        "distribution_percentages": {
            k: round(v / n * 100, 2) for k, v in bleu_ranges.items()
        }
    },
    "samples": samples[:100]  # Chỉ lưu 100 samples đầu
}

# Ghi file JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Đã xuất metrics ra: {OUTPUT_JSON}")
print(f"\n📊 Tổng quan BLEU:")
print(f"   BLEU-1: {avg_bleu['bleu1']} ({avg_bleu['bleu1']*100:.2f}%)")
print(f"   BLEU-2: {avg_bleu['bleu2']} ({avg_bleu['bleu2']*100:.2f}%)")
print(f"   BLEU-3: {avg_bleu['bleu3']} ({avg_bleu['bleu3']*100:.2f}%)")
print(f"   BLEU-4: {avg_bleu['bleu4']} ({avg_bleu['bleu4']*100:.2f}%)")
print(f"   BLEU:   {avg_bleu['bleu']} ({avg_bleu['bleu']*100:.2f}%)")
if meta:
    total = meta.get("total_images") or 1
    print(f"\n📦 Quality:")
    print(f"   Good:   {meta.get('good_captions', 0)} ({meta.get('good_captions', 0)*100/total:.1f}%)")
    print(f"   Medium: {meta.get('medium_captions', 0)} ({meta.get('medium_captions', 0)*100/total:.1f}%)")
    print(f"   Bad:    {meta.get('bad_captions', 0)} ({meta.get('bad_captions', 0)*100/total:.2f}%)")
else:
    print(f"\n📦 (Quality metadata không có — chạy với --batch-json để xem thêm)")
