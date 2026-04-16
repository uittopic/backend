#!/usr/bin/env python3
"""
Xuất metrics ra file JSON
"""

import json
import csv
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
BATCH_TEST_JSON = BASE / "outputs" / "batch_test" / "batch_test_results_20260415_125950.json"
METRICS_CSV = BASE / "outputs" / "metrics_full_test_fixed.csv"
OUTPUT_JSON = BASE / "outputs" / "metrics.json"

print("📊 Đang xuất metrics ra JSON...")

# Đọc batch test metadata
with open(BATCH_TEST_JSON) as f:
    batch_data = json.load(f)

meta = batch_data["metadata"]["stats"]

# Đọc metrics BLEU
bleu1_list, bleu2_list, bleu3_list, bleu4_list, bleu_list = [], [], [], [], []
samples = []
with open(METRICS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        bleu1_list.append(float(row['bleu1']))
        bleu2_list.append(float(row['bleu2']))
        bleu3_list.append(float(row['bleu3']))
        bleu4_list.append(float(row['bleu4']))
        bleu_list.append(float(row['bleu']))
        samples.append({
            "image": row['image'],
            "ground_truth": row['ground_truth'],
            "prediction": row['prediction'],
            "bleu": float(row['bleu']),
            "bleu1": float(row['bleu1']),
            "bleu2": float(row['bleu2']),
            "bleu3": float(row['bleu3']),
            "bleu4": float(row['bleu4'])
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

# Đọc processing time từ batch test
processing_times = [r["processing_time"] for r in batch_data["results"] if r.get("success")]
avg_processing_time = round(sum(processing_times) / len(processing_times), 2) if processing_times else 0

# Build JSON output
output = {
    "metadata": {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "export_metrics.py",
        "sources": {
            "batch_test": str(BATCH_TEST_JSON.name),
            "metrics_csv": str(METRICS_CSV.name)
        }
    },
    "batch_test": {
        "total_images": meta["total_images"],
        "success": meta["success"],
        "failed": meta["failed"],
        "error_count": meta["error_count"],
        "success_rate": meta["success_rate"],
        "quality": {
            "good_captions": meta["good_captions"],
            "medium_captions": meta["medium_captions"],
            "bad_captions": meta["bad_captions"],
            "no_caption": meta["no_caption"]
        },
        "processing_time": {
            "avg_seconds_per_image": avg_processing_time,
            "total_seconds": round(sum(processing_times), 2),
            "unit": "seconds"
        }
    },
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
print(f"\n📦 Quality:")
print(f"   Good: {meta['good_captions']} ({meta['good_captions']*100/meta['total_images']:.1f}%)")
print(f"   Medium: {meta['medium_captions']} ({meta['medium_captions']*100/meta['total_images']:.1f}%)")
print(f"   Bad: {meta['bad_captions']} ({meta['bad_captions']*100/meta['total_images']:.2f}%)")
