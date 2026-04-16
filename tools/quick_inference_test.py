"""
Quick Inference Test - Test config mới
Chạy inference trên 50 samples để xem output trước khi train lại
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Any, Dict

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import (
    MODEL_PATH, NUM_BEAMS, MAX_NEW_TOKENS,
    REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE, get_device
)

# Config mới (từ app/core/config.py)
print("=" * 60)
print("⚙️  CONFIG MỚI (đã tối ưu)")
print("=" * 60)
print(f"  NUM_BEAMS:          {NUM_BEAMS} (trước: 5)")
print(f"  REPETITION_PENALTY: {REPETITION_PENALTY} (trước: 1.2)")
print(f"  MAX_NEW_TOKENS:     {MAX_NEW_TOKENS} (trước: 50)")
print()

# Load model
device_str: str = get_device()
device: torch.device = torch.device(device_str)
print(f"📱 Device: {device}")
print(f"📦 Load model: {MODEL_PATH}")

blip_processor: BlipProcessor = BlipProcessor.from_pretrained(str(MODEL_PATH))  # type: ignore[assignment]
blip_model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))  # type: ignore[assignment]
blip_model.to(device)  # type: ignore[arg-type]
blip_model.eval()
print("✅ Model loaded")

# Load test data (50 samples)
test_csv = Path("data/test_20.csv")
images_dir = Path("data/images")

with open(test_csv) as f:
    reader = csv.DictReader(f)
    test_samples = list(reader)[:50]

print(f"🧪 Test inference trên {len(test_samples)} samples...")
print()

# Run inference
results: list[Dict[str, Any]] = []
start_time = time.time()

for i, row in enumerate(test_samples, 1):
    img_path = images_dir / row["image"]
    gt = row.get("caption_vi") or row.get("caption") or ""

    if not img_path.exists():
        continue

    image = Image.open(img_path).convert("RGB")

    # Xử lý inputs với processor
    raw_inputs = blip_processor(images=image, return_tensors="pt")

    # Ép kiểu an toàn - BatchEncoding có thể access như dict
    if isinstance(raw_inputs, tuple):
        # Tuple (processor, extra_dict) - lấy phần tử thứ 2 là dict
        inputs: Dict[str, Any] = raw_inputs[1] if len(raw_inputs) > 1 else dict(raw_inputs[0])  # type: ignore[index]
    else:
        # BatchEncoding có thể convert sang dict
        inputs = dict(raw_inputs)  # type: ignore[arg-type]

    # Chuyển inputs sang device
    inputs_device: Dict[str, Any] = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs_device[k] = v.to(device)  # type: ignore[arg-type]
        else:
            inputs_device[k] = v

    with torch.no_grad():
        output = blip_model.generate(
            **inputs_device,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )

    # Decode caption
    caption: str = blip_processor.batch_decode(output, skip_special_tokens=True)[0]  # type: ignore[union-attr]

    results.append({
        "image": row["image"],
        "gt": gt,
        "pred": caption,
        "pred_len": len(caption.split())
    })

    if i % 10 == 0:
        print(f"  [{i}/{len(test_samples)}] ...")

elapsed = time.time() - start_time

print()
print("=" * 60)
print(f"✅ Hoàn thành {len(results)} samples trong {elapsed:.1f}s")
print()

# Thống kê
avg_len = sum(r["pred_len"] for r in results) / len(results)
print("=" * 60)
print("📊 THỐNG KÊ SAMPLE MỚI")
print("=" * 60)
print(f"  Số samples: {len(results)}")
print(f"  Avg words/sample: {avg_len:.1f}")
print()

# Hiển thị 10 samples
print("=" * 60)
print("📝 SAMPLE OUTPUT MỚI (10 samples)")
print("=" * 60)
for i, r in enumerate(results[:10], 1):
    print(f"\n[{i}] GT:   {r['gt'][:65]}...")
    print(f"    Pred:  {r['pred'][:65]}...")
    print(f"    Words: {r['pred_len']}")

# So sánh với file prediction cũ (nếu có)
print()
print("=" * 60)
print("🔍 SO SÁNH VỚI PREDICTION CŨ")
print("=" * 60)

old_preds_path = Path("outputs/predictions_test.csv")
if old_preds_path.exists():
    old_preds = {}
    with open(old_preds_path) as f:
        for row in csv.DictReader(f):
            old_preds[row["image"]] = row["caption_full"]

    # Lấy 10 mẫu so sánh
    for i, r in enumerate(results[:10], 1):
        old = old_preds.get(r["image"], "N/A")
        print(f"\n[{i}] GT:    {r['gt'][:50]}...")
        print(f"    OLD:   {old[:50]}...")
        print(f"    NEW:   {r['pred'][:50]}...")
else:
    print("  (Không tìm thấy file predictions cũ để so sánh)")

# Lưu kết quả
output_path = Path("outputs/quick_test_new_config.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "pred_len"])
    writer.writeheader()
    writer.writerows(results)

print()
print(f"💾 Lưu kết quả: {output_path}")
