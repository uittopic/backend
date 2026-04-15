"""
Quick Inference Test - Test config mới
Chạy inference trên 50 samples để xem output trước khi train lại
"""
import csv
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

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
device = get_device()
print(f"📱 Device: {device}")
print(f"📦 Load model: {MODEL_PATH}")

processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval()
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
results = []
start_time = time.time()

for i, row in enumerate(test_samples, 1):
    img_path = images_dir / row["image"]
    gt = row.get("caption_vi") or row.get("caption") or ""
    
    if not img_path.exists():
        continue
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    
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

# So sánh với file prediction cũ
print()
print("=" * 60)
print("🔍 SO SÁNH VỚI PREDICTION CŨ")
print("=" * 60)

old_preds = {}
with open("outputs/predictions_test.csv") as f:
    for row in csv.DictReader(f):
        old_preds[row["image"]] = row["caption_full"]

# Lấy 10 mẫu so sánh
for i, r in enumerate(results[:10], 1):
    old = old_preds.get(r["image"], "N/A")
    print(f"\n[{i}] GT:    {r['gt'][:50]}...")
    print(f"    OLD:   {old[:50]}...")
    print(f"    NEW:   {r['pred'][:50]}...")

# Lưu kết quả
output_path = Path("outputs/quick_test_new_config.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "pred_len"])
    writer.writeheader()
    writer.writerows(results)

print()
print(f"💾 Lưu kết quả: {output_path}")