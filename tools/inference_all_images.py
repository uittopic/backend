#!/usr/bin/env python3
"""
Inference Full Images Directory
================================
Chạy inference trên TẤT CẢ ảnh trong data/images/ mà không cần file CSV.

Usage:
    python tools/inference_all_images.py                    # Full 7444 ảnh
    python tools/inference_all_images.py --limit 50         # 50 ảnh đầu
    python tools/inference_all_images.py --random --limit 50 # 50 ảnh ngẫu nhiên
    python tools/inference_all_images.py --output results.csv  # Đổi tên file output
"""

import os
import sys
import argparse
import csv
import json
import time
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Tự động thêm project root vào sys.path
BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from PIL import Image
import torch

from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent
from app.core.config import (
    MODEL_PATH, IMAGES_DIR, MAX_NEW_TOKENS, NUM_BEAMS,
    REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE, LENGTH_PENALTY
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_caption(image_path: Path, device: torch.device):
    """Sinh caption có dấu tiếng Việt cho một ảnh."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Chuyển sang device (MPS cần CPU helper)
        if device.type == "mps":
            model_cpu = model.cpu()
            inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
        else:
            model_cpu = model
            inputs_cpu = inputs
            if device.type == "cuda":
                inputs_cpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Sinh caption không dấu
        with torch.no_grad():
            output = model_cpu.generate(
                **inputs_cpu,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
            )

        caption_no_accent = processor.decode(output[0], skip_special_tokens=True)

        # Khôi phục dấu tiếng Việt
        caption_with_accent = restore_accent(caption_no_accent)

        return {
            "success": True,
            "caption_no_accent": caption_no_accent,
            "caption_with_accent": caption_with_accent,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "caption_no_accent": None,
            "caption_with_accent": None,
            "error": str(e),
        }


def get_all_images(images_dir: Path) -> list:
    """Lấy danh sách tất cả ảnh trong thư mục."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    images = []
    for ext in extensions:
        images.extend(images_dir.glob(f"*{ext}"))
        images.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(description="Inference all images in data/images/")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR,
                        help=f"Thư mục ảnh (default: {IMAGES_DIR})")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("outputs/all_images_predictions.csv"),
                        help="File CSV output")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Giới hạn số ảnh (mặc định: full)")
    parser.add_argument("--random", "-r", action="store_true",
                        help="Chọn ngẫu nhiên thay vì theo thứ tự")
    parser.add_argument("--json", action="store_true",
                        help="Lưu thêm file JSON chi tiết")
    args = parser.parse_args()

    device = get_device()
    print(f"\n{'='*60}")
    print(f"INFERENCE FULL IMAGES DIRECTORY")
    print(f"{'='*60}")
    print(f"📁 Images dir: {args.images_dir}")
    print(f"📱 Device:     {device}")
    print(f"📝 Model:     {MODEL_PATH}")
    print()

    # Lấy danh sách ảnh
    all_images = get_all_images(args.images_dir)
    if not all_images:
        print("❌ Không tìm thấy ảnh nào!")
        return

    print(f"✅ Tìm thấy {len(all_images)} ảnh")

    images_to_process = all_images.copy()
    if args.random:
        random.seed(42)
        random.shuffle(images_to_process)
        print("🎲 Chế độ: Ngẫu nhiên")

    if args.limit:
        images_to_process = images_to_process[:args.limit]
        print(f"📊 Giới hạn: {args.limit} ảnh")

    print(f"\n🚀 Bắt đầu inference {len(images_to_process)} ảnh...")
    print()

    results = []
    start_time = time.time()
    errors = []

    for img_path in tqdm(images_to_process, desc="Inferencing", unit="img"):
        infer_start = time.time()
        result = generate_caption(img_path, device)
        infer_time = time.time() - infer_start

        results.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "caption_with_accent": result["caption_with_accent"],
            "caption_no_accent": result["caption_no_accent"],
            "success": result["success"],
            "error": result["error"],
            "infer_time_sec": round(infer_time, 3),
        })

        if not result["success"]:
            errors.append({"image": img_path.name, "error": result["error"]})

    total_time = time.time() - start_time

    # Thống kê
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success

    print()
    print(f"{'='*60}")
    print(f"KẾT QUẢ")
    print(f"{'='*60}")
    print(f"📊 Tổng ảnh:        {total}")
    print(f"✅ Thành công:      {success} ({success/total*100:.1f}%)")
    print(f"❌ Thất bại:        {failed} ({failed/total*100:.1f}%)")
    print(f"⏱️  Tổng thời gian:  {total_time/60:.1f} phút")
    print(f"⏱️  TB/ảnh:          {total_time/total:.2f} giây")

    # Thống kê độ dài caption
    captions_with_accent = [r["caption_with_accent"] for r in results if r["caption_with_accent"]]
    if captions_with_accent:
        lengths = [len(c.split()) for c in captions_with_accent]
        avg_len = sum(lengths) / len(lengths)
        print(f"📏 TB từ/caption:   {avg_len:.1f}")

    # Sample kết quả
    print()
    print(f"📝 MẪU KẾT QUẢ:")
    for r in results[:5]:
        status = "✅" if r["success"] else "❌"
        caption = r.get("caption_with_accent", r.get("error", "N/A"))
        print(f"  {status} {r['image_name']}: {caption[:70]}")

    if errors:
        print(f"\n❌ LỖI ({len(errors)} ảnh):")
        for e in errors[:5]:
            print(f"  - {e['image']}: {e['error']}")

    # Lưu CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "caption_with_accent", "caption_no_accent",
            "success", "infer_time_sec", "error"
        ])
        for r in results:
            writer.writerow([
                r["image_name"],
                r.get("caption_with_accent", ""),
                r.get("caption_no_accent", ""),
                r["success"],
                r["infer_time_sec"],
                r.get("error", ""),
            ])

    print(f"\n💾 CSV saved: {args.output}")

    # Lưu JSON chi tiết
    if args.json:
        json_path = args.output.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "images_dir": str(args.images_dir),
                    "total_images": total,
                    "success": success,
                    "failed": failed,
                    "device": str(device),
                    "model_path": str(MODEL_PATH),
                    "total_time_min": round(total_time / 60, 2),
                    "avg_time_per_image_sec": round(total_time / total, 3),
                    "timestamp": datetime.now().isoformat(),
                },
                "results": results,
                "errors": errors,
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON saved: {json_path}")

    print(f"\n✅ Xong!")


if __name__ == "__main__":
    main()
