#!/usr/bin/env python3
"""
Script chạy inference trên tập test Shopee
- Đọc test.csv và test_images/ từ thư mục shopee-product-matching
- Sinh caption có dấu cho toàn bộ ảnh
- Lưu kết quả thành CSV dạng submit
"""
import csv
import sys
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

from PIL import Image
import torch

from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent
from app.core.config import (
    MAX_NEW_TOKENS,
    NUM_BEAMS,
    EARLY_STOPPING,
    REPETITION_PENALTY,
    NO_REPEAT_NGRAM_SIZE,
    get_device,
    synchronize_device,
    clear_device_cache,
)

BASE_DIR = Path(__file__).parent.parent


def find_shopee_directory() -> Optional[Path]:
    if len(sys.argv) > 1:
        shopee_dir = Path(sys.argv[1])
        if shopee_dir.exists() and (shopee_dir / "test_images").exists():
            return shopee_dir
        print(f"⚠️  Đường dẫn được chỉ định không hợp lệ: {shopee_dir}")

    candidates = [
        BASE_DIR.parent / "shopee-product-matching",
        BASE_DIR.parent.parent / "shopee-product-matching",
        BASE_DIR / "shopee-product-matching",
        Path.home() / "shopee-product-matching",
        Path("/Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/shopee-product-matching"),
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "test_images").exists():
            return candidate
    return None


def resolve_shopee_paths() -> tuple:
    if len(sys.argv) > 1:
        shopee_dir = Path(sys.argv[1])
    else:
        shopee_dir = find_shopee_directory()
    shopee_dir = shopee_dir  # type: ignore
    test_img_dir = (shopee_dir / "test_images") if shopee_dir else None
    test_csv = (shopee_dir / "test.csv") if shopee_dir else None
    return shopee_dir, test_img_dir, test_csv


SHOPEE_DIR, TEST_IMG_DIR, TEST_CSV = resolve_shopee_paths()
OUTPUT_CSV = BASE_DIR / "outputs" / "shopee_test_predictions.csv"

GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": NUM_BEAMS,
    "early_stopping": EARLY_STOPPING,
    "repetition_penalty": REPETITION_PENALTY,
    "length_penalty": 1.1,
}
if NO_REPEAT_NGRAM_SIZE > 0:
    GENERATION_KWARGS["no_repeat_ngram_size"] = NO_REPEAT_NGRAM_SIZE

device = get_device()
print(f"📱 Device: {device}")

model_any: torch.nn.Module = model  # type: ignore
processor_any: Any = processor  # type: ignore


def generate_caption(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor_any(images=image, return_tensors="pt").to(device)  # type: ignore

    if device == "mps":
        model_cpu = model_any.cpu()  # type: ignore
        inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
    else:
        model_cpu = model_any
        inputs_cpu = inputs

    with torch.no_grad():
        output = model_cpu.generate(**inputs_cpu, **GENERATION_KWARGS)  # type: ignore

    if device == "mps":
        model_any.to(device)  # type: ignore
        synchronize_device()
    else:
        synchronize_device()

    if hasattr(processor_any, "tokenizer") and processor_any.tokenizer is not None:
        caption_no_accent = processor_any.tokenizer.decode(output[0], skip_special_tokens=True)  # type: ignore
    else:
        caption_no_accent = processor_any.decode(output[0], skip_special_tokens=True)  # type: ignore

    caption_no_accent = caption_no_accent.strip()
    caption_with_accent = restore_accent(caption_no_accent)

    output = output.detach().cpu()
    del output
    inputs = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in inputs.items()}
    del inputs
    clear_device_cache()

    return caption_no_accent, caption_with_accent


def main():
    print("=" * 60)
    print("🖼️  CHẠY INFERENCE TRÊN TẬP TEST SHOPEE")
    print("=" * 60)

    if SHOPEE_DIR is None or TEST_IMG_DIR is None or not TEST_IMG_DIR.exists():
        print("=" * 60)
        print("❌ KHÔNG TÌM THẤY THƯ MỤC SHOPEE-PRODUCT-MATCHING")
        print("=" * 60)
        print("\n💡 Script đã tìm kiếm ở các vị trí sau nhưng không thấy:")
        print("   1. Cùng cấp với project: ../shopee-product-matching")
        print("   2. Trong thư mục cha: ../../shopee-product-matching")
        print("   3. Trong project: ./shopee-product-matching")
        print("   4. Trong home: ~/shopee-product-matching")
        print("\n🚀 Cách sử dụng:")
        print("   python tools/run_inference_shopee_test.py /path/to/shopee-product-matching")
        print("=" * 60)
        sys.exit(1)

    print(f"📁 Thư mục Shopee: {SHOPEE_DIR}")
    print(f"📂 Test images: {TEST_IMG_DIR}")
    print(f"📄 Test CSV: {TEST_CSV}")
    print(f"💾 Output: {OUTPUT_CSV}")
    print()

    if TEST_CSV is None or not TEST_CSV.exists():
        print("=" * 60)
        print("❌ KHÔNG TÌM THẤY FILE test.csv")
        print("=" * 60)
        print(f"📁 Đã tìm thấy thư mục: {SHOPEE_DIR}")
        print(f"📂 Test images: {TEST_IMG_DIR} {'✅' if TEST_IMG_DIR.exists() else '❌'}")
        print(f"📄 Test CSV: {TEST_CSV} ❌")
        print("=" * 60)
        sys.exit(1)

    print("📖 Đang đọc test.csv...")
    image_list = []
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            if img_name:
                image_list.append(img_name)

    print(f"✅ Đã đọc {len(image_list)} ảnh từ CSV")

    if len(image_list) < 100:
        print(f"\n⚠️  CẢNH BÁO: test.csv chỉ có {len(image_list)} ảnh")
        print("   Đây có thể là bản SAMPLE.\n")
    else:
        print(f"✅ Số lượng ảnh hợp lý ({len(image_list)} ảnh)\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_count = 0
    processed_count = 0

    print("🔄 Đang chạy inference...")
    for img_name in tqdm(image_list, desc="Processing", unit="img"):
        img_path = TEST_IMG_DIR / img_name

        if not img_path.exists():
            print(f"\n⚠️  Không tìm thấy ảnh: {img_name}")
            missing_count += 1
            continue

        try:
            caption_no_ac, caption_full = generate_caption(img_path)
            rows.append({
                "image": img_name,
                "caption_no_accent": caption_no_ac,
                "caption_full": caption_full,
            })
            processed_count += 1
        except Exception as e:
            print(f"\n❌ Lỗi khi xử lý {img_name}: {e}")
            missing_count += 1
            continue

    print(f"\n💾 Đang lưu kết quả vào {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption_no_accent", "caption_full"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ")
    print("=" * 60)
    print(f"📋 Tổng số ảnh trong CSV: {len(image_list)}")
    print(f"✅ Đã xử lý thành công: {processed_count} ảnh")
    if missing_count > 0:
        print(f"⚠️  Bỏ qua/Thiếu/Lỗi: {missing_count} ảnh")
    if processed_count > 0:
        success_rate = (processed_count / len(image_list)) * 100
        print(f"📈 Tỉ lệ thành công: {success_rate:.1f}%")
    print(f"📁 File output: {OUTPUT_CSV}")
    print("=" * 60)
    if processed_count == len(image_list) and processed_count > 0:
        print("🎉 DONE! Đã xử lý 100% tập test!")
    else:
        print("🎉 DONE!")


if __name__ == "__main__":
    main()
