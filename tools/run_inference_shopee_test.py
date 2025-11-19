#!/usr/bin/env python3
"""
Script chạy inference trên tập test Shopee
- Đọc test.csv và test_images/ từ thư mục shopee-product-matching
- Sinh caption có dấu cho toàn bộ ảnh
- Lưu kết quả thành CSV dạng submit
"""
import os
import csv
import sys
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import torch

# Import từ app
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

# -----------------------------
# 1. ĐƯỜNG DẪN TẬP TEST SHOPEE
# -----------------------------
BASE_DIR = Path(__file__).parent.parent


def find_shopee_directory() -> Path:
    """
    Tìm thư mục shopee-product-matching ở các vị trí có thể:
    1. Từ command line argument
    2. Cùng cấp với project (mặc định)
    3. Trong thư mục cha (Chuyen_De)
    4. Trong thư mục hiện tại
    """
    # 1. Từ command line argument
    if len(sys.argv) > 1:
        shopee_dir = Path(sys.argv[1])
        if shopee_dir.exists() and (shopee_dir / "test_images").exists():
            return shopee_dir
        else:
            print(f"⚠️  Đường dẫn được chỉ định không hợp lệ: {shopee_dir}")
    
    # 2. Cùng cấp với project (mặc định)
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
    
    # Không tìm thấy
    return None


# Tìm thư mục Shopee
if len(sys.argv) > 1:
    SHOPEE_DIR = Path(sys.argv[1])
else:
    SHOPEE_DIR = find_shopee_directory()

TEST_IMG_DIR = SHOPEE_DIR / "test_images" if SHOPEE_DIR else None
TEST_CSV = SHOPEE_DIR / "test.csv" if SHOPEE_DIR else None
OUTPUT_CSV = BASE_DIR / "outputs" / "shopee_test_predictions.csv"

# -----------------------------
# 2. GENERATION CONFIG
# -----------------------------
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

# -----------------------------
# 3. HÀM SINH CAPTION
# -----------------------------
def generate_caption(image_path: Path):
    """
    Sinh caption cho một ảnh:
    - BLIP: caption tiếng Việt không dấu
    - Accent Restoration: thêm dấu tiếng Việt
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, **GENERATION_KWARGS)
        synchronize_device()

    # Decode caption không dấu
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        caption_no_accent = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        caption_no_accent = processor.decode(output[0], skip_special_tokens=True)
    
    caption_no_accent = caption_no_accent.strip()

    # Restore accent
    caption_with_accent = restore_accent(caption_no_accent)

    # Cleanup tensors
    output = output.detach().cpu()
    del output
    inputs = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in inputs.items()}
    del inputs
    clear_device_cache()

    return caption_no_accent, caption_with_accent


# -----------------------------
# 4. HÀM MAIN
# -----------------------------
def main():
    print("=" * 60)
    print("🖼️  CHẠY INFERENCE TRÊN TẬP TEST SHOPEE")
    print("=" * 60)
    
    # Kiểm tra thư mục và file TRƯỚC KHI in thông tin
    if SHOPEE_DIR is None or not TEST_IMG_DIR or not TEST_IMG_DIR.exists():
        print("=" * 60)
        print("❌ KHÔNG TÌM THẤY THƯ MỤC SHOPEE-PRODUCT-MATCHING")
        print("=" * 60)
        print("\n💡 Script đã tìm kiếm ở các vị trí sau nhưng không thấy:")
        print("   1. Cùng cấp với project: ../shopee-product-matching")
        print("   2. Trong thư mục cha: ../../shopee-product-matching")
        print("   3. Trong project: ./shopee-product-matching")
        print("   4. Trong home: ~/shopee-product-matching")
        print("\n📋 Cấu trúc thư mục cần có:")
        print("   shopee-product-matching/")
        print("   ├── test_images/     # Thư mục chứa ảnh test")
        print("   └── test.csv          # File CSV chứa danh sách ảnh (cột 'image')")
        print("\n🚀 Cách sử dụng:")
        print("   1. Tạo thư mục shopee-product-matching với cấu trúc trên")
        print("   2. Hoặc chỉ định đường dẫn qua argument:")
        print("      python tools/run_inference_shopee_test.py /path/to/shopee-product-matching")
        print("\n📝 Ví dụ:")
        print("   python tools/run_inference_shopee_test.py ~/Downloads/shopee-product-matching")
        print("=" * 60)
        sys.exit(1)

    # In thông tin sau khi đã xác nhận thư mục tồn tại
    print(f"📁 Thư mục Shopee: {SHOPEE_DIR}")
    print(f"📂 Test images: {TEST_IMG_DIR}")
    print(f"📄 Test CSV: {TEST_CSV}")
    print(f"💾 Output: {OUTPUT_CSV}")
    print()

    if not TEST_CSV.exists():
        print("=" * 60)
        print("❌ KHÔNG TÌM THẤY FILE test.csv")
        print("=" * 60)
        print(f"📁 Đã tìm thấy thư mục: {SHOPEE_DIR}")
        print(f"📂 Test images: {TEST_IMG_DIR} {'✅' if TEST_IMG_DIR.exists() else '❌'}")
        print(f"📄 Test CSV: {TEST_CSV} ❌")
        print("\n💡 Hãy đảm bảo file test.csv có trong thư mục shopee-product-matching/")
        print("   Format CSV cần có cột 'image' chứa tên file ảnh")
        print("=" * 60)
        sys.exit(1)

    # Đọc danh sách ảnh từ CSV
    print("📖 Đang đọc test.csv...")
    image_list = []
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            if img_name:
                image_list.append(img_name)

    print(f"✅ Đã đọc {len(image_list)} ảnh từ CSV")
    
    # Cảnh báo nếu test.csv quá nhỏ (có thể là sample)
    if len(image_list) < 100:
        print(f"\n⚠️  CẢNH BÁO: test.csv chỉ có {len(image_list)} ảnh")
        print("   Đây có thể là bản SAMPLE, không phải tập test đầy đủ!")
        print("   Tập test Shopee thường có vài trăm đến vài nghìn ảnh.")
        print("   Hãy kiểm tra lại file test.csv từ thầy.\n")
    else:
        print(f"✅ Số lượng ảnh hợp lý ({len(image_list)} ảnh) - có thể là tập test đầy đủ\n")

    # Tạo thư mục output nếu chưa có
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Chạy inference
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

    # Lưu kết quả
    print(f"\n💾 Đang lưu kết quả vào {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "caption_no_accent", "caption_full"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Thống kê
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

