#!/usr/bin/env python3
"""
Tool tự động copy ảnh từ thư mục nguồn vào data/images/ dựa trên CSV
Sử dụng khi bạn đã có ảnh ở một thư mục khác và muốn copy vào project
"""
import os
import sys
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "data" / "train_bilingual_clean_v2.csv"
TARGET_DIR = BASE_DIR / "data" / "images"

# Thư mục nguồn - Có thể nhập từ command line hoặc sửa ở đây
if len(sys.argv) > 1:
    SOURCE_DIR = Path(sys.argv[1])
else:
    # Mặc định - THAY ĐỔI ĐƯỜNG DẪN NÀY nếu cần
    SOURCE_DIR = Path("/Users/viet/Datasets/Shopee_images")  # ⚠️ SỬA ĐƯỜNG DẪN NÀY

# Extensions ảnh được hỗ trợ
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

def find_image_file(base_path, image_name):
    """
    Tìm file ảnh với tên đã cho, thử nhiều extension
    """
    # Thử tên chính xác
    if base_path.exists():
        return base_path
    
    # Thử các extension khác nhau
    name_without_ext = base_path.stem
    for ext in IMAGE_EXTENSIONS:
        test_path = base_path.parent / f"{name_without_ext}{ext}"
        if test_path.exists():
            return test_path
    
    return None

def main():
    print("=" * 60)
    print("🖼️  Tool Copy Ảnh Tự Động Theo CSV")
    print("=" * 60)
    
    # Kiểm tra CSV
    if not CSV_PATH.exists():
        print(f"❌ Không tìm thấy file CSV: {CSV_PATH}")
        sys.exit(1)
    
    # Kiểm tra thư mục nguồn
    if not SOURCE_DIR.exists():
        print(f"❌ Không tìm thấy thư mục nguồn: {SOURCE_DIR}")
        print(f"\n💡 Vui lòng sửa đường dẫn SOURCE_DIR trong file này:")
        print(f"   {__file__}")
        sys.exit(1)
    
    # Tạo thư mục đích
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Thư mục đích: {TARGET_DIR}")
    
    # Đọc CSV
    print(f"\n📂 Đang đọc CSV: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Đã load {len(df)} dòng từ CSV")
    except Exception as e:
        print(f"❌ Lỗi đọc CSV: {e}")
        sys.exit(1)
    
    # Kiểm tra cột image
    if "image" not in df.columns:
        print("❌ CSV không có cột 'image'")
        sys.exit(1)
    
    # Lấy danh sách ảnh unique
    image_names = df["image"].unique()
    total_images = len(image_names)
    print(f"📊 Tổng số ảnh cần copy: {total_images}")
    print(f"📁 Thư mục nguồn: {SOURCE_DIR}")
    print(f"📁 Thư mục đích: {TARGET_DIR}\n")
    
    # Copy ảnh
    copied = 0
    missing = 0
    skipped = 0  # Đã tồn tại
    errors = []
    
    print("🔄 Đang copy ảnh...")
    for img_name in tqdm(image_names, desc="Copying", unit="img"):
        # Đường dẫn nguồn
        src_path = SOURCE_DIR / img_name
        
        # Tìm file (thử nhiều extension)
        found_path = find_image_file(src_path, img_name)
        
        if found_path is None:
            missing += 1
            errors.append(f"Không tìm thấy: {img_name}")
            continue
        
        # Đường dẫn đích
        dst_path = TARGET_DIR / img_name
        
        # Kiểm tra đã tồn tại chưa
        if dst_path.exists():
            skipped += 1
            continue
        
        # Copy file
        try:
            shutil.copy2(found_path, dst_path)
            copied += 1
        except Exception as e:
            missing += 1
            errors.append(f"Lỗi copy {img_name}: {e}")
    
    # Kết quả
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ")
    print("=" * 60)
    print(f"✅ Đã copy: {copied} ảnh")
    print(f"⏭️  Đã bỏ qua (tồn tại): {skipped} ảnh")
    print(f"❌ Thiếu/Không tìm thấy: {missing} ảnh")
    print(f"📈 Tổng: {total_images} ảnh")
    
    if missing > 0:
        print(f"\n⚠️  Có {missing} ảnh không tìm thấy hoặc lỗi")
        if len(errors) <= 10:
            print("\nChi tiết:")
            for err in errors[:10]:
                print(f"  - {err}")
        else:
            print(f"\n(Hiển thị 10 lỗi đầu tiên trong tổng {len(errors)} lỗi)")
    
    # Kiểm tra tổng số ảnh trong thư mục đích
    final_count = len(list(TARGET_DIR.glob("*.*")))
    print(f"\n📁 Tổng số ảnh trong {TARGET_DIR}: {final_count}")
    
    if copied + skipped == total_images:
        print("\n🎉 Hoàn thành! Tất cả ảnh đã được copy.")
    elif missing > 0:
        print(f"\n⚠️  Còn thiếu {missing} ảnh. Vui lòng kiểm tra lại thư mục nguồn.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

