#!/usr/bin/env python3
"""
Script tìm file test.csv đầy đủ (có nhiều hơn 100 dòng)
"""
import os
from pathlib import Path
import csv

# Thư mục cần bỏ qua (cloud storage, trash, etc.)
SKIP_DIRS = {
    "OneDrive",
    "iCloud",
    "Dropbox",
    "Google Drive",
    ".Trash",
    "Library",
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}


def should_skip_path(path: Path) -> bool:
    """Kiểm tra xem có nên bỏ qua path này không"""
    parts = path.parts
    for part in parts:
        if part.startswith(".") and part != ".":
            return True
        if part in SKIP_DIRS:
            return True
        # Bỏ qua cloud storage paths
        if "CloudStorage" in part:
            return True
    return False


def find_full_test_csv():
    """Tìm file test.csv có nhiều hơn 100 dòng"""
    search_paths = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path("/Users/nguyenhuuviet/UIT/Ky3/Chuyen_De"),
    ]
    
    candidates = []
    
    print("🔍 Đang tìm file test.csv đầy đủ...")
    print("=" * 60)
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        if should_skip_path(base_path):
            continue
            
        print(f"📂 Đang quét: {base_path}")
        
        # Tìm tất cả file test.csv (giới hạn độ sâu để tránh timeout)
        try:
            for test_file in base_path.rglob("test.csv"):
                # Bỏ qua nếu path chứa thư mục cần skip
                if should_skip_path(test_file):
                    continue
                
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        count = sum(1 for _ in reader)
                        
                        if count > 100:  # Có nhiều hơn 100 dòng data
                            candidates.append((test_file, count))
                            print(f"✅ Tìm thấy: {test_file}")
                            print(f"   📊 Số ảnh: {count}")
                            print()
                except (PermissionError, TimeoutError, OSError) as e:
                    # Bỏ qua file không đọc được
                    continue
                except Exception as e:
                    continue
        except (PermissionError, TimeoutError, OSError) as e:
            print(f"⚠️  Bỏ qua {base_path}: {type(e).__name__}")
            continue
    
    # Tìm tất cả file test.csv (kể cả nhỏ) để hiển thị
    all_files = []
    for base_path in search_paths:
        if not base_path.exists() or should_skip_path(base_path):
            continue
        try:
            for test_file in base_path.rglob("test.csv"):
                if should_skip_path(test_file):
                    continue
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        count = sum(1 for _ in reader)
                        all_files.append((test_file, count))
                except:
                    continue
        except:
            continue
    
    if all_files:
        print("\n📋 Tất cả file test.csv tìm thấy:")
        for f, count in sorted(all_files, key=lambda x: x[1], reverse=True):
            status = "✅ ĐẦY ĐỦ" if count > 100 else "⚠️  SAMPLE"
            print(f"   {status}: {f} ({count} ảnh)")
        print()
    
    if not candidates:
        print("❌ Không tìm thấy file test.csv đầy đủ (>100 ảnh)")
        print("\n💡 Hướng dẫn:")
        print("   1. Kiểm tra email/file thầy gửi")
        print("   2. Tải lại file test.csv từ Kaggle/Google Drive")
        print("   3. Đảm bảo file có vài trăm đến vài nghìn dòng")
        print("=" * 60)
        return None
    
    # Sắp xếp theo số lượng ảnh (lớn nhất trước)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    print("=" * 60)
    print(f"🎯 File lớn nhất: {candidates[0][0]}")
    print(f"   📊 Số ảnh: {candidates[0][1]}")
    print("=" * 60)
    
    return candidates[0][0]


if __name__ == "__main__":
    result = find_full_test_csv()
    if result:
        print(f"\n🚀 Chạy inference với lệnh:")
        print(f"   PYTHONPATH=$(pwd) python tools/run_inference_shopee_test.py \"{result.parent}\"")

