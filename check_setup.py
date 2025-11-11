#!/usr/bin/env python3
"""
Pre-flight Check Script
Kiểm tra toàn bộ setup trước khi training
"""
import sys
from pathlib import Path

# Colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def check_mark(condition):
    return f"{GREEN}✅{RESET}" if condition else f"{RED}❌{RESET}"

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def check_file(path, description):
    exists = path.exists()
    status = check_mark(exists)
    print(f"{status} {description}: {path}")
    if exists and path.is_file():
        size = path.stat().st_size / (1024 * 1024)  # MB
        print(f"   └─ Kích thước: {size:.2f} MB")
    return exists

def check_dir(path, description):
    exists = path.exists() and path.is_dir()
    status = check_mark(exists)
    print(f"{status} {description}: {path}")
    if exists:
        count = len(list(path.iterdir()))
        print(f"   └─ Số file/thư mục: {count}")
    return exists

def check_python_packages():
    print_header("📦 Kiểm tra Python Packages")
    
    packages = [
        "transformers",
        "torch",
        "torchvision",
        "datasets",
        "pandas",
        "PIL",
        "fastapi",
        "uvicorn"
    ]
    
    all_ok = True
    for pkg in packages:
        try:
            if pkg == "PIL":
                __import__("PIL")
                version = "installed"
            else:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "installed")
            print(f"{GREEN}✅{RESET} {pkg}: {version}")
        except (ImportError, AttributeError, Exception) as e:
            print(f"{RED}❌{RESET} {pkg}: Lỗi - {str(e)[:50]}")
            all_ok = False
    
    return all_ok

def check_device():
    print_header("📱 Kiểm tra Device (GPU/MPS/CPU)")
    
    try:
        import torch
        if torch.backends.mps.is_available():
            print(f"{GREEN}✅{RESET} MPS (Metal) - macOS M1 GPU: Sẵn sàng")
            device = "mps"
        elif torch.cuda.is_available():
            print(f"{GREEN}✅{RESET} CUDA GPU: Sẵn sàng")
            device = "cuda"
        else:
            print(f"{YELLOW}⚠️{RESET} CPU only (sẽ chậm hơn)")
            device = "cpu"
        
        print(f"   └─ Device sẽ sử dụng: {device}")
        return True
    except ImportError:
        print(f"{RED}❌{RESET} PyTorch chưa được cài đặt")
        return False

def main():
    print_header("🔍 PRE-FLIGHT CHECK - BLIP Vietnamese Training")
    
    BASE_DIR = Path(__file__).parent
    all_checks = []
    
    # 1. Check project structure
    print_header("📁 Kiểm tra Cấu trúc Project")
    
    checks = [
        check_dir(BASE_DIR / "app", "Thư mục app/"),
        check_dir(BASE_DIR / "app/api", "Thư mục app/api/"),
        check_dir(BASE_DIR / "app/core", "Thư mục app/core/"),
        check_dir(BASE_DIR / "train", "Thư mục train/"),
        check_dir(BASE_DIR / "data", "Thư mục data/"),
        check_dir(BASE_DIR / "models", "Thư mục models/"),
        check_dir(BASE_DIR / "logs", "Thư mục logs/"),
    ]
    all_checks.append(all(checks))
    
    # 2. Check data files
    print_header("📊 Kiểm tra Dataset")
    
    csv_path = BASE_DIR / "data" / "train_bilingual_clean_v2.csv"
    images_dir = BASE_DIR / "data" / "images"
    
    csv_ok = check_file(csv_path, "File CSV dataset")
    if csv_ok:
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            print(f"   └─ Số dòng: {len(df)}")
            print(f"   └─ Các cột: {', '.join(df.columns.tolist()[:5])}...")
            
            # Check required columns
            required_cols = ["image", "caption_vi"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"{RED}❌{RESET} Thiếu cột: {', '.join(missing_cols)}")
                csv_ok = False
            else:
                print(f"{GREEN}✅{RESET} Có đủ cột cần thiết: image, caption_vi")
        except Exception as e:
            print(f"{RED}❌{RESET} Lỗi đọc CSV: {e}")
            csv_ok = False
    
    images_ok = check_dir(images_dir, "Thư mục images/")
    if images_ok:
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"   └─ Số ảnh: {len(image_files)}")
        if len(image_files) == 0:
            print(f"{YELLOW}⚠️{RESET} Thư mục images/ trống - cần thêm ảnh vào đây")
    
    all_checks.append(csv_ok and images_ok)
    
    # 3. Check model directory
    print_header("🤖 Kiểm tra Model Directory")
    
    model_dir = BASE_DIR / "models" / "blip_vietnamese"
    model_ok = check_dir(model_dir, "Thư mục models/blip_vietnamese/")
    
    if model_ok:
        model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.json"))
        if model_files:
            print(f"{YELLOW}⚠️{RESET} Model đã tồn tại - training sẽ ghi đè")
        else:
            print(f"{GREEN}✅{RESET} Thư mục model trống - sẵn sàng để train")
    
    all_checks.append(model_ok)
    
    # 4. Check Python packages
    packages_ok = check_python_packages()
    all_checks.append(packages_ok)
    
    # 5. Check device
    device_ok = check_device()
    all_checks.append(device_ok)
    
    # 6. Check training script
    print_header("📝 Kiểm tra Training Script")
    
    train_script = BASE_DIR / "train" / "train_blip_vietnamese.py"
    script_ok = check_file(train_script, "File train_blip_vietnamese.py")
    all_checks.append(script_ok)
    
    # 7. Check API files
    print_header("🌐 Kiểm tra API Files")
    
    api_checks = [
        check_file(BASE_DIR / "app" / "main.py", "File app/main.py"),
        check_file(BASE_DIR / "app" / "api" / "routes_caption.py", "File routes_caption.py"),
        check_file(BASE_DIR / "app" / "core" / "model_loader.py", "File model_loader.py"),
        check_file(BASE_DIR / "app" / "core" / "config.py", "File config.py"),
    ]
    all_checks.append(all(api_checks))
    
    # Final summary
    print_header("📋 TÓM TẮT")
    
    total = len(all_checks)
    passed = sum(all_checks)
    
    if all(all_checks):
        print(f"{GREEN}✅ TẤT CẢ KIỂM TRA ĐÃ PASS ({passed}/{total}){RESET}")
        print(f"\n{GREEN}🚀 Sẵn sàng để training!{RESET}")
        print(f"\n{BLUE}Chạy lệnh:{RESET}")
        print(f"  source venv/bin/activate")
        print(f"  cd train")
        print(f"  python train_blip_vietnamese.py")
        return 0
    else:
        print(f"{RED}❌ CÓ {total - passed} KIỂM TRA FAIL ({passed}/{total}){RESET}")
        print(f"\n{YELLOW}⚠️  Vui lòng sửa các lỗi trên trước khi training{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

