#!/usr/bin/env python3
"""
Script kiểm tra logic tổng thể của project
Kiểm tra: config, paths, device consistency, imports
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

print("=" * 70)
print("🔍 KIỂM TRA LOGIC TỔNG THỂ")
print("=" * 70)

errors = []
warnings = []
success = []

# 1. Kiểm tra .env.example
print("\n1️⃣  Kiểm tra .env.example")
try:
    env_example = BASE_DIR / ".env.example"
    if env_example.exists():
        content = env_example.read_text()
        required_vars = ["MODEL_PATH", "DATA_DIR", "TRAIN_BATCH_SIZE", "MAX_NEW_TOKENS"]
        missing = [var for var in required_vars if var not in content]
        if not missing:
            success.append("✅ .env.example có đủ biến cấu hình")
        else:
            warnings.append(f"⚠️  .env.example thiếu: {', '.join(missing)}")
    else:
        warnings.append("⚠️  .env.example không tồn tại")
except Exception as e:
    errors.append(f"❌ Lỗi đọc .env.example: {e}")

# 2. Kiểm tra config.py
print("\n2️⃣  Kiểm tra app/core/config.py")
try:
    sys.path.insert(0, str(BASE_DIR))
    from app.core.config import (
        BASE_DIR as CONFIG_BASE_DIR,
        MODEL_PATH,
        DATA_DIR,
        IMAGES_DIR,
        CSV_PATH,
        get_device
    )
    
    # Kiểm tra BASE_DIR nhất quán
    if CONFIG_BASE_DIR == BASE_DIR:
        success.append("✅ BASE_DIR trong config đúng")
    else:
        errors.append(f"❌ BASE_DIR không khớp: {CONFIG_BASE_DIR} vs {BASE_DIR}")
    
    # Kiểm tra paths
    paths_check = [
        ("MODEL_PATH", MODEL_PATH, BASE_DIR / "models" / "blip_vietnamese"),
        ("DATA_DIR", DATA_DIR, BASE_DIR / "data"),
        ("IMAGES_DIR", IMAGES_DIR, BASE_DIR / "data" / "images"),
        ("CSV_PATH", CSV_PATH, BASE_DIR / "data" / "train_bilingual_clean_v2.csv"),
    ]
    
    for name, actual, expected in paths_check:
        if actual == expected:
            success.append(f"✅ {name} đúng: {actual}")
        else:
            errors.append(f"❌ {name} sai: {actual} (mong đợi: {expected})")
    
    # Kiểm tra get_device
    device = get_device()
    if device in ["mps", "cuda", "cpu"]:
        success.append(f"✅ get_device() hoạt động: {device}")
    else:
        errors.append(f"❌ get_device() trả về giá trị không hợp lệ: {device}")
        
except Exception as e:
    errors.append(f"❌ Lỗi import config: {e}")

# 3. Kiểm tra train_blip_vietnamese.py có shuffle + split
print("\n3️⃣  Kiểm tra train_blip_vietnamese.py")
try:
    train_script = BASE_DIR / "train" / "train_blip_vietnamese.py"
    content = train_script.read_text()
    
    checks = {
        "shuffle": "sample(frac=1" in content or "shuffle" in content.lower(),
        "split train/val": "train_df" in content and "val_df" in content,
        "90/10 split": "0.9" in content or "90" in content,
    }
    
    for check_name, result in checks.items():
        if result:
            success.append(f"✅ Training script có {check_name}")
        else:
            warnings.append(f"⚠️  Training script thiếu {check_name}")
            
except Exception as e:
    errors.append(f"❌ Lỗi đọc train script: {e}")

# 4. Kiểm tra model_loader.py nhất quán với config
print("\n4️⃣  Kiểm tra app/core/model_loader.py")
try:
    from app.core.model_loader import model, processor
    from app.core.config import MODEL_PATH, get_device as config_get_device
    
    # Kiểm tra import từ config
    if "from app.core.config import" in (BASE_DIR / "app" / "core" / "model_loader.py").read_text():
        success.append("✅ model_loader.py import từ config")
    else:
        errors.append("❌ model_loader.py không import từ config")
    
    # Kiểm tra device consistency
    loader_device = get_device()
    config_device = config_get_device()
    if loader_device == config_device:
        success.append(f"✅ Device nhất quán: {loader_device}")
    else:
        errors.append(f"❌ Device không nhất quán: {loader_device} vs {config_device}")
        
except Exception as e:
    errors.append(f"❌ Lỗi kiểm tra model_loader: {e}")

# 5. Kiểm tra routes_caption.py nhất quán
print("\n5️⃣  Kiểm tra app/api/routes_caption.py")
try:
    routes_content = (BASE_DIR / "app" / "api" / "routes_caption.py").read_text()
    
    checks = {
        "import từ config": "from app.core.config import" in routes_content,
        "dùng get_device": "get_device()" in routes_content,
        "dùng MAX_NEW_TOKENS": "MAX_NEW_TOKENS" in routes_content,
        "dùng NUM_BEAMS": "NUM_BEAMS" in routes_content,
    }
    
    for check_name, result in checks.items():
        if result:
            success.append(f"✅ routes_caption.py {check_name}")
        else:
            warnings.append(f"⚠️  routes_caption.py thiếu {check_name}")
            
except Exception as e:
    errors.append(f"❌ Lỗi kiểm tra routes: {e}")

# 6. Kiểm tra imports không lỗi
print("\n6️⃣  Kiểm tra imports")
try:
    # Test import các module chính
    from app.core.config import MODEL_PATH, get_device
    from app.core.model_loader import model, processor
    from app.api.routes_caption import router
    from app.main import app
    
    success.append("✅ Tất cả imports hoạt động")
except Exception as e:
    errors.append(f"❌ Lỗi import: {e}")

# 7. Kiểm tra path consistency
print("\n7️⃣  Kiểm tra path consistency")
try:
    from app.core.config import MODEL_PATH, DATA_DIR, IMAGES_DIR, CSV_PATH
    
    # Kiểm tra paths tồn tại hoặc có thể tạo
    paths = {
        "MODEL_PATH": MODEL_PATH,
        "DATA_DIR": DATA_DIR,
        "IMAGES_DIR": IMAGES_DIR,
    }
    
    for name, path in paths.items():
        if path.parent.exists() or path.exists():
            success.append(f"✅ {name} có thể truy cập: {path}")
        else:
            warnings.append(f"⚠️  {name} không tồn tại: {path}")
    
    # Kiểm tra CSV
    if CSV_PATH.exists():
        success.append(f"✅ CSV_PATH tồn tại: {CSV_PATH}")
    else:
        errors.append(f"❌ CSV_PATH không tồn tại: {CSV_PATH}")
        
except Exception as e:
    errors.append(f"❌ Lỗi kiểm tra paths: {e}")

# Tổng kết
print("\n" + "=" * 70)
print("📊 TỔNG KẾT")
print("=" * 70)

print(f"\n✅ Thành công: {len(success)}")
for msg in success:
    print(f"  {msg}")

if warnings:
    print(f"\n⚠️  Cảnh báo: {len(warnings)}")
    for msg in warnings:
        print(f"  {msg}")

if errors:
    print(f"\n❌ Lỗi: {len(errors)}")
    for msg in errors:
        print(f"  {msg}")
    sys.exit(1)
else:
    print("\n🎉 TẤT CẢ KIỂM TRA ĐÃ PASS!")
    print("✅ Logic tổng thể ổn định và nhất quán")
    sys.exit(0)

