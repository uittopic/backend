#!/usr/bin/env python3
"""
Script test nhanh để kiểm tra xem model có load được không
"""
import sys
import time

print("=" * 60)
print("BẮT ĐẦU TEST")
print("=" * 60)
print(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("Bước 1: Import model_loader...")
start = time.time()
from app.core.model_loader import model, processor
print(f"✅ Model BLIP đã load (mất {time.time() - start:.2f}s)")
print()

print("Bước 2: Import accent_restoration_loader...")
start = time.time()
from app.core.accent_restoration_loader import restore_accent
print(f"✅ Accent Restoration đã load (mất {time.time() - start:.2f}s)")
print()

print("=" * 60)
print("✅ TẤT CẢ MODEL ĐÃ LOAD XONG!")
print("=" * 60)
print(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}")

