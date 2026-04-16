#!/usr/bin/env python3
"""
Full Pipeline Script - Từ data cleaning → train → eval → demo

Chạy tuần tự tất cả các bước để nâng cấp model.
"""

import subprocess
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent


def run_step(name: str, command: list, env: dict = None, check: bool = True) -> bool:
    """
    Chạy một bước và hiển thị tiến trình.
    
    Args:
        name: Tên bước để hiển thị
        command: List of command arguments
        env: Optional dict of environment variables
        check: Nếu True, fail pipeline nếu step thất bại
    """
    print(f"\n{'='*60}")
    print(f"🚀 {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    if env:
        print(f"Env vars: {env}")
    print("-"*60)

    # Merge với os.environ hiện tại
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(command, cwd=BASE, env=full_env)

    if check and result.returncode != 0:
        print(f"❌ Step '{name}' failed with exit code {result.returncode}")
        return False

    print(f"✅ {name} completed")
    return True


def main():
    print("🔧 FULL PIPELINE - BLIP Vietnamese Upgrade")
    print("="*60)
    print("Bước 1: Data Cleaning v2 (đã có sẵn - skip nếu đã có)")
    print("Bước 2: Train model với data v2")
    print("Bước 3: Post-processing")
    print("Bước 4: Export metrics")
    print("Bước 5: Detailed evaluation")
    print("="*60)

    # Confirm
    response = input("\n❓ Bạn có muốn tiếp tục? (y/n): ").strip().lower()
    if response != 'y':
        print("Đã hủy.")
        return

    success = True

    # Bước 1: Check data v2 có tồn tại chưa
    data_v2 = BASE / "data" / "train_80_cleaned_v2.csv"
    if not data_v2.exists():
        print(f"\n⚠️  Chưa có data v2. Chạy clean_data_v2.py trước...")
        if not run_step(
            "DATA CLEANING v2",
            ["python", "tools/clean_data_v2.py",
             "--input", "data/train_80.csv",
             "--output", "data/train_80_cleaned_v2.csv"],
            check=True
        ):
            print("\n❌ Data cleaning failed. Dừng pipeline.")
            return
    else:
        print(f"\n✅ Data v2 đã tồn tại: {data_v2}")

    # Bước 2: Train model với env vars đúng
    print("\n" + "="*60)
    print("🚀 TRAINING MODEL V2")
    print("="*60)
    print("⚠️  LƯU Ý: Training sẽ mất 15-20 giờ!")
    print("   Có thể bấm Ctrl+C để dừng nếu cần.")
    print("-"*60)

    train_env = {
        "USE_CLEANED_V2": "true",
        "NUM_EPOCHS": "8",
        "LEARNING_RATE": "2e-5",
        "MAX_LENGTH": "100",
        "GRADIENT_ACCUMULATION_STEPS": "2",
        "MODEL_OUTPUT_DIR": "models/blip_vietnamese_80_20",
        "OVERWRITE_OUTPUT_DIR": "false",
        "AUTO_RESUME": "true",
    }

    print("Training với hyperparameters:")
    for k, v in train_env.items():
        print(f"  {k}={v}")

    response = input("\n❓ Tiếp tục training? (y/n): ").strip().lower()
    if response != 'y':
        print("Đã hủy training. Bạn có thể chạy manual:")
        print("  USE_CLEANED_V2=true NUM_EPOCHS=8 ...")
        return

    if not run_step(
        "TRAINING MODEL V2",
        ["python", "train/train_blip_vietnamese.py"],
        env=train_env,
        check=True
    ):
        print("\n⚠️  Training failed.")
        return

    # Bước 3: Post-processing
    print("\n" + "="*60)
    print("⚠️  LƯU Ý: PHẢI CHẠY INFERENCE TRƯỚC KHI POST-PROCESS")
    print("="*60)
    print("""
    Trước khi chạy post-processing, bạn cần:
    
    1. Chạy inference trên test set:
       python tools/run_inference_full_test.py \\
           --model models/blip_vietnamese_80_20 \\
           --output outputs/batch_test_v2.json

    2. Sau đó chạy script này để post-process
    """)

    response = input("\n❓ Đã chạy inference chưa? (y để tiếp tục post-processing, n để bỏ qua): ").strip().lower()
    
    if response == 'y':
        predictions_file = BASE / "outputs" / "batch_test_v2.json"
        if not predictions_file.exists():
            print(f"\n⚠️  Không tìm thấy {predictions_file}")
            print("   Bỏ qua post-processing...")
        else:
            # Copy file để post-process
            import shutil
            postprocess_input = BASE / "outputs" / "batch_test" / "batch_test_results_latest.json"
            if not postprocess_input.parent.exists():
                postprocess_input.parent.mkdir(parents=True)
            shutil.copy(predictions_file, postprocess_input)
            
            if not run_step(
                "POST-PROCESSING CAPTIONS",
                ["python", "tools/postprocess_captions.py"],
                check=False
            ):
                print("⚠️  Post-processing có lỗi nhưng tiếp tục...")

    # Bước 4: Export metrics
    if not run_step(
        "EXPORT METRICS TO JSON",
        ["python", "tools/export_metrics.py"],
        check=False
    ):
        print("⚠️  Export metrics có lỗi nhưng tiếp tục...")

    # Bước 5: Detailed evaluation
    if not run_step(
        "DETAILED EVALUATION (BLEU + ROUGE)",
        ["python", "tools/eval_detailed.py"],
        check=False
    ):
        print("⚠️  Detailed eval có lỗi nhưng tiếp tục...")

    print("\n" + "="*60)
    print("✅ PIPELINE HOÀN TẤT!")
    print("="*60)
    print("\n📁 Files quan trọng:")
    print(f"   📦 Model: models/blip_vietnamese_80_20/")
    print(f"   📄 Metrics: outputs/metrics.json")
    print(f"   📄 Detailed: outputs/metrics_detailed.json")
    print(f"   🎮 Demo: demo_streamlit.py")
    print("\n🚀 Chạy demo:")
    print("   # Terminal 1: uvicorn app.main:app --port 8000")
    print("   # Terminal 2: streamlit run demo_streamlit.py")


if __name__ == '__main__':
    main()
