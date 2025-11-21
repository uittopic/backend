#!/usr/bin/env python3
"""
Script đánh giá model trên tập test 20%
So sánh predictions với ground truth từ test_20.csv
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List

# Import evaluate function
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_metrics import evaluate_predictions

BASE_DIR = Path(__file__).parent.parent

# Đường dẫn mặc định
PREDICTIONS_CSV = BASE_DIR / "outputs" / "predictions_test.csv"
GROUND_TRUTH_CSV = BASE_DIR / "data" / "test_20.csv"
OUTPUT_EVAL_CSV = BASE_DIR / "outputs" / "evaluation_results.csv"


def prepare_ground_truth_for_evaluation(ground_truth_csv: Path, output_csv: Path):
    """
    Chuyển đổi test_20.csv (có cột caption_vi) 
    thành format đơn giản (image, caption) để evaluate
    """
    print(f"📖 Đang đọc ground truth từ: {ground_truth_csv}")
    
    ground_truth_simple = {}
    with open(ground_truth_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            caption_vi = row.get("caption_vi", "").strip()
            if img_name and caption_vi:
                ground_truth_simple[img_name] = caption_vi
    
    print(f"✅ Đã load {len(ground_truth_simple)} ground truth captions")
    
    # Ghi file ground truth đơn giản
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        for img_name, caption in ground_truth_simple.items():
            writer.writerow({"image": img_name, "caption": caption})
    
    print(f"💾 Đã ghi ground truth đơn giản vào: {output_csv}")
    return output_csv


def create_comparison_table(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_csv: Path
):
    """
    Tạo bảng so sánh predictions vs ground truth
    """
    print("\n" + "=" * 60)
    print("📊 TẠO BẢNG SO SÁNH")
    print("=" * 60)
    
    # Load predictions
    predictions = {}
    with open(predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            caption_full = row.get("caption_full", "").strip()
            if img_name and caption_full:
                predictions[img_name] = caption_full
    
    # Load ground truth
    ground_truth = {}
    with open(ground_truth_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            caption_vi = row.get("caption_vi", "").strip()
            if img_name and caption_vi:
                ground_truth[img_name] = caption_vi
    
    # Tạo bảng so sánh
    comparison_rows = []
    matched = 0
    for img_name in predictions:
        if img_name in ground_truth:
            comparison_rows.append({
                "image": img_name,
                "ground_truth": ground_truth[img_name],
                "prediction": predictions[img_name],
            })
            matched += 1
    
    print(f"✅ Đã match {matched}/{len(predictions)} predictions với ground truth")
    
    # Ghi file so sánh
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction"])
        writer.writeheader()
        writer.writerows(comparison_rows)
    
    print(f"💾 Đã ghi bảng so sánh vào: {output_csv}")
    return comparison_rows


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Đánh giá model trên tập test 20%"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(PREDICTIONS_CSV),
        help="File CSV chứa predictions (mặc định: outputs/predictions_test.csv)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=str(GROUND_TRUTH_CSV),
        help="File CSV chứa ground truth (mặc định: data/test_20.csv)",
    )
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Không dùng Sentence-BERT (nhanh hơn)",
    )
    parser.add_argument(
        "--create-comparison",
        action="store_true",
        help="Tạo file CSV so sánh predictions vs ground truth",
    )
    
    args = parser.parse_args()
    
    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    
    if not predictions_path.exists():
        print(f"❌ Không tìm thấy file predictions: {predictions_path}")
        sys.exit(1)
    
    if not ground_truth_path.exists():
        print(f"❌ Không tìm thấy file ground truth: {ground_truth_path}")
        sys.exit(1)
    
    # Chuẩn bị ground truth đơn giản (nếu cần)
    ground_truth_simple = BASE_DIR / "outputs" / "ground_truth_simple.csv"
    prepare_ground_truth_for_evaluation(ground_truth_path, ground_truth_simple)
    
    # Tạo bảng so sánh nếu cần
    if args.create_comparison:
        comparison_csv = BASE_DIR / "outputs" / "comparison_test_20.csv"
        create_comparison_table(predictions_path, ground_truth_path, comparison_csv)
    
    # Chạy evaluate
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ")
    print("=" * 60)
    
    results = evaluate_predictions(
        predictions_path,
        ground_truth_simple,
        use_sbert=not args.no_sbert
    )
    
    # Ghi kết quả vào file
    if results:
        results_csv = BASE_DIR / "outputs" / "evaluation_metrics.csv"
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "score"])
            writer.writeheader()
            for metric, score in results.items():
                writer.writerow({"metric": metric, "score": score})
        print(f"\n💾 Đã ghi metrics vào: {results_csv}")
    
    print("\n✅ Hoàn thành đánh giá!")


if __name__ == "__main__":
    main()

