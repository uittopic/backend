#!/usr/bin/env python3
"""
Đo BLEU-1, BLEU-2, BLEU-3, BLEU-4 chi tiết
+ Tối ưu generation parameters để cải thiện BLEU
"""
import csv
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Thử import nltk, nếu chưa có thì thông báo
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Cai dat nltk: pip install nltk")
    NLTK_AVAILABLE = False


BASE_DIR = Path(__file__).parent.parent
DEFAULT_COMPARISON_CSV = BASE_DIR / "outputs" / "comparison_test_20.csv"


def normalize_text(text: str) -> str:
    """Chuẩn hóa text TRƯỚC KHI đo BLEU - rất quan trọng!"""
    text = text.lower().strip()
    # Loại bỏ ký tự đặc biệt, giữ chữ và số
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vi(text: str) -> List[str]:
    """Tokenize tiếng Việt (split by space - đơn giản nhưng hiệu quả)"""
    return text.strip().split()


def calculate_all_bleu(prediction: str, reference: str) -> Dict[str, float]:
    """Tính BLEU-1, BLEU-2, BLEU-3, BLEU-4"""
    if not NLTK_AVAILABLE:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}

    # Normalize TRƯỚC KHI tokenize
    pred = normalize_text(prediction)
    ref = normalize_text(reference)

    pred_tokens = tokenize_vi(pred)
    ref_tokens = tokenize_vi(ref)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}

    smoothing = SmoothingFunction().method1  # type: ignore

    try:
        # BLEU-1 (unigram)
        bleu1: float = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)  # type: ignore[reportGeneralTypeIssues]

        # BLEU-2 (bigram)
        bleu2: float = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)  # type: ignore[reportGeneralTypeIssues]

        # BLEU-3 (trigram)
        bleu3: float = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)  # type: ignore[reportGeneralTypeIssues]

        # BLEU-4 (4-gram) - chuẩn BLEU
        bleu4: float = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)  # type: ignore[reportGeneralTypeIssues]

        # BLEU tổng (BTEC tradition: cân nhắc 1-4)
        bleu: float = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)  # type: ignore[reportGeneralTypeIssues]

        return {
            "bleu1": float(bleu1),
            "bleu2": float(bleu2),
            "bleu3": float(bleu3),
            "bleu4": float(bleu4),
            "bleu": float(bleu)
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}


def analyze_comparison_csv(csv_path: Path) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Phân tích file comparison CSV"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay: {csv_path}")

    results: List[Dict[str, Any]] = []
    all_bleu: Dict[str, List[float]] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            gt = row.get("ground_truth", "").strip()
            pred = row.get("prediction", "").strip()

            if not img or not gt or not pred:
                continue

            bleu_scores = calculate_all_bleu(pred, gt)
            for k, v in bleu_scores.items():
                all_bleu[k].append(v)

            results.append({
                "image": img,
                "ground_truth": gt,
                "prediction": pred,
                **bleu_scores
            })

    # Tính trung bình
    avg: Dict[str, float] = {}
    for k, v in all_bleu.items():
        avg[k] = sum(v) / len(v) if v else 0.0

    return avg, results


def find_best_generation_params(csv_path: Path) -> None:
    """
    Thử nghiệm các generation parameters khác nhau
    và đo lại BLEU để tìm config tốt nhất
    """
    # Đọc ground truth
    gt_map: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_map[row["image"].strip()] = row["ground_truth"].strip()

    # Các params cần thử
    beam_sizes = [3, 5, 7]
    repetition_penalties = [1.0, 1.2, 1.5, 2.0]
    length_penalties = [0.6, 1.0, 1.2, 1.5]
    max_new_tokens_list = [30, 50, 75, 100]

    print("\n" + "=" * 70)
    print("THI NGHIEM GENERATION PARAMETERS")
    print("=" * 70)
    print("Luu y: Can chay lai inference voi tung config de do BLEU thuc te")
    print("Day la grid search parameters, can integrate vao inference script")
    print()

    # Gợi ý thử nghiệm
    print("Grid search parameters goi y:")
    print()
    print("  BEAM_SIZES:        ", beam_sizes)
    print("  REPETITION_PEN:    ", repetition_penalties)
    print("  LENGTH_PENALTIES:  ", length_penalties)
    print("  MAX_NEW_TOKENS:    ", max_new_tokens_list)
    print()

    # Tính baseline với current predictions
    avg, _ = analyze_comparison_csv(csv_path)

    print("Baseline hien tai (chua optimize):")
    print(f"   BLEU-1: {avg['bleu1']:.4f}")
    print(f"   BLEU-2: {avg['bleu2']:.4f}")
    print(f"   BLEU-3: {avg['bleu3']:.4f}")
    print(f"   BLEU-4: {avg['bleu4']:.4f}")
    print(f"   BLEU:   {avg['bleu']:.4f}")
    print()

    print("HUONG DAN CAI THIEN BLEU:")
    print("=" * 70)
    print()
    print("1. TANG BLEU-1 (trung tu don):")
    print("   Model goc BLIP da tot o buoc nay")
    print("   Can dam bao text normalize dung khi do")
    print()
    print("2. TANG BLEU-4 (trung chuoi dai):")
    print("   Thu num_beams=5 hoac 7 (nhieu candidate hon)")
    print("   Thu length_penalty=1.5 (khuyen khich sinh cau dai hon)")
    print("   Thu max_new_tokens=75-100 (sinh du dai)")
    print("   Thu repetition_penalty=1.5-2.0 (han che lap)")
    print()


def print_detailed_report(csv_path: Path) -> None:
    """In báo cáo chi tiết với phân tích mẫu tốt/xấu"""
    print("\n" + "=" * 70)
    print("BAO CAO CHI TIET BLEU-1→4")
    print("=" * 70)

    avg, results = analyze_comparison_csv(csv_path)

    print()
    print("KET QUA TRUNG BINH (Normalized - lowercase, clean)")
    print(f"BLEU-1: {avg['bleu1']:.4f}")
    print(f"BLEU-2: {avg['bleu2']:.4f}")
    print(f"BLEU-3: {avg['bleu3']:.4f}")
    print(f"BLEU-4: {avg['bleu4']:.4f}")
    print(f"BLEU:   {avg['bleu']:.4f}")
    print()

    # Top 5 tốt nhất và xấu nhất
    sorted_results = sorted(results, key=lambda x: x['bleu'], reverse=True)

    print("=" * 70)
    print("TOP 5 CAPTION TOT NHAT (BLEU cao nhat):")
    print("=" * 70)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {r['image']}")
        print(f"   GT:  {r['ground_truth'][:80]}...")
        print(f"   PRED: {r['prediction'][:80]}...")
        print(f"   BLEU-1: {r['bleu1']:.3f} | BLEU-4: {r['bleu4']:.3f} | BLEU: {r['bleu']:.3f}")

    print()
    print("=" * 70)
    print("TOP 5 CAPTION XAU NHAT (BLEU thap nhat):")
    print("=" * 70)
    for i, r in enumerate(sorted_results[-5:], 1):
        print(f"\n{i}. {r['image']}")
        print(f"   GT:  {r['ground_truth'][:80]}...")
        print(f"   PRED: {r['prediction'][:80]}...")
        print(f"   BLEU-1: {r['bleu1']:.3f} | BLEU-4: {r['bleu4']:.3f} | BLEU: {r['bleu']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Do BLEU-1→4 va toi uu generation")
    parser.add_argument("--csv", type=Path, default=DEFAULT_COMPARISON_CSV,
                        help="Duong dan file comparison CSV")
    parser.add_argument("--tune", action="store_true",
                        help="Chay thi nghiem toi uu parameters")
    parser.add_argument("--report", action="store_true", default=True,
                        help="In bao cao chi tiet (mac dinh)")

    args = parser.parse_args()

    if not NLTK_AVAILABLE:
        print("Can cai nltk: pip install nltk")
        return

    # Luôn in báo cáo
    print_detailed_report(args.csv)

    # Nếu có flag --tune thì chạy thêm grid search
    if args.tune:
        find_best_generation_params(args.csv)


if __name__ == "__main__":
    main()
