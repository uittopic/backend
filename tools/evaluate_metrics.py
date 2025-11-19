#!/usr/bin/env python3
"""
Script đánh giá kết quả caption bằng các metric:
- BLEU (n-gram overlap)
- ROUGE-L (longest common subsequence)
- Sentence-BERT Similarity (semantic similarity)
"""
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK chưa được cài đặt. Chạy: pip install nltk")
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️  rouge-score chưa được cài đặt. Chạy: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers chưa được cài đặt. Chạy: pip install sentence-transformers")
    SBERT_AVAILABLE = False

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).parent.parent

# Đường dẫn mặc định
PREDICTIONS_CSV = BASE_DIR / "outputs" / "shopee_test_predictions.csv"
GROUND_TRUTH_CSV = None  # Có thể truyền qua argument

# Model Sentence-BERT cho tiếng Việt
# Có thể dùng: "keepitreal/vietnamese-sbert" hoặc "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_MODEL_NAME = "keepitreal/vietnamese-sbert"  # Model tiếng Việt tốt nhất
FALLBACK_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Fallback nếu model VN không có

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_nltk_data():
    """Tải dữ liệu NLTK cần thiết"""
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        print(f"⚠️  Không thể tải NLTK data: {e}")


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenize tiếng Việt đơn giản (split by space)
    NLTK word_tokenize không tốt cho tiếng Việt
    """
    return text.strip().split()


# -----------------------------
# METRIC FUNCTIONS
# -----------------------------
def calculate_bleu(prediction: str, reference: str) -> float:
    """
    Tính BLEU score giữa prediction và reference
    """
    if not NLTK_AVAILABLE:
        return 0.0

    try:
        pred_tokens = tokenize_vietnamese(prediction)
        ref_tokens = tokenize_vietnamese(reference)

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # Dùng smoothing để tránh score = 0 khi không có n-gram match
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"⚠️  Lỗi tính BLEU: {e}")
        return 0.0


def calculate_rouge_l(prediction: str, reference: str) -> float:
    """
    Tính ROUGE-L score (F1 score của longest common subsequence)
    """
    if not ROUGE_AVAILABLE:
        return 0.0

    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except Exception as e:
        print(f"⚠️  Lỗi tính ROUGE-L: {e}")
        return 0.0


def calculate_sbert_similarity(prediction: str, reference: str, model) -> float:
    """
    Tính cosine similarity giữa prediction và reference bằng Sentence-BERT
    """
    if not SBERT_AVAILABLE or model is None:
        return 0.0

    try:
        # Encode cả hai câu
        embeddings = model.encode([prediction, reference], convert_to_tensor=True)
        
        # Tính cosine similarity
        from torch.nn.functional import cosine_similarity
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0].item()
        
        # Normalize về [0, 1] (cosine similarity thường trong [-1, 1])
        similarity = (similarity + 1) / 2
        return similarity
    except Exception as e:
        print(f"⚠️  Lỗi tính SBERT similarity: {e}")
        return 0.0


# -----------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------
def evaluate_predictions(
    predictions_csv: Path,
    ground_truth_csv: Optional[Path] = None,
    use_sbert: bool = True
) -> Dict[str, float]:
    """
    Đánh giá predictions với ground truth
    
    Args:
        predictions_csv: File CSV chứa predictions (có cột 'image', 'caption_full')
        ground_truth_csv: File CSV chứa ground truth (có cột 'image', 'caption')
        use_sbert: Có dùng Sentence-BERT không (cần model lớn)
    
    Returns:
        Dict chứa các metric scores
    """
    print("=" * 60)
    print("📊 ĐÁNH GIÁ KẾT QUẢ CAPTION")
    print("=" * 60)
    print(f"📄 Predictions: {predictions_csv}")
    if ground_truth_csv:
        print(f"📄 Ground truth: {ground_truth_csv}")
    else:
        print("⚠️  Không có ground truth - chỉ có thể đánh giá thủ công")
    print()

    # Load predictions
    if not predictions_csv.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file: {predictions_csv}")

    predictions = {}
    with open(predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            caption = row.get("caption_full", "").strip()
            if img_name and caption:
                predictions[img_name] = caption

    print(f"✅ Đã load {len(predictions)} predictions\n")

    # Load ground truth nếu có
    ground_truth = {}
    if ground_truth_csv and ground_truth_csv.exists():
        with open(ground_truth_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image", "").strip()
                caption = row.get("caption", "").strip()
                if img_name and caption:
                    ground_truth[img_name] = caption
        print(f"✅ Đã load {len(ground_truth)} ground truth captions\n")

    # Nếu không có ground truth, chỉ in thông tin
    if not ground_truth:
        print("⚠️  Không có ground truth để đánh giá tự động.")
        print("💡 Bạn có thể:")
        print("   1. Xem thủ công một số predictions trong file CSV")
        print("   2. Cung cấp file ground truth CSV với format:")
        print("      image,caption")
        print("      img1.jpg,caption tiếng Việt có dấu")
        return {}

    # Load Sentence-BERT model nếu cần
    sbert_model = None
    if use_sbert and SBERT_AVAILABLE:
        print("🔄 Đang load Sentence-BERT model...")
        try:
            sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
            print(f"✅ Đã load model: {SBERT_MODEL_NAME}")
        except Exception as e:
            print(f"⚠️  Không thể load {SBERT_MODEL_NAME}: {e}")
            try:
                print(f"🔄 Thử load fallback model: {FALLBACK_SBERT_MODEL}")
                sbert_model = SentenceTransformer(FALLBACK_SBERT_MODEL)
                print(f"✅ Đã load fallback model")
            except Exception as e2:
                print(f"⚠️  Không thể load fallback model: {e2}")
                sbert_model = None
        print()

    # Load NLTK data nếu cần
    if NLTK_AVAILABLE:
        load_nltk_data()

    # Tính metrics
    print("🔄 Đang tính metrics...")
    bleu_scores = []
    rouge_scores = []
    sbert_scores = []

    matched_images = []
    for img_name in predictions:
        if img_name not in ground_truth:
            continue

        pred = predictions[img_name]
        ref = ground_truth[img_name]

        # BLEU
        if NLTK_AVAILABLE:
            bleu = calculate_bleu(pred, ref)
            bleu_scores.append(bleu)

        # ROUGE-L
        if ROUGE_AVAILABLE:
            rouge = calculate_rouge_l(pred, ref)
            rouge_scores.append(rouge)

        # SBERT
        if sbert_model is not None:
            sbert = calculate_sbert_similarity(pred, ref, sbert_model)
            sbert_scores.append(sbert)

        matched_images.append(img_name)

    # Tính trung bình
    results = {}
    if bleu_scores:
        results["BLEU"] = sum(bleu_scores) / len(bleu_scores)
    if rouge_scores:
        results["ROUGE-L"] = sum(rouge_scores) / len(rouge_scores)
    if sbert_scores:
        results["SBERT_Similarity"] = sum(sbert_scores) / len(sbert_scores)

    # In kết quả
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"📈 Số mẫu đánh giá: {len(matched_images)}")
    print()
    if "BLEU" in results:
        print(f"📊 BLEU Score: {results['BLEU']:.4f}")
    if "ROUGE-L" in results:
        print(f"📊 ROUGE-L F1: {results['ROUGE-L']:.4f}")
    if "SBERT_Similarity" in results:
        print(f"📊 SBERT Similarity: {results['SBERT_Similarity']:.4f}")
    print("=" * 60)

    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá kết quả caption bằng các metric")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(PREDICTIONS_CSV),
        help="Đường dẫn file CSV chứa predictions (mặc định: outputs/shopee_test_predictions.csv)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Đường dẫn file CSV chứa ground truth (format: image,caption)",
    )
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Không dùng Sentence-BERT (nhanh hơn nhưng thiếu metric semantic similarity)",
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth) if args.ground_truth else None

    evaluate_predictions(
        predictions_path,
        ground_truth_path,
        use_sbert=not args.no_sbert
    )


if __name__ == "__main__":
    main()

