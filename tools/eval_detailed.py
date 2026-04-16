#!/usr/bin/env python3
"""
Đánh giá model với nhiều metrics hơn:
- BLEU (có dấu & không dấu)
- ROUGE-L
- METEOR (nếu có)
- Human evaluation (20 mẫu ngẫu nhiên)
"""

import json
import csv
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re

BASE = Path(__file__).parent.parent
# File 1: Baseline (20260416 - timestamp cũ hơn, quality thấp hơn)
BATCH_TEST_BASELINE = BASE / "outputs" / "batch_test" / "batch_test_results_20260416_040010.json"
# File 2: V1 (20260415 - timestamp mới hơn, quality tốt hơn)
BATCH_TEST_V1 = BASE / "outputs" / "batch_test" / "batch_test_results_20260415_125950.json"
TEST_CSV = BASE / "data" / "test_20.csv"
METRICS_JSON = BASE / "outputs" / "comparison_baseline_vs_v1.json"

# Default: dùng V1 để đánh giá đơn lẻ
BATCH_TEST_JSON = BATCH_TEST_V1

def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics using unicodedata."""
    import unicodedata
    # Normalize NFD, then remove combining marks
    nfd = unicodedata.normalize('NFD', text)
    # Keep only base characters (remove combining marks)
    result = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', result)


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Tạo n-grams từ tokens."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: List[str], hypothesis: List[str], n: int = 4,
               smooth: bool = True) -> float:
    """
    Tính BLEU score với công thức chuẩn.
    Reference: Papineni et al. (2002) - BLEU: a Method for Automatic Evaluation of Machine Translation

    Args:
        reference: list of tokens in reference
        hypothesis: list of tokens in hypothesis
        n: max n-gram order
        smooth: whether to apply smoothing for short sentences

    Returns:
        BLEU score (0.0 to 1.0)
    """
    if not reference or not hypothesis:
        return 0.0

    # Calculate modified precision for each n-gram level
    precisions = []
    for i in range(1, n + 1):
        if i > len(reference) or i > len(hypothesis):
            precisions.append(0.0)
            continue

        hyp_ngrams = get_ngrams(hypothesis, i)
        ref_ngrams = get_ngrams(reference, i)

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        # Modified precision: count clip by max ref count
        matches = 0
        for ng in hyp_ngrams:
            max_count = max(ref_ngrams.get(ng, 0), 0)
            matches += min(hyp_ngrams[ng], max_count)

        total = sum(hyp_ngrams.values())
        precision = matches / total if total > 0 else 0.0

        # Apply smoothing for low counts
        if smooth and precision == 0:
            precision = 1e-10  # Avoid log(0)

        precisions.append(precision)

    # Log average of precisions
    log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
    avg_log_precision = sum(log_precisions) / n

    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Công thức chuẩn: BP = exp(min(0, 1 - ref_len/hyp_len))
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    bleu = bp * math.exp(avg_log_precision)
    return max(0.0, min(1.0, bleu))  # Clamp to [0, 1]


def rouge_l(reference: List[str], hypothesis: List[str]) -> float:
    """
    Tính ROUGE-L (Longest Common Subsequence).
    """
    m = len(reference)
    n = len(hypothesis)

    if m == 0 or n == 0:
        return 0.0

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1].lower() == hypothesis[j-1].lower():
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    recall = lcs_len / m if m > 0 else 0.0
    precision = lcs_len / n if n > 0 else 0.0

    if recall + precision == 0:
        return 0.0

    f1 = 2 * recall * precision / (recall + precision)
    return f1


def tokenize(text: str) -> List[str]:
    """Tokenize text thành words."""
    return re.findall(r'\w+', text.lower())


def compute_metrics(ref: str, hyp: str) -> Dict[str, float]:
    """Compute all metrics cho một cặp ref-hyp."""
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)

    # BLEU với dấu
    bleu1 = bleu_score(ref_tokens, hyp_tokens, 1)
    bleu2 = bleu_score(ref_tokens, hyp_tokens, 2)
    bleu3 = bleu_score(ref_tokens, hyp_tokens, 3)
    bleu4 = bleu_score(ref_tokens, hyp_tokens, 4)

    # BLEU không dấu
    ref_no_diac = tokenize(remove_diacritics(ref))
    hyp_no_diac = tokenize(remove_diacritics(hyp))
    bleu1_nodiac = bleu_score(ref_no_diac, hyp_no_diac, 1)
    bleu4_nodiac = bleu_score(ref_no_diac, hyp_no_diac, 4)

    # ROUGE-L
    rouge = rouge_l(ref_tokens, hyp_tokens)

    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'bleu1_nodiac': bleu1_nodiac,
        'bleu4_nodiac': bleu4_nodiac,
        'rouge_l': rouge,
    }


def human_evaluation_sample(samples: List[Dict], n: int = 20) -> List[Dict]:
    """
    Tạo random samples để human evaluation.
    """
    random.seed(42)
    selected = random.sample(samples, min(n, len(samples)))

    return [
        {
            'id': i + 1,
            'image': s['image'],
            'ground_truth': s['ground_truth'],
            'prediction': s['prediction'],
            'bleu4': s.get('bleu4', 0),
            'human_score': None,  # Để người đánh giá điền
            'notes': ''  # Ghi chú thêm
        }
        for i, s in enumerate(selected)
    ]


def main():
    print("📊 Đánh giá chi tiết Model BLIP Vietnamese")
    print("=" * 60)

    # Đọc test data
    print("\n📖 Đang đọc test data...")
    gt_data = {}
    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data[row['image']] = row.get('caption_vi', row.get('caption_vi_cleaned', ''))

    # Đọc batch test results
    print("📖 Đang đọc batch test results...")
    with open(BATCH_TEST_JSON) as f:
        batch_data = json.load(f)

    results = batch_data['results']
    total_images = batch_data['metadata']['stats']['total_images']

    print(f"📊 Total images: {total_images}")

    # Compute metrics
    print("\n🔄 Đang tính metrics...")
    all_metrics = []
    sample_predictions = []

    for item in results:
        if not item.get('success'):
            continue

        image_name = item['image_name']
        gt = gt_data.get(image_name, '')
        hyp = item.get('caption_vi', '')

        if not gt or not hyp:
            continue

        metrics = compute_metrics(gt, hyp)
        metrics['image'] = image_name
        metrics['ground_truth'] = gt
        metrics['prediction'] = hyp

        all_metrics.append(metrics)
        sample_predictions.append(metrics)

    n = len(all_metrics)
    print(f"📊 Valid samples: {n}")

    # Tổng hợp metrics
    avg_metrics = {
        'bleu1': sum(m['bleu1'] for m in all_metrics) / n,
        'bleu2': sum(m['bleu2'] for m in all_metrics) / n,
        'bleu3': sum(m['bleu3'] for m in all_metrics) / n,
        'bleu4': sum(m['bleu4'] for m in all_metrics) / n,
        'bleu1_nodiac': sum(m['bleu1_nodiac'] for m in all_metrics) / n,
        'bleu4_nodiac': sum(m['bleu4_nodiac'] for m in all_metrics) / n,
        'rouge_l': sum(m['rouge_l'] for m in all_metrics) / n,
    }

    # Print results
    print("\n" + "=" * 60)
    print("📈 KẾT QUẢ ĐÁNH GIÁ CHI TIẾT")
    print("=" * 60)

    print("\n🎯 BLEU Scores (có dấu):")
    print(f"   BLEU-1: {avg_metrics['bleu1']:.4f} ({avg_metrics['bleu1']*100:.2f}%)")
    print(f"   BLEU-2: {avg_metrics['bleu2']:.4f} ({avg_metrics['bleu2']*100:.2f}%)")
    print(f"   BLEU-3: {avg_metrics['bleu3']:.4f} ({avg_metrics['bleu3']*100:.2f}%)")
    print(f"   BLEU-4: {avg_metrics['bleu4']:.4f} ({avg_metrics['bleu4']*100:.2f}%)")

    print("\n🎯 BLEU Scores (không dấu):")
    print(f"   BLEU-1: {avg_metrics['bleu1_nodiac']:.4f} ({avg_metrics['bleu1_nodiac']*100:.2f}%)")
    print(f"   BLEU-4: {avg_metrics['bleu4_nodiac']:.4f} ({avg_metrics['bleu4_nodiac']*100:.2f}%)")

    print("\n🎯 ROUGE-L Score:")
    print(f"   ROUGE-L: {avg_metrics['rouge_l']:.4f} ({avg_metrics['rouge_l']*100:.2f}%)")

    # Tính BLEU distribution
    bleu4_nodiac_values = [m['bleu4_nodiac'] for m in all_metrics]
    bleu_ranges = {
        '0': sum(1 for v in bleu4_nodiac_values if v == 0),
        '0_0.1': sum(1 for v in bleu4_nodiac_values if 0 < v <= 0.1),
        '0.1_0.2': sum(1 for v in bleu4_nodiac_values if 0.1 < v <= 0.2),
        '0.2_0.3': sum(1 for v in bleu4_nodiac_values if 0.2 < v <= 0.3),
        '0.3_plus': sum(1 for v in bleu4_nodiac_values if v > 0.3),
    }

    print("\n📊 BLEU-4 (no diac) Distribution:")
    for range_name, count in bleu_ranges.items():
        print(f"   {range_name}: {count} ({count*100/n:.1f}%)")

    # Human evaluation samples
    print("\n🎲 Đang tạo samples cho human evaluation...")
    human_samples = human_evaluation_sample(all_metrics, n=20)

    # Save detailed metrics JSON
    output = {
        'metadata': {
            'timestamp': '2026-04-15',
            'total_samples': n,
            'model': 'BLIP Vietnamese (cleaned_v1)',
        },
        'average_metrics': {k: round(v, 4) for k, v in avg_metrics.items()},
        'bleu_distribution': bleu_ranges,
        'sample_predictions': sample_predictions[:100],
        'human_evaluation_samples': human_samples,
    }

    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu metrics chi tiết: {METRICS_JSON}")

    # Print human evaluation samples
    print("\n" + "=" * 60)
    print("📝 MẪU CHO HUMAN EVALUATION (20 mẫu ngẫu nhiên)")
    print("=" * 60)

    for s in human_samples:
        print(f"\n[#{s['id']}] {s['image']}")
        print(f"    GT:  {s['ground_truth'][:60]}")
        print(f"    Pred: {s['prediction'][:60]}")
        print(f"    BLEU-4: {s['bleu4']:.3f}")


if __name__ == '__main__':
    main()
