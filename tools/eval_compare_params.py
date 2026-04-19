#!/usr/bin/env python3
"""
Đánh giá nhanh: So sánh INFERENCE PARAMS
==========================================

So sánh 2 config:
  - Config A: baseline (MAX_NEW_TOKENS=60, BEAM=3, ...)
  - Config B: optimized (MAX_NEW_TOKENS=128, BEAM=5, LENGTH_PENALTY=1.2, ...)

Chạy inference trên test set, so sánh BLEU-1/4, ROUGE-L

Usage:
  python tools/eval_compare_params.py --test-csv data/test_20.csv --max-samples 200

Đầu ra:
  - So sánh metrics A vs B
  - Top 10 predictions tốt nhất / xấu nhất
  - Sample captions để xem chất lượng
"""

import os
import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
TEST_CSV = DATA_DIR / "test_20.csv"

# Thêm parent vào path để import
import sys
sys.path.insert(0, str(BASE))

# ============================================================
# SECTION 1: METRICS
# ============================================================

def remove_diacritics(text: str) -> str:
    import unicodedata
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference, hypothesis, n=4):
    if not reference or not hypothesis:
        return 0.0

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
        matches = sum(min(hyp_ngrams[ng], max(ref_ngrams.get(ng, 0), 0)) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precision = matches / total if total > 0 else 0.0
        if precision == 0:
            precision = 1e-10
        precisions.append(precision)

    log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
    avg_log_precision = sum(log_precisions) / n

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    bleu = bp * math.exp(avg_log_precision)
    return max(0.0, min(1.0, bleu))


def rouge_l(reference, hypothesis):
    m, n = len(reference), len(hypothesis)
    if m == 0 or n == 0:
        return 0.0
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
    return 2 * recall * precision / (recall + precision)


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def compute_metrics(ref, hyp):
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    bleu1 = bleu_score(ref_tokens, hyp_tokens, 1)
    bleu4 = bleu_score(ref_tokens, hyp_tokens, 4)
    rouge = rouge_l(ref_tokens, hyp_tokens)

    # No-diacritics
    ref_nd = tokenize(remove_diacritics(ref))
    hyp_nd = tokenize(remove_diacritics(hyp))
    bleu4_nd = bleu_score(ref_nd, hyp_nd, 4)

    return {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'bleu4_nodiac': bleu4_nd,
        'rouge_l': rouge,
    }


# ============================================================
# SECTION 2: CONFIGURATIONS
# ============================================================

# Config A: BASELINE
CONFIG_A = {
    "max_new_tokens": 60,
    "num_beams": 3,
    "early_stopping": True,
    "repetition_penalty": 1.05,
    "length_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "top_k": 0,
    "do_sample": False,
}

# Config B: OPTIMIZED
CONFIG_B = {
    "max_new_tokens": 128,
    "num_beams": 5,
    "early_stopping": True,
    "repetition_penalty": 1.1,
    "length_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "top_k": 50,
    "do_sample": False,
}


# ============================================================
# SECTION 3: INFERENCE
# ============================================================

def load_test_data(test_csv_path, max_samples=0):
    """Load test data từ CSV"""
    gt_data = {}
    with open(test_csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get('image', '')
            cap = row.get('caption_vi', row.get('caption_vi_cleaned', ''))
            if img and cap:
                gt_data[img] = cap

    items = list(gt_data.items())
    if max_samples > 0:
        items = items[:max_samples]
    return dict(items)


def run_inference(image_name, config, model, processor, device_obj, image_dir):
    """Chạy inference với config cho trước"""
    import torch
    from PIL import Image

    img_path = image_dir / image_name
    if not img_path.exists():
        return None

    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device_obj)

    with torch.no_grad():
        output = model.generate(**inputs, **config)

    caption = processor.decode(output[0], skip_special_tokens=True)
    # Cleanup
    del inputs, output
    if device_obj.type == 'mps':
        torch.mps.synchronize()
    return caption.strip()


# ============================================================
# SECTION 4: MAIN
# ============================================================

def main():
    import torch

    parser = argparse.ArgumentParser(description='So sánh inference params')
    parser.add_argument('--test-csv', default=str(TEST_CSV))
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Số mẫu test (0 = tất cả)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 65)
    print("EVALUATE: So sánh INFERENCE PARAMS")
    print("=" * 65)

    # Load test data
    test_data = load_test_data(args.test_csv, args.max_samples)
    print(f"📊 Test samples: {len(test_data)}")

    # Load model
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(DEVICE)
    print(f"📱 Device: {DEVICE}")

    MODEL_PATH = BASE / "models" / "blip_vietnamese_cleaned_v1"
    if not MODEL_PATH.exists() or not any(MODEL_PATH.iterdir()):
        MODEL_PATH = BASE / "models" / "blip_vietnamese"
        if not MODEL_PATH.exists():
            print(f"❌ Không tìm thấy model tại {MODEL_PATH}")
            print("⚠️  Sẽ dùng pretrained BLIP để demo")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model.to(device_obj)
            model.eval()
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
            model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
            model.to(device_obj)
            model.eval()
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
        model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
        model.to(device_obj)
        model.eval()

    IMAGE_DIR = BASE / "data" / "images"

    # Run inference với 2 configs
    results_a = []
    results_b = []

    print(f"\n🔄 Running inference với Config A (baseline)...")
    for i, (img, gt) in enumerate(test_data.items()):
        pred = run_inference(img, CONFIG_A, model, processor, device_obj, IMAGE_DIR)
        if pred:
            metrics = compute_metrics(gt, pred)
            metrics['image'] = img
            metrics['ground_truth'] = gt
            metrics['prediction'] = pred
            results_a.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"   Đã xử lý {i+1}/{len(test_data)}")

    print(f"\n🔄 Running inference với Config B (optimized)...")
    for i, (img, gt) in enumerate(test_data.items()):
        pred = run_inference(img, CONFIG_B, model, processor, device_obj, IMAGE_DIR)
        if pred:
            metrics = compute_metrics(gt, pred)
            metrics['image'] = img
            metrics['ground_truth'] = gt
            metrics['prediction'] = pred
            results_b.append(metrics)

    n = len(results_a)
    if n == 0:
        print("❌ Không có kết quả nào!")
        return

    # ============================================================
    # SECTION 5: TỔNG HỢP KẾT QUẢ
    # ============================================================

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    print("\n" + "=" * 65)
    print("📊 KẾT QUẢ SO SÁNH")
    print("=" * 65)
    print(f"\n{'Metric':<20} {'Config A (baseline)':<20} {'Config B (optimized)':<20} {'Δ':<10}")
    print("-" * 70)

    metrics_keys = ['bleu1', 'bleu4', 'bleu4_nodiac', 'rouge_l']
    metric_names = {
        'bleu1': 'BLEU-1',
        'bleu4': 'BLEU-4',
        'bleu4_nodiac': 'BLEU-4 (no diac)',
        'rouge_l': 'ROUGE-L',
    }

    for k in metrics_keys:
        v_a = avg(results_a, k)
        v_b = avg(results_b, k)
        delta = v_b - v_a
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{metric_names[k]:<20} {v_a:.4f} ({v_a*100:.2f}%){'':<8} {v_b:.4f} ({v_b*100:.2f}%){'':<8} {delta_str}")

    print("\n" + "=" * 65)
    print("📝 MẪU CAPTIONS (Top 5 tốt nhất — Config B)")
    print("=" * 65)

    # Sort by bleu4
    sorted_b = sorted(results_b, key=lambda x: x['bleu4'], reverse=True)
    for i, r in enumerate(sorted_b[:5], 1):
        print(f"\n[{i}] {r['image']}")
        print(f"    GT:      {r['ground_truth'][:70]}")
        print(f"    Pred(A): {results_a[i-1]['prediction'][:70]}")
        print(f"    Pred(B): {r['prediction'][:70]}")
        print(f"    BLEU-4 A: {results_a[i-1]['bleu4']:.3f} | B: {r['bleu4']:.3f}")

    # BLEU-4 distribution
    print("\n" + "=" * 65)
    print("📊 BLEU-4 Distribution (Config B)")
    print("=" * 65)
    bleu_vals = [r['bleu4_nodiac'] for r in results_b]
    ranges = {
        '= 0.000': sum(1 for v in bleu_vals if v == 0),
        '0.00 – 0.10': sum(1 for v in bleu_vals if 0 < v <= 0.1),
        '0.10 – 0.20': sum(1 for v in bleu_vals if 0.1 < v <= 0.2),
        '0.20 – 0.30': sum(1 for v in bleu_vals if 0.2 < v <= 0.3),
        '0.30 – 0.40': sum(1 for v in bleu_vals if 0.3 < v <= 0.4),
        '> 0.40': sum(1 for v in bleu_vals if v > 0.4),
    }
    for label, count in ranges.items():
        pct = 100 * count / n
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {label:<15} | {bar} | {count:>3} ({pct:.1f}%)")

    # Save to JSON
    OUTPUT = BASE / "outputs" / "eval_compare_params.json"
    OUTPUT.parent.mkdir(exist_ok=True)
    output_data = {
        'config_a': CONFIG_A,
        'config_b': CONFIG_B,
        'metrics_a': {k: round(avg(results_a, k), 4) for k in metrics_keys},
        'metrics_b': {k: round(avg(results_b, k), 4) for k in metrics_keys},
        'results_a': results_a[:20],
        'results_b': results_b[:20],
        'n_samples': n,
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu: {OUTPUT}")

    print("\n" + "=" * 65)
    print("💡 KHUYẾN NGHỊ")
    print("=" * 65)
    bleu4_b = avg(results_b, 'bleu4_nodiac')
    if bleu4_b > 0.2:
        print("✅ Config B (optimized) cho kết quả TỐT HƠN rõ rệt")
        print("   → Nên dùng config mới trong production")
    elif bleu4_b > 0.1:
        print("⚠️  Config B cải thiện NHẸ, nên dùng thêm train tốt hơn")
    else:
        print("❌ Cả 2 config đều cho kết quả THẤP")
        print("   → Cần: (1) train lại với data tốt hơn, (2) tăng epoch")
        print("   → Có thể data noisy, cần clean kỹ hơn")

    print("=" * 65)


if __name__ == '__main__':
    main()
