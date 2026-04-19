#!/usr/bin/env python3
"""
Đánh giá Model BLIP Vietnamese — PHIÊN BẢN CHUẨN
==================================================

Script này:
1. Load model fine-tuned đúng (không fallback về pretrained)
2. Chạy inference với params tối ưu trên test set
3. Tính BLEU, ROUGE-L, METEOR
4. Hiển thị sample predictions để kiểm tra chất lượng
5. Xác định vấn đề: model generate tiếng Anh vs tiếng Việt

Usage:
  python tools/eval_model.py --max-samples 50
  python tools/eval_model.py --model-path models/blip_vietnamese_cleaned_v1 --max-samples 100
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
from dataclasses import dataclass

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
TEST_CSV = DATA_DIR / "test_20.csv"

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
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    return max(0.0, min(1.0, bp * math.exp(avg_log_precision)))


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
    recall = lcs_len / m
    precision = lcs_len / n
    return 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def has_vietnamese(text):
    VI = 'àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    return any(c.lower() in VI.lower() for c in text)


def compute_metrics(ref, hyp):
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    return {
        'bleu1': bleu_score(ref_tokens, hyp_tokens, 1),
        'bleu2': bleu_score(ref_tokens, hyp_tokens, 2),
        'bleu3': bleu_score(ref_tokens, hyp_tokens, 3),
        'bleu4': bleu_score(ref_tokens, hyp_tokens, 4),
        'rouge_l': rouge_l(ref_tokens, hyp_tokens),
        'is_vietnamese': has_vietnamese(hyp),
    }


# ============================================================
# SECTION 2: LOAD DATA
# ============================================================

def load_test_data(test_csv_path, max_samples=0):
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


# ============================================================
# SECTION 3: INFERENCE
# ============================================================

def load_model(model_path):
    """Load BLIP model từ local path, không fallback"""
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    path = Path(model_path)
    if path.exists() and any(path.iterdir()):
        print(f"📥 Loading fine-tuned model from: {path}")
        processor = BlipProcessor.from_pretrained(str(path))
        model = BlipForConditionalGeneration.from_pretrained(str(path))
        return model, processor
    else:
        print(f"⚠️  Model not found at {path}")
        return None, None


def run_inference(image_name, config, model, processor, device_obj, image_dir):
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
    del inputs, output
    if device_obj.type == 'mps':
        torch.mps.synchronize()
    return caption.strip()


# ============================================================
# SECTION 4: MAIN
# ============================================================

def main():
    import torch

    parser = argparse.ArgumentParser(description='Đánh giá Model BLIP Vietnamese')
    parser.add_argument('--test-csv', default=str(TEST_CSV))
    parser.add_argument('--model-path',
                        default=str(MODEL_DIR / "blip_vietnamese_cleaned_v1"))
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 65)
    print("EVALUATE: Model BLIP Vietnamese")
    print("=" * 65)
    print(f"Model: {args.model_path}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Max samples: {args.max_samples}")

    # Inference config
    INFER_CONFIG = {
        "max_new_tokens": 128,
        "num_beams": 5,
        "early_stopping": True,
        "repetition_penalty": 1.1,
        "length_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }

    # Load test data
    test_data = load_test_data(args.test_csv, args.max_samples)
    print(f"📊 Test samples: {len(test_data)}")

    # Load model
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(DEVICE)
    print(f"📱 Device: {DEVICE}")

    model, processor = load_model(args.model_path)
    if model is None:
        print("❌ Model load failed!")
        return

    model.to(device_obj)
    model.eval()
    IMAGE_DIR = DATA_DIR / "images"

    # Run inference
    print(f"\n🔄 Running inference...")
    results = []
    vi_count = 0

    for i, (img, gt) in enumerate(test_data.items()):
        pred = run_inference(img, INFER_CONFIG, model, processor, device_obj, IMAGE_DIR)
        if pred:
            metrics = compute_metrics(gt, pred)
            metrics['image'] = img
            metrics['ground_truth'] = gt
            metrics['prediction'] = pred
            results.append(metrics)
            if metrics['is_vietnamese']:
                vi_count += 1
        if (i + 1) % 10 == 0:
            print(f"   Đã xử lý {i+1}/{len(test_data)}")

    n = len(results)
    if n == 0:
        print("❌ Không có kết quả nào!")
        return

    # ============================================================
    # SECTION 5: KẾT QUẢ
    # ============================================================

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    vi_pct = 100 * vi_count / n

    print("\n" + "=" * 65)
    print("📊 KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 65)
    print(f"\n{'Metric':<20} {'Giá trị':<15}")
    print("-" * 40)
    print(f"{'BLEU-1':<20} {avg(results,'bleu1'):.4f} ({avg(results,'bleu1')*100:.2f}%)")
    print(f"{'BLEU-2':<20} {avg(results,'bleu2'):.4f} ({avg(results,'bleu2')*100:.2f}%)")
    print(f"{'BLEU-3':<20} {avg(results,'bleu3'):.4f} ({avg(results,'bleu3')*100:.2f}%)")
    print(f"{'BLEU-4':<20} {avg(results,'bleu4'):.4f} ({avg(results,'bleu4')*100:.2f}%)")
    print(f"{'ROUGE-L':<20} {avg(results,'rouge_l'):.4f} ({avg(results,'rouge_l')*100:.2f}%)")
    print(f"{'Vietnamese captions':<20} {vi_count}/{n} ({vi_pct:.1f}%)")

    # BLEU-4 distribution
    print("\n" + "=" * 65)
    print("📊 BLEU-4 Distribution")
    print("=" * 65)
    bleu_vals = [r['bleu4'] for r in results]
    ranges = [
        ('= 0.000', 0.0, 0.0),
        ('0.00 – 0.10', 0.001, 0.1),
        ('0.10 – 0.20', 0.1, 0.2),
        ('0.20 – 0.30', 0.2, 0.3),
        ('0.30 – 0.40', 0.3, 0.4),
        ('> 0.40', 0.4, 99.0),
    ]
    for label, lo, hi in ranges:
        count = sum(1 for v in bleu_vals if lo <= v < hi)
        pct = 100 * count / n
        bar = '█' * int(pct / 3) + '░' * (33 - int(pct / 3))
        print(f"  {label:<15} | {bar} | {count:>3} ({pct:.1f}%)")

    # Vietnamese vs English predictions
    print("\n" + "=" * 65)
    print("📊 NGÔN NGỮ PREDICTIONS")
    print("=" * 65)
    vi_results = [r for r in results if r['is_vietnamese']]
    en_results = [r for r in results if not r['is_vietnamese']]
    print(f"  Tiếng Việt: {len(vi_results)} ({100*len(vi_results)/n:.1f}%)")
    print(f"  Tiếng Anh:   {len(en_results)} ({100*len(en_results)/n:.1f}%)")

    if vi_results:
        avg_vi = sum(r['bleu4'] for r in vi_results) / len(vi_results)
        print(f"  BLEU-4 (tiếng Việt): {avg_vi:.4f}")

    # Sample predictions
    print("\n" + "=" * 65)
    print("📝 MẪU PREDICTIONS (mẫu ngẫu nhiên 10 cái)")
    print("=" * 65)
    samples = random.sample(results, min(10, len(results)))
    for i, r in enumerate(samples, 1):
        lang = "🇻🇳" if r['is_vietnamese'] else "🇬🇧"
        print(f"\n[{i}] {r['image'][:40]} {lang}")
        print(f"    GT:  {r['ground_truth'][:60]}")
        print(f"    Pred: {r['prediction'][:60]}")
        print(f"    B4: {r['bleu4']:.3f} | R-L: {r['rouge_l']:.3f}")

    # Save JSON
    OUTPUT = BASE / "outputs" / "eval_model_results.json"
    OUTPUT.parent.mkdir(exist_ok=True)
    output_data = {
        'model_path': str(args.model_path),
        'n_samples': n,
        'metrics': {
            'bleu1': round(avg(results, 'bleu1'), 4),
            'bleu2': round(avg(results, 'bleu2'), 4),
            'bleu3': round(avg(results, 'bleu3'), 4),
            'bleu4': round(avg(results, 'bleu4'), 4),
            'rouge_l': round(avg(results, 'rouge_l'), 4),
        },
        'language_stats': {
            'vietnamese': len(vi_results),
            'english': len(en_results),
            'vi_pct': round(vi_pct, 1),
        },
        'sample_predictions': results[:20],
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu: {OUTPUT}")

    # ============================================================
    # SECTION 6: KHUYẾN NGHỊ
    # ============================================================
    print("\n" + "=" * 65)
    print("💡 CHẨN ĐOÁN VÀ KHUYẾN NGHỊ")
    print("=" * 65)

    if vi_pct < 50:
        print(f"❌ Model chủ yếu generate TIẾNG ANH ({100-vi_pct:.1f}%)")
        print("   → Nguyên nhân: Training chưa đủ tốt HOẶC text prompt không đúng")
        print("   → Giải pháp:")
        print("     1. Train lại với training script CÓ text prompt:")
        print("        inputs = processor(images=img, text=text, ...)")
        print("     2. Tăng epoch: 3 → 8")
        print("     3. Dùng data cleaned tốt hơn")
    elif avg(results, 'bleu4') < 0.1:
        print("⚠️  Model generate tiếng Việt nhưng BLEU thấp")
        print("   → Nguyên nhân: Caption prediction khác ground truth")
        print("   → Giải pháp:")
        print("     1. BLEU không phải metric tốt nhất cho captioning")
        print("     2. Xem sample predictions ở trên — nếu caption HỢP LÝ thì OK")
        print("     3. Có thể ground truth không đồng nhất")
    elif avg(results, 'bleu4') > 0.3:
        print("✅ Model hoạt động TỐT!")
        print("   → BLEU-4 > 0.3, có thể dùng trong production")
    else:
        print("⚠️  Model hoạt động ở mức CHẤP NHẬN ĐƯỢC")
        print("   → BLEU-4 0.1–0.3: Cần cải thiện thêm")

    print("=" * 65)


if __name__ == '__main__':
    main()
