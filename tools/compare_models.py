#!/usr/bin/env python3
"""
So sánh 2 model: Baseline vs V1
- Baseline: batch_test_results_20260416_040010.json
- V1: batch_test_results_20260415_125950.json
"""
import json
import csv
from pathlib import Path
from collections import Counter
import math
import re

BASE = Path(__file__).parent.parent
BATCH_TEST_BASELINE = BASE / "outputs" / "batch_test" / "batch_test_results_20260416_040010.json"
BATCH_TEST_V1 = BASE / "outputs" / "batch_test" / "batch_test_results_20260415_125950.json"
TEST_CSV = BASE / "data" / "test_20.csv"
OUTPUT_JSON = BASE / "outputs" / "final_comparison.json"

# ============ HELPER FUNCTIONS ============

def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics."""
    import unicodedata
    nfd = unicodedata.normalize('NFD', text)
    result = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', result)

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_score(reference, hypothesis, n=4, smooth=True):
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
        matches = 0
        for ng in hyp_ngrams:
            max_count = max(ref_ngrams.get(ng, 0), 0)
            matches += min(hyp_ngrams[ng], max_count)
        total = sum(hyp_ngrams.values())
        precision = matches / total if total > 0 else 0.0
        if smooth and precision == 0:
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
    f1 = 2 * recall * precision / (recall + precision)
    return f1

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def compute_metrics(ref, hyp):
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    bleu1 = bleu_score(ref_tokens, hyp_tokens, 1)
    bleu4 = bleu_score(ref_tokens, hyp_tokens, 4)
    # No diacritics
    ref_no_diac = tokenize(remove_diacritics(ref))
    hyp_no_diac = tokenize(remove_diacritics(hyp))
    bleu4_nodiac = bleu_score(ref_no_diac, hyp_no_diac, 4)
    rouge = rouge_l(ref_tokens, hyp_tokens)
    return {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'bleu4_nodiac': bleu4_nodiac,
        'rouge_l': rouge,
    }

# ============ MAIN ============

print("=" * 70)
print("🔍 SO SÁNH 2 MODEL: BASELINE vs V1")
print("=" * 70)

# Load ground truth
print("\n📖 Đang đọc ground truth...")
gt_data = {}
with open(TEST_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt_data[row['image']] = row.get('caption_vi', row.get('caption_vi_cleaned', ''))
print(f"   ✅ Load {len(gt_data)} ground truth")

# Load baseline
print("\n📖 Đang đọc Baseline...")
with open(BATCH_TEST_BASELINE) as f:
    baseline_data = json.load(f)
baseline_results = baseline_data['results']
baseline_stats = baseline_data['metadata']['stats']
print(f"   ✅ Baseline: {baseline_stats['total_images']} images, Good: {baseline_stats['good_captions']}")

# Load V1
print("\n📖 Đang đọc V1...")
with open(BATCH_TEST_V1) as f:
    v1_data = json.load(f)
v1_results = v1_data['results']
v1_stats = v1_data['metadata']['stats']
print(f"   ✅ V1: {v1_stats['total_images']} images, Good: {v1_stats['good_captions']}")

# Compute metrics
print("\n🔄 Đang tính metrics...")

baseline_metrics = []
v1_metrics = []
matched_count = 0

for i, (b_item, v_item) in enumerate(zip(baseline_results, v1_results)):
    img_name = b_item['image_name']
    if img_name not in gt_data:
        continue
    gt = gt_data[img_name]
    hyp_b = b_item.get('caption_vi', '')
    hyp_v = v_item.get('caption_vi', '')
    if not gt or not hyp_b or not hyp_v:
        continue
    
    metrics_b = compute_metrics(gt, hyp_b)
    metrics_v = compute_metrics(gt, hyp_v)
    
    baseline_metrics.append(metrics_b)
    v1_metrics.append(metrics_v)
    matched_count += 1

print(f"   ✅ Computed {matched_count} pairs")

# Calculate averages
n = len(baseline_metrics)
baseline_avg = {
    'bleu1': sum(m['bleu1'] for m in baseline_metrics) / n,
    'bleu4': sum(m['bleu4'] for m in baseline_metrics) / n,
    'bleu4_nodiac': sum(m['bleu4_nodiac'] for m in baseline_metrics) / n,
    'rouge_l': sum(m['rouge_l'] for m in baseline_metrics) / n,
}
v1_avg = {
    'bleu1': sum(m['bleu1'] for m in v1_metrics) / n,
    'bleu4': sum(m['bleu4'] for m in v1_metrics) / n,
    'bleu4_nodiac': sum(m['bleu4_nodiac'] for m in v1_metrics) / n,
    'rouge_l': sum(m['rouge_l'] for m in v1_metrics) / n,
}

# ============ PRINT RESULTS ============

print("\n" + "=" * 70)
print("📊 KẾT QUẢ SO SÁNH CHI TIẾT")
print("=" * 70)

print("\n📁 BASELINE FILE:")
print(f"   Timestamp: {baseline_data['metadata']['timestamp']}")
print(f"   Good captions: {baseline_stats['good_captions']} ({baseline_stats['good_captions']*100/baseline_stats['total_images']:.1f}%)")
print(f"   Medium captions: {baseline_stats['medium_captions']} ({baseline_stats['medium_captions']*100/baseline_stats['total_images']:.1f}%)")
print(f"   Bad captions: {baseline_stats['bad_captions']} ({baseline_stats['bad_captions']*100/baseline_stats['total_images']:.2f}%)")

print("\n📁 V1 FILE:")
print(f"   Timestamp: {v1_data['metadata']['timestamp']}")
print(f"   Good captions: {v1_stats['good_captions']} ({v1_stats['good_captions']*100/v1_stats['total_images']:.1f}%)")
print(f"   Medium captions: {v1_stats['medium_captions']} ({v1_stats['medium_captions']*100/v1_stats['total_images']:.1f}%)")
print(f"   Bad captions: {v1_stats['bad_captions']} ({v1_stats['bad_captions']*100/v1_stats['total_images']:.2f}%)")

print("\n" + "-" * 70)
print("📈 METRICS SO SÁNH")
print("-" * 70)

def improvement(old, new):
    if old == 0:
        return "N/A"
    diff = (new - old) / old * 100
    if diff > 0:
        return f"✅ +{diff:.1f}%"
    else:
        return f"❌ {diff:.1f}%"

print(f"\n{'Metric':<20} {'Baseline':<15} {'V1':<15} {'So sánh':<15}")
print("-" * 65)
print(f"{'BLEU-1':<20} {baseline_avg['bleu1']:.4f} ({baseline_avg['bleu1']*100:.2f}%) {v1_avg['bleu1']:.4f} ({v1_avg['bleu1']*100:.2f}%) {improvement(baseline_avg['bleu1'], v1_avg['bleu1']):<15}")
print(f"{'BLEU-4':<20} {baseline_avg['bleu4']:.4f} ({baseline_avg['bleu4']*100:.2f}%) {v1_avg['bleu4']:.4f} ({v1_avg['bleu4']*100:.2f}%) {improvement(baseline_avg['bleu4'], v1_avg['bleu4']):<15}")
print(f"{'BLEU-4 (no diac)':<20} {baseline_avg['bleu4_nodiac']:.4f} ({baseline_avg['bleu4_nodiac']*100:.2f}%) {v1_avg['bleu4_nodiac']:.4f} ({v1_avg['bleu4_nodiac']*100:.2f}%) {improvement(baseline_avg['bleu4_nodiac'], v1_avg['bleu4_nodiac']):<15}")
print(f"{'ROUGE-L':<20} {baseline_avg['rouge_l']:.4f} ({baseline_avg['rouge_l']*100:.2f}%) {v1_avg['rouge_l']:.4f} ({v1_avg['rouge_l']*100:.2f}%) {improvement(baseline_avg['rouge_l'], v1_avg['rouge_l']):<15}")

# Save to JSON
output = {
    'metadata': {
        'baseline_file': str(BATCH_TEST_BASELINE.name),
        'v1_file': str(BATCH_TEST_V1.name),
        'baseline_timestamp': baseline_data['metadata']['timestamp'],
        'v1_timestamp': v1_data['metadata']['timestamp'],
        'matched_samples': matched_count,
    },
    'baseline_stats': baseline_stats,
    'v1_stats': v1_stats,
    'baseline_metrics': {k: round(v, 4) for k, v in baseline_avg.items()},
    'v1_metrics': {k: round(v, 4) for k, v in v1_avg.items()},
}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Đã lưu kết quả: {OUTPUT_JSON}")

# Show sample predictions
print("\n" + "=" * 70)
print("📝 MẪU SO SÁNH CAPTION")
print("=" * 70)

import random
random.seed(42)
sample_indices = random.sample(range(len(baseline_results)), min(5, len(baseline_results)))

for idx in sample_indices:
    img = baseline_results[idx]['image_name']
    gt = gt_data.get(img, '')
    pred_b = baseline_results[idx].get('caption_vi', '')
    pred_v = v1_results[idx].get('caption_vi', '')
    
    metrics_b = compute_metrics(gt, pred_b) if gt and pred_b else {}
    metrics_v = compute_metrics(gt, pred_v) if gt and pred_v else {}
    
    print(f"\n🖼️  {img}")
    print(f"   GT:     {gt[:60]}...")
    print(f"   Baseline: {pred_b[:60]}... (BLEU-4: {metrics_b.get('bleu4', 0):.3f})")
    print(f"   V1:      {pred_v[:60]}... (BLEU-4: {metrics_v.get('bleu4', 0):.3f})")