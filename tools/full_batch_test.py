#!/usr/bin/env python3
"""
Full Batch Test with Incremental Save & Metrics
===============================================
- Test ALL images in data/images/ via API
- Incremental save: save after every batch (resume on crash)
- Compute BLEU, ROUGE-L metrics on test_20.csv
- Compare models if needed

Usage:
    python tools/full_batch_test.py                    # Full test (resume if exists)
    python tools/full_batch_test.py --force            # Restart from scratch
    python tools/full_batch_test.py --limit 100       # Test subset first
    python tools/full_batch_test.py --metrics-only     # Skip inference, compute metrics only
"""

import argparse
import csv
import json
import os
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import requests

# ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

API_URL = "http://localhost:8000/api/caption_full"
IMAGES_DIR = BASE / "data" / "images"
OUTPUT_DIR = BASE / "outputs" / "full_batch_test"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_results.json"
METRICS_CSV = OUTPUT_DIR / "full_metrics_summary.csv"
TEST_CSV = BASE / "data" / "test_20.csv"
API_TIMEOUT = 30

# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

def tokenize(text):
    import re
    return re.findall(r'\w+', text.lower())

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_score(reference, hypothesis, n=4):
    import math
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

def compute_metrics(ref, hyp):
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    return {
        'bleu1': bleu_score(ref_tokens, hyp_tokens, 1),
        'bleu2': bleu_score(ref_tokens, hyp_tokens, 2),
        'bleu3': bleu_score(ref_tokens, hyp_tokens, 3),
        'bleu4': bleu_score(ref_tokens, hyp_tokens, 4),
        'rouge_l': rouge_l(ref_tokens, hyp_tokens),
    }

# ──────────────────────────────────────────────────────────────
# API TEST
# ──────────────────────────────────────────────────────────────

def test_image_via_api(image_path: Path, api_url: str = API_URL) -> dict:
    result = {
        "image_name": image_path.name,
        "success": False,
        "caption_vi": None,
        "caption_vi_no_accent": None,
        "processing_time": None,
        "error": None,
    }
    start_time = time.time()
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=API_TIMEOUT)
        elapsed = time.time() - start_time
        result["processing_time"] = round(elapsed, 2)
        if response.status_code == 200:
            data = response.json()
            result["success"] = data.get("success", False)
            result["caption_vi"] = data.get("caption_vi")
            result["caption_vi_no_accent"] = data.get("caption_vi_no_accent")
        else:
            result["error"] = f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        result["error"] = f"Timeout {API_TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        result["error"] = "Cannot connect to API"
    except Exception as e:
        result["error"] = str(e)[:100]
    return result

def get_all_images(images_dir: Path) -> list:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    images = []
    for ext in extensions:
        images.extend(images_dir.glob(f"*{ext}"))
        images.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))

# ──────────────────────────────────────────────────────────────
# LOAD / SAVE CHECKPOINT
# ──────────────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        completed = {r["image_name"] for r in data.get("results", [])}
        print(f"📂 Resume checkpoint: {len(completed)}/{data.get('total', '?')} images already done")
        return data, completed
    return None, set()

def save_checkpoint(results, all_images_count, errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(results)
    success = sum(1 for r in results if r["success"])
    # Incremental save - also save CSV each time
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total": all_images_count,
            "completed": total,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "errors": errors,
        }, f, ensure_ascii=False, indent=2)

    csv_path = OUTPUT_DIR / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "caption_vi", "caption_vi_no_accent", "success", "processing_time", "error"])
        for r in results:
            writer.writerow([
                r["image_name"],
                r.get("caption_vi") or "",
                r.get("caption_vi_no_accent") or "",
                r["success"],
                r.get("processing_time") or "",
                r.get("error") or "",
            ])
    print(f"  💾 Checkpoint saved ({total}/{all_images_count}) → {csv_path}")

# ──────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────

def load_test_gt():
    gt = {}
    if TEST_CSV.exists():
        with open(TEST_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img = row.get("image", "")
                cap = row.get("caption_vi", "")
                if img and cap:
                    gt[img] = cap
    return gt

def compute_metrics_summary(results, gt_data):
    matched = []
    for r in results:
        if r["success"] and r["caption_vi"] and r["image_name"] in gt_data:
            m = compute_metrics(gt_data[r["image_name"]], r["caption_vi"])
            m["image"] = r["image_name"]
            m["ground_truth"] = gt_data[r["image_name"]]
            m["prediction"] = r["caption_vi"]
            matched.append(m)

    if not matched:
        print("  ⚠️  No images match test_20.csv — metrics skipped")
        return None

    def avg(lst, key):
        return sum(x[key] for x in lst) / len(lst)

    stats = {
        "total_matched": len(matched),
        "bleu1": round(avg(matched, "bleu1"), 4),
        "bleu2": round(avg(matched, "bleu2"), 4),
        "bleu3": round(avg(matched, "bleu3"), 4),
        "bleu4": round(avg(matched, "bleu4"), 4),
        "rouge_l": round(avg(matched, "rouge_l"), 4),
    }

    # BLEU-4 distribution
    bleu_vals = [r["bleu4"] for r in matched]
    ranges = [("=0", 0.0, 0.0), ("0-0.1", 0.001, 0.1), ("0.1-0.2", 0.1, 0.2),
              ("0.2-0.3", 0.2, 0.3), ("0.3-0.4", 0.3, 0.4), (">0.4", 0.4, 99.0)]
    dist_lines = []
    for label, lo, hi in ranges:
        count = sum(1 for v in bleu_vals if lo <= v < hi)
        pct = 100 * count / len(matched)
        bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
        dist_lines.append(f"  {label:<10}|{bar}| {count:>3} ({pct:.1f}%)")

    # Save metrics CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in stats.items():
            writer.writerow({"metric": k, "value": v})
    # Save detailed predictions
    detailed_path = OUTPUT_DIR / "detailed_predictions.csv"
    with open(detailed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth", "prediction", "bleu1", "bleu2", "bleu3", "bleu4", "rouge_l"])
        for r in matched:
            writer.writerow([
                r["image"], r["ground_truth"], r["prediction"],
                round(r["bleu1"], 4), round(r["bleu2"], 4), round(r["bleu3"], 4),
                round(r["bleu4"], 4), round(r["rouge_l"], 4),
            ])

    print(f"  💾 Metrics saved: {METRICS_CSV}")
    print(f"  💾 Detailed saved: {detailed_path}")

    return stats, dist_lines, matched[:5]

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full batch test with incremental save")
    parser.add_argument("--force", action="store_true", help="Restart from scratch (ignore checkpoint)")
    parser.add_argument("--limit", type=int, default=None, help="Limit images (for quick test)")
    parser.add_argument("--metrics-only", action="store_true", help="Skip inference, compute metrics from checkpoint")
    parser.add_argument("--batch-save", type=int, default=50, help="Save checkpoint every N images")
    args = parser.parse_args()

    print("=" * 60)
    print("FULL BATCH TEST — Incremental Save & Metrics")
    print("=" * 60)
    print(f"📁 Images:      {IMAGES_DIR}")
    print(f"🔗 API:        {API_URL}")
    print(f"📂 Output:     {OUTPUT_DIR}")
    print(f"💾 Checkpoint: {CHECKPOINT_FILE}")
    print()

    # Check API
    try:
        r = requests.get("http://localhost:8000/api/health", timeout=5)
        health = r.json()
        print(f"✅ API online: BLIP={health.get('blip_model_loaded')}, Accent={health.get('accent_model_loaded')}")
    except:
        print("❌ Cannot connect to API. Start server first:")
        print("   .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return

    if args.metrics_only:
        checkpoint_data, _ = load_checkpoint()
        if not checkpoint_data:
            print("❌ No checkpoint found. Run without --metrics-only first.")
            return
        results = checkpoint_data.get("results", [])
        print(f"\n📊 Computing metrics for {len(results)} results...")
        gt_data = load_test_gt()
        output = compute_metrics_summary(results, gt_data)
        if output:
            stats, dist, samples = output
            print(f"\n{'='*60}")
            print(f"📊 METRICS (vs test_20.csv, n={stats['total_matched']})")
            print(f"{'='*60}")
            print(f"  BLEU-1:  {stats['bleu1']:.4f} ({stats['bleu1']*100:.2f}%)")
            print(f"  BLEU-2:  {stats['bleu2']:.4f} ({stats['bleu2']*100:.2f}%)")
            print(f"  BLEU-3:  {stats['bleu3']:.4f} ({stats['bleu3']*100:.2f}%)")
            print(f"  BLEU-4:  {stats['bleu4']:.4f} ({stats['bleu4']*100:.2f}%)")
            print(f"  ROUGE-L: {stats['rouge_l']:.4f} ({stats['rouge_l']*100:.2f}%)")
            print(f"\n  BLEU-4 Distribution:")
            for line in dist:
                print(line)
        return

    # Load all images
    all_images = get_all_images(IMAGES_DIR)
    if not all_images:
        print("❌ No images found!")
        return
    if args.limit:
        all_images = all_images[:args.limit]
    total = len(all_images)
    print(f"✅ Total images: {total}")

    # Load checkpoint or start fresh
    checkpoint_data, completed_names = load_checkpoint()
    results = []
    errors = []

    if checkpoint_data and not args.force:
        results = checkpoint_data.get("results", [])
        errors = checkpoint_data.get("errors", [])
        print(f"▶️  Resuming from checkpoint: {len(results)} already done")
    else:
        if args.force:
            print("⚠️  --force: Starting from scratch")

    # Filter remaining images
    remaining = [img for img in all_images if img.name not in completed_names]
    done_count = total - len(remaining)
    print(f"📊 To do: {len(remaining)} / {total} images")
    print(f"💾 Auto-save every {args.batch_save} images")
    print()

    if not remaining:
        print("🎉 All images already done!")
    else:
        print(f"🚀 Starting inference on {len(remaining)} images...")
        print()

        start_time = time.time()
        for i, img_path in enumerate(tqdm(remaining, desc="Inferencing", unit="img")):
            result = test_image_via_api(img_path)
            results.append(result)

            if result["error"]:
                errors.append({"image": img_path.name, "error": result["error"]})

            # Incremental save
            done_now = done_count + i + 1
            if done_now % args.batch_save == 0 or done_now == total:
                elapsed_total = time.time() - start_time
                avg_time = elapsed_total / done_now
                eta = avg_time * (total - done_now)
                print(f"\n  📊 Progress: {done_now}/{total} ({done_now/total*100:.1f}%) | ETA: {eta/60:.1f} min")
                save_checkpoint(results, total, errors)

        total_time = time.time() - start_time
        print(f"\n✅ Inference complete in {total_time/60:.1f} min")

    # Final stats
    success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"📊 INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:   {len(results)}")
    print(f"  Success: {success} ({success/len(results)*100:.1f}%)")
    if errors:
        print(f"  Errors:  {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e['image']}: {e['error']}")

    # Compute metrics
    print(f"\n{'='*60}")
    print(f"📊 COMPUTING METRICS (vs test_20.csv)")
    print(f"{'='*60}")
    gt_data = load_test_gt()
    matched = sum(1 for r in results if r["success"] and r["image_name"] in gt_data)
    print(f"  Matched {matched} images with test_20.csv")

    output = compute_metrics_summary(results, gt_data)
    if output:
        stats, dist, samples = output
        print(f"\n  BLEU-1:  {stats['bleu1']:.4f} ({stats['bleu1']*100:.2f}%)")
        print(f"  BLEU-2:  {stats['bleu2']:.4f} ({stats['bleu2']*100:.2f}%)")
        print(f"  BLEU-3:  {stats['bleu3']:.4f} ({stats['bleu3']*100:.2f}%)")
        print(f"  BLEU-4:  {stats['bleu4']:.4f} ({stats['bleu4']*100:.2f}%)")
        print(f"  ROUGE-L: {stats['rouge_l']:.4f} ({stats['rouge_l']*100:.2f}%)")
        print(f"\n  BLEU-4 Distribution:")
        for line in dist:
            print(line)

        print(f"\n  Sample predictions:")
        for s in samples:
            print(f"    GT:  {s['ground_truth'][:55]}")
            print(f"    Pred: {s['prediction'][:55]}")
            print(f"    B4={s['bleu4']:.3f} R={s['rouge_l']:.3f}")
            print()

    # Final save
    save_checkpoint(results, total, errors)
    print(f"\n✅ ALL DONE! Results at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
