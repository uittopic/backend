#!/usr/bin/env python3
"""
Eval Direct — Inference + Evaluate trong 1 script
====================================================

Không cần batch test, không cần API, không cần file trung gian.

Flow:
  1. Load model từ local path
  2. Chạy inference trên test set
  3. Tính BLEU, ROUGE-L, METEOR
  4. Hiển thị sample predictions

Usage:
  python tools/eval_direct.py                          # 50 mẫu
  python tools/eval_direct.py --max-samples 20          # 20 mẫu
  python tools/eval_direct.py --model-path models/blip_vietnamese_cleaned_v1
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

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
TEST_CSV = DATA_DIR / "test_20.csv"

import sys
sys.path.insert(0, str(BASE))


# ============================================================
# METRICS
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
        'bleu4_nodiac': bleu_score(tokenize(remove_diacritics(ref)), tokenize(remove_diacritics(hyp)), 4),
        'rouge_l': rouge_l(ref_tokens, hyp_tokens),
        'is_vietnamese': has_vietnamese(hyp),
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data(test_csv_path, max_samples=0, seed=42):
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
        items = random.sample(items, min(max_samples, len(items)))
    return dict(items)


# ============================================================
# INFERENCE
# ============================================================

def load_model(model_path):
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    path = Path(model_path)
    if path.exists() and any(path.iterdir()):
        print(f"  Loading from: {path}")
        processor = BlipProcessor.from_pretrained(str(path))
        model = BlipForConditionalGeneration.from_pretrained(str(path))
        return model, processor
    else:
        print(f"  Model not found at {path}")
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
        output = model.generate(pixel_values=inputs["pixel_values"], **config)
        # output = model.generate(**inputs, **config)

    caption = processor.decode(output[0], skip_special_tokens=True)
    del inputs, output
    return caption.strip()


def restore_accent_from_path(text_no_accent: str,
                            accent_tokenizer_ref=None,
                            accent_model_ref=None,
                            label_list_ref=None,
                            device_ref=None):
    """
    Restore diacritics WITHOUT importing from app (avoids module load side-effects).
    Re-implements the accent restoration logic inline.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    import numpy as np
    import time as _time

    ACCENT_MODEL_NAME = "peterhung/vietnamese-accent-marker-xlm-roberta"

    if accent_model_ref is None:
        return text_no_accent, 0.0, None

    at = accent_tokenizer_ref
    am = accent_model_ref
    ll = label_list_ref

    def merge(tokens, preds):
        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            lbls = {int(preds[i])}
            if tok.startswith("▁"):
                tok2 = tok[1:]
                cur = [tok2] if tok2 else []
                j = i + 1
                while j < len(tokens) and not tokens[j].startswith("▁"):
                    cur.append(tokens[j])
                    lbls.add(int(preds[j]))
                    j += 1
                word = tok2 if tok2 else (''.join(cur) if cur else tok)
                result.append((word, lbls))
                i = j
            else:
                if tok not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                    result.append((tok, lbls))
                i += 1
        return result

    def apply_accents(merged, labels):
        out = []
        for word, lbls in merged:
            if not word:
                out.append(word)
                continue
            best, best_len = None, 0
            for li in lbls:
                if li < len(labels):
                    tag = labels[int(li)]
                    if "-" in tag:
                        raw, vowel = tag.split("-", 1)
                        if raw and len(raw) >= best_len and raw in word:
                            best = vowel
                            best_len = len(raw)
            out.append(word.replace(word[:best_len], best, 1) if best else word)
        return out

    try:
        tokens_in = text_no_accent.strip().split()
        if not tokens_in:
            return text_no_accent, 0.0, None
        t0 = _time.time()
        inputs = at(tokens_in, is_split_into_words=True, truncation=True,  # type: ignore
                    padding=True, max_length=512, return_tensors="pt").to(device_ref)
        with torch.no_grad():
            outputs = am(**inputs)  # type: ignore
        logits = outputs.logits
        preds = np.argmax(logits.cpu().numpy(), axis=2)[0]
        tok_list = at.convert_ids_to_tokens(inputs['input_ids'][0])  # type: ignore
        del inputs, outputs
        if len(tok_list) > 2:
            tok_list = tok_list[1:-1]
            preds = preds[1:-1]
        merged = merge(tok_list, preds)
        accented = ' '.join(apply_accents(merged, ll))
        elapsed = _time.time() - t0
        return accented.strip(), elapsed, None
    except Exception as e:
        return text_no_accent, 0.0, str(e)


# ============================================================
# MAIN
# ============================================================

def main():
    import torch

    parser = argparse.ArgumentParser(description='Eval Direct — Inference + Evaluate')
    parser.add_argument('--test-csv', default=str(TEST_CSV))
    parser.add_argument('--model-path',
                        default=str(MODEL_DIR / "blip_vietnamese_cleaned_v1"))
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Also run accent restoration model (BLIP -> accent)')
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("EVAL DIRECT — Inference + Evaluate (1 script)")
    print("=" * 60)
    print(f"  Model:    {args.model_path}")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Samples:  {args.max_samples}")
    print(f"  Full pipeline: {args.full_pipeline}")
    print(f"  NOTE: BLEU/ROUGE la simplified version (khong dung sacreBLEU)")
    print(f"  NOTE: Tokenization don gian (\\w+) — chua tot nhu word-level tokenization")

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
    test_data = load_test_data(args.test_csv, args.max_samples, args.seed)
    print(f"\n  Test samples: {len(test_data)}")

    # Device
    DEVICE = "cpu"
    device_obj = torch.device(DEVICE)
    print(f"  Device: {DEVICE}")

    # Load model
    print("\n  Loading model...")
    model, processor = load_model(args.model_path)
    if model is None:
        print("  Model load failed!")
        return

    model.to(device_obj)  # type: ignore
    model.eval()
    IMAGE_DIR = DATA_DIR / "images"

    # Load accent model (optional)
    accent_tok, accent_mod, label_list_accent = None, None, None
    accent_latencies = []
    if args.full_pipeline:
        print("\n  Loading accent restoration model...")
        from transformers import AutoTokenizer as ATok, AutoModelForTokenClassification as AMTC
        from app.core.config import ACCENT_MODEL_NAME
        try:
            accent_tok = ATok.from_pretrained(ACCENT_MODEL_NAME, add_prefix_space=True)
            accent_mod = AMTC.from_pretrained(ACCENT_MODEL_NAME)
            accent_mod.to(device_obj)
            accent_mod.eval()
            # Load labels
            from huggingface_hub import hf_hub_download
            tags_file = hf_hub_download(repo_id=ACCENT_MODEL_NAME, filename="selected_tags_names.txt")
            with open(tags_file, encoding='utf-8') as f:
                label_list_accent = [l.strip() for l in f if l.strip()]
            print(f"  Accent model loaded ({len(label_list_accent)} labels)")
        except Exception as e:
            print(f"  Accent model load failed: {e}")
            args.full_pipeline = False

    # Run inference
    print(f"\n  Running inference on {len(test_data)} images...")
    results = []
    vi_count = 0
    en_count = 0
    latency_times = []
    accent_issues = 0
    accent_latencies = []

    import time
    for i, (img, gt) in enumerate(test_data.items()):
        t0 = time.time()
        pred = run_inference(img, INFER_CONFIG, model, processor, device_obj, IMAGE_DIR)
        elapsed = time.time() - t0

        # Full pipeline: restore accents
        pred_acc = pred
        if pred and args.full_pipeline and accent_mod is not None:
            pred_acc, acc_elapsed, acc_err = restore_accent_from_path(
                pred, accent_tok, accent_mod, label_list_accent, device_obj
            )
            if acc_elapsed is not None:
                accent_latencies.append(acc_elapsed)
            else:
                accent_latencies.append(0.0)
            if acc_err:
                accent_issues += 1
            elapsed += acc_elapsed

        latency_times.append(elapsed)
        if pred:
            metrics = compute_metrics(gt, pred_acc if args.full_pipeline else pred)
            metrics['image'] = img
            metrics['ground_truth'] = gt
            metrics['prediction'] = pred
            metrics['prediction_acc'] = pred_acc if args.full_pipeline else None
            metrics['latency_s'] = round(elapsed, 3)
            results.append(metrics)
            check_pred = pred_acc if args.full_pipeline else pred
            if metrics['is_vietnamese']:
                vi_count += 1
            else:
                en_count += 1
        if (i + 1) % 10 == 0:
            print(f"    Done {i+1}/{len(test_data)}")

    n = len(results)
    if n == 0:
        print("  No results!")
        return

    # ============================================================
    # RESULTS
    # ============================================================

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    vi_pct = 100 * vi_count / n

    print("\n" + "=" * 60)
    print("KET QUA DANH GIA")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Gia tri':<15}")
    print("-" * 40)
    print(f"{'BLEU-1':<20} {avg(results,'bleu1'):.4f} ({avg(results,'bleu1')*100:.2f}%)")
    print(f"{'BLEU-2':<20} {avg(results,'bleu2'):.4f} ({avg(results,'bleu2')*100:.2f}%)")
    print(f"{'BLEU-3':<20} {avg(results,'bleu3'):.4f} ({avg(results,'bleu3')*100:.2f}%)")
    print(f"{'BLEU-4':<20} {avg(results,'bleu4'):.4f} ({avg(results,'bleu4')*100:.2f}%)")
    print(f"{'BLEU-4 (nodiac)':<20} {avg(results,'bleu4_nodiac'):.4f} ({avg(results,'bleu4_nodiac')*100:.2f}%)")
    print(f"{'ROUGE-L':<20} {avg(results,'rouge_l'):.4f} ({avg(results,'rouge_l')*100:.2f}%)")
    print(f"{'Tieng Viet':<20} {vi_count}/{n} ({vi_pct:.1f}%)")
    print(f"{'Tieng Anh':<20} {en_count}/{n} ({100-vi_pct:.1f}%)")

    # Latency
    if latency_times:
        print(f"\n{'Latency (s)':<20}")
        print("-" * 40)
        avg_lat = sum(latency_times) / len(latency_times)
        print(f"  Avg:    {avg_lat:.3f}s")
        print(f"  Min:    {min(latency_times):.3f}s")
        print(f"  Max:    {max(latency_times):.3f}s")
        print(f"  Total:  {sum(latency_times):.2f}s  ({len(latency_times)} images)")
        if args.full_pipeline and accent_latencies:
            avg_acc = sum(accent_latencies) / len(accent_latencies)
            print(f"  Accent avg: {avg_acc:.3f}s ({accent_issues} errors)")

    # BLEU-4 distribution
    print("\n" + "=" * 60)
    print("BLEU-4 Distribution")
    print("=" * 60)
    bleu_vals = [r['bleu4'] for r in results]
    ranges = [
        ('= 0.000', 0.0, 0.0),
        ('0.00 - 0.10', 0.001, 0.1),
        ('0.10 - 0.20', 0.1, 0.2),
        ('0.20 - 0.30', 0.2, 0.3),
        ('0.30 - 0.40', 0.3, 0.4),
        ('> 0.40', 0.4, 99.0),
    ]
    for label, lo, hi in ranges:
        count = sum(1 for v in bleu_vals if lo <= v < hi)
        pct = 100 * count / n
        bar = '#' * int(pct / 3) + '-' * (33 - int(pct / 3))
        print(f"  {label:<15} | {bar} | {count:>3} ({pct:.1f}%)")

    # Vietnamese vs English predictions
    print("\n" + "=" * 60)
    print("NGON NGU PREDICTIONS")
    print("=" * 60)
    vi_results = [r for r in results if r['is_vietnamese']]
    en_results = [r for r in results if not r['is_vietnamese']]
    print(f"  Tieng Viet: {len(vi_results)} ({100*len(vi_results)/n:.1f}%)")
    print(f"  Tieng Anh:   {len(en_results)} ({100*len(en_results)/n:.1f}%)")

    if vi_results:
        avg_vi = sum(r['bleu4'] for r in vi_results) / len(vi_results)
        print(f"  BLEU-4 (tieng Viet): {avg_vi:.4f}")

    # Sample predictions
    print("\n" + "=" * 60)
    print("MAU PREDICTIONS (10 mau ngau nhien)")
    print("=" * 60)
    samples = random.sample(results, min(10, len(results)))
    for i, r in enumerate(samples, 1):
        lang = "VI" if r['is_vietnamese'] else "EN"
        print(f"\n[{i}] {r['image'][:40]} [{lang}]")
        print(f"    GT:   {r['ground_truth'][:70]}")
        print(f"    Pred: {r['prediction'][:70]}")
        if args.full_pipeline and r.get('prediction_acc'):
            print(f"    Acc:  {r['prediction_acc'][:70]}")
        print(f"    B4: {r['bleu4']:.3f} | B4_nodiac: {r['bleu4_nodiac']:.3f} | R-L: {r['rouge_l']:.3f}")

    # Save JSON
    OUTPUT = BASE / "outputs" / "eval_direct_results.json"
    OUTPUT.parent.mkdir(exist_ok=True)
    output_data = {
        'model_path': str(args.model_path),
        'full_pipeline': args.full_pipeline,
        'n_samples': n,
        'seed': args.seed,
        'metrics': {
            'bleu1': round(avg(results, 'bleu1'), 4),
            'bleu2': round(avg(results, 'bleu2'), 4),
            'bleu3': round(avg(results, 'bleu3'), 4),
            'bleu4': round(avg(results, 'bleu4'), 4),
            'bleu4_nodiac': round(avg(results, 'bleu4_nodiac'), 4),
            'rouge_l': round(avg(results, 'rouge_l'), 4),
        },
        'language_stats': {
            'vietnamese': len(vi_results),
            'english': len(en_results),
            'vi_pct': round(vi_pct, 1),
        },
        'latency': {
            'avg_s': round(avg_lat, 4) if latency_times else 0.0,
            'min_s': round(min(latency_times), 4) if latency_times else 0.0,
            'max_s': round(max(latency_times), 4) if latency_times else 0.0,
            'total_s': round(sum(latency_times), 2) if latency_times else 0.0,
        },
        'accent_model': {
            'used': args.full_pipeline,
            'avg_latency_s': round(sum(accent_latencies)/len(accent_latencies), 4) if accent_latencies else 0.0,
            'errors': accent_issues,
        } if args.full_pipeline else None,
        'limitations': [
            'BLEU/ROUGE la simplified implementation (khong dung sacreBLEU)',
            'Tokenization don gian \\w+ — chua tot nhu word-level tokenizer',
            'BLEU khong phan anh day du chat luong captioning',
            'Ket qua co the khac voi sacreBLEU chuan',
        ],
        'sample_predictions': results[:20],
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {OUTPUT}")

    # ============================================================
    # RECOMMENDATION
    # ============================================================
    print("\n" + "=" * 60)
    print("CHAN DOAN & KHUYEN NGHI")
    print("=" * 60)

    if vi_pct < 50:
        print("  Model generate TIENG ANH ({:.1f}%)".format(100-vi_pct))
        print("  Nguyen nhan: Training chua du toc hoac text prompt khong dung")
        print("  Giai phap:")
        print("    1. Train lai voi training script co text prompt")
        print("    2. Tang epoch: 3 -> 8")
        print("    3. Dung data cleaned tot hon")
    elif avg(results, 'bleu4') < 0.1:
        print("  Model generate tieng Viet nhung BLEU thap")
        print("  Nguyen nhan: Caption prediction khac ground truth")
        print("  Giai phap:")
        print("    1. BLEU khong phai metric tot nhat cho captioning")
        print("    2. Xem sample predictions - neu caption HOP LY thi OK")
        print("    3. Co the ground truth khong dong nhat")
    elif avg(results, 'bleu4') > 0.3:
        print("  Model hoat dong TOT!")
        print("  BLEU-4 > 0.3, co the dung trong production")
    else:
        print("  Model hoat dong o muc CHAP NHAN DUOC")
        print("  BLEU-4 0.1-0.3: Can cai thien them")

    print("=" * 60)


if __name__ == '__main__':
    main()
