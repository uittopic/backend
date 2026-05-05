#!/usr/bin/env python3
"""
Quick BLEU comparison across generation configs — run on 100 samples.
Tests different length_penalty, max_new_tokens, repetition_penalty combos.
"""
import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(BASE_DIR))
from app.core.accent_restoration_loader import restore_accent


def tokenize(text):
    return re.findall(r'\w+', text.lower())

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_n(ref_toks, hyp_toks, n):
    if n > len(ref_toks) or n > len(hyp_toks):
        return 0.0
    hyp_ng = get_ngrams(hyp_toks, n)
    ref_ng = get_ngrams(ref_toks, n)
    if not hyp_ng:
        return 0.0
    matches = sum(min(hyp_ng[ng], max(ref_ng.get(ng, 0), 0)) for ng in hyp_ng)
    total = sum(hyp_ng.values())
    precision = matches / total if total > 0 else 0.0
    log_p = math.log(precision) if precision > 0 else float('-inf')
    avg_log_p = log_p / n
    bp = 1.0 if len(hyp_toks) >= len(ref_toks) else math.exp(1 - len(ref_toks) / len(hyp_toks)) if len(hyp_toks) > 0 else 0.0
    return max(0.0, min(1.0, bp * math.exp(avg_log_p)))

import math

def bleu_score(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0, 0.0
    b1 = bleu_n(ref_tokens, hyp_tokens, 1)
    b2 = bleu_n(ref_tokens, hyp_tokens, 2)
    b3 = bleu_n(ref_tokens, hyp_tokens, 3)
    b4 = bleu_n(ref_tokens, hyp_tokens, 4)
    return b1, b2, b3, b4

def normalize_subword_text(text: str) -> str:
    tokens = text.strip().split()
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"):
            piece = token[2:]
            if not piece:
                continue
            if merged_tokens:
                merged_tokens[-1] = f"{merged_tokens[-1]}{piece}"
            else:
                merged_tokens.append(piece)
        else:
            merged_tokens.append(token)
    merged = " ".join(merged_tokens)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def load_model_and_processor(model_path):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(str(model_path))
    model = BlipForConditionalGeneration.from_pretrained(str(model_path))
    device = "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def run_blip(model, processor, image, device, gen_kwargs):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    caption = processor.decode(output.detach().cpu()[0], skip_special_tokens=True)
    caption = normalize_subword_text(caption)
    return caption


def test_config(model, processor, device, test_csv, image_dir, gen_kwargs, limit=100):
    gt_map = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = row.get("image", "").strip()
            caption = row.get("caption_vi") or row.get("caption") or ""
            if img and caption:
                gt_map[img] = caption.strip()

    items = list(gt_map.items())[:limit]

    b1_acc, b2_acc, b3_acc, b4_acc = 0.0, 0.0, 0.0, 0.0
    lens = []

    for img_name, gt in tqdm(items, desc="Testing"):
        img_path = image_dir / img_name
        if not img_path.exists():
            continue
        try:
            image = Image.open(img_path)
            caption_no_acc = run_blip(model, processor, image, device, gen_kwargs)
            caption_with_acc = restore_accent(caption_no_acc)

            ref_tokens = tokenize(gt)
            hyp_tokens = tokenize(caption_with_acc)
            b1, b2, b3, b4 = bleu_score(ref_tokens, hyp_tokens)
            b1_acc += b1
            b2_acc += b2
            b3_acc += b3
            b4_acc += b4
            lens.append(len(hyp_tokens))
        except Exception as e:
            print(f"Error: {e}")
            continue

    n = len(items)
    return {
        "bleu1": b1_acc / n,
        "bleu2": b2_acc / n,
        "bleu3": b3_acc / n,
        "bleu4": b4_acc / n,
        "avg_len": sum(lens) / len(lens) if lens else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/blip_vietnamese_80_20/checkpoint-15275")
    parser.add_argument("--test-csv", default="data/test_20.csv")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    MODEL_PATH = BASE_DIR / args.model_path
    TEST_CSV = BASE_DIR / args.test_csv
    IMAGE_DIR = BASE_DIR / args.images_dir

    from transformers import BlipProcessor, BlipForConditionalGeneration
    print(f"Loading model: {MODEL_PATH}")
    processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
    model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    device = "cpu"
    model.to(device)
    model.eval()
    print("Model loaded!")

    configs = [
        ("baseline",        dict(max_new_tokens=128, num_beams=5, repetition_penalty=1.1, length_penalty=1.2, early_stopping=True)),
        ("short_48",        dict(max_new_tokens=48,  num_beams=5, repetition_penalty=1.0, length_penalty=0.8, early_stopping=True)),
        ("short_32",        dict(max_new_tokens=32,  num_beams=5, repetition_penalty=1.0, length_penalty=0.6, early_stopping=True)),
        ("short_48_lp1.0", dict(max_new_tokens=48,  num_beams=5, repetition_penalty=1.0, length_penalty=1.0, early_stopping=True)),
        ("medium_64",       dict(max_new_tokens=64,  num_beams=5, repetition_penalty=1.0, length_penalty=0.8, early_stopping=True)),
    ]

    results = {}
    for name, kwargs in configs:
        print(f"\n{'='*50}")
        print(f"Testing config: {name}")
        print(f"  Params: {kwargs}")
        r = test_config(model, processor, device, TEST_CSV, IMAGE_DIR, kwargs, limit=args.limit)
        results[name] = r
        print(f"  BLEU1={r['bleu1']:.4f} BLEU4={r['bleu4']:.4f} len={r['avg_len']:.1f}")

    print(f"\n{'='*60}")
    print(f"QUICK TEST RESULTS ({args.limit} samples, {args.model_path})")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'BLEU1':>8} {'BLEU2':>8} {'BLEU3':>8} {'BLEU4':>8} {'Len':>6}")
    print("-" * 60)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['bleu4']):
        print(f"{name:<20} {r['bleu1']:>8.4f} {r['bleu2']:>8.4f} {r['bleu3']:>8.4f} {r['bleu4']:>8.4f} {r['avg_len']:>6.1f}")

    best = max(results.items(), key=lambda x: x[1]['bleu4'])
    print(f"\n🏆 Best: {best[0]} with BLEU4={best[1]['bleu4']:.4f}")

if __name__ == "__main__":
    main()
