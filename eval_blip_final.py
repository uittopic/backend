#!/usr/bin/env python3
"""
Eval BLIP Vietnamese — chỉ dùng cho test set (không train)
Dùng sau khi train_blip_final.py đã train xong

Usage:
    python eval_blip_final.py --model-path models/blip_shopee_vicaps_v1_0_1_mps
    python eval_blip_final.py --model-path models/blip_shopee_vicaps_v1_0_1_mps --test-csv data/test_clean_v3_final_noleak_filtered.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, cast

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_TEST_CSV = DATA_DIR / "test_clean_v3_final_noleak_filtered.csv"
DEFAULT_IMAGE_DIR = DATA_DIR / "images"
DEFAULT_CAPTION_COLUMN = "caption_vi_clean"

# ============================================================
# ARGPARSE
# ============================================================

parser = argparse.ArgumentParser(description="Eval BLIP Vietnamese trên test set")
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--test-csv", type=str, default=None)
parser.add_argument("--image-dir", type=str, default=None)
parser.add_argument("--caption-column", type=str, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--max-samples", type=int, default=0, help="0 = full test set")
parser.add_argument(
    "--no-accent", action="store_true",
    help="Eval trên caption không dấu (so sánh before/after accent restoration)"
)
parser.add_argument("--num-beams", type=int, default=4)
parser.add_argument("--max-length", type=int, default=40)
parser.add_argument("--repetition-penalty", type=float, default=1.2)
parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
parser.add_argument("--quiet", action="store_true")
_args = parser.parse_args()

MODEL_PATH = Path(_args.model_path)
TEST_CSV = Path(_args.test_csv) if _args.test_csv else DEFAULT_TEST_CSV
IMAGE_DIR = Path(_args.image_dir) if _args.image_dir else DEFAULT_IMAGE_DIR
CAPTION_COLUMN = _args.caption_column or DEFAULT_CAPTION_COLUMN
OUTPUT_DIR_P = Path(_args.output_dir) if _args.output_dir else (OUTPUT_DIR / MODEL_PATH.name)
OUTPUT_DIR_P.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("📊 EVAL BLIP VIETNAMESE — Test Set")
print("=" * 60)
print(f"\n📱 Device:     {DEVICE}")
print(f"🤖 Model:      {MODEL_PATH}")
print(f"📂 Test CSV:   {TEST_CSV}")
print(f"🖼️  Image dir: {IMAGE_DIR}")

t0 = time.time()
processor_loaded = BlipProcessor.from_pretrained(str(MODEL_PATH))
if isinstance(processor_loaded, tuple):
    processor_loaded = processor_loaded[0]
processor = cast(BlipProcessor, processor_loaded)

model_loaded = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
if isinstance(model_loaded, tuple):
    model_loaded = model_loaded[0]
model = cast(BlipForConditionalGeneration, model_loaded)
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded in {time.time()-t0:.1f}s")

# Generation kwargs
gen_kwargs = dict(
    max_length=_args.max_length,
    num_beams=_args.num_beams,
    repetition_penalty=_args.repetition_penalty,
    no_repeat_ngram_size=_args.no_repeat_ngram_size,
)

# ============================================================
# LOAD TEST DATA
# ============================================================

print(f"\n📂 Loading test data...")

def has_vietnamese(text: str) -> bool:
    return bool(re.compile(
        r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
        r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
        r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
        r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]"
    ).search(str(text)))

def resolve_image_path(image_value: str) -> Path:
    """Resolve path for either bare filename or nested relative paths."""
    candidate = IMAGE_DIR / image_value
    if candidate.exists():
        return candidate

    fallback = IMAGE_DIR / Path(image_value).name
    if fallback.exists():
        return fallback

    return candidate


rows = []
missing_image_count = 0
missing_caption_count = 0

with open(TEST_CSV, encoding="utf-8-sig", newline="") as f:
    header = f.readline()
    delimiter = ";" if header.count(";") >= header.count(",") else ","
    f.seek(0)

    reader = csv.DictReader(f, delimiter=delimiter)
    for row in reader:
        img = row.get("image", "").strip()
        cap = row.get(CAPTION_COLUMN, "").strip()

        if not cap:
            missing_caption_count += 1
            continue

        if not img:
            missing_image_count += 1
            continue

        img_path = resolve_image_path(img)
        if not img_path.exists():
            missing_image_count += 1
            continue

        rows.append({"image": img, "image_path": str(img_path), "caption": cap})

if _args.max_samples > 0:
    rows = rows[:_args.max_samples]

print(f"✅ Test samples: {len(rows)}")
print(f"📝 Text mode:  {'no-accent' if _args.no_accent else 'original accents'}")
if missing_caption_count > 0:
    print(f"ℹ️  Bỏ qua {missing_caption_count} dòng do caption rỗng")
if missing_image_count > 0:
    print(f"ℹ️  Bỏ qua {missing_image_count} dòng do thiếu ảnh / path ảnh sai")

if not rows:
    print("❌ Không có sample hợp lệ để eval.")
    print("   Kiểm tra lại --test-csv, --image-dir và delimiter CSV (',' hoặc ';').")
    sys.exit(1)

# ============================================================
# TOKENIZATION & METRICS
# ============================================================

def normalize_text(text: str) -> str:
    if not _args.no_accent:
        return text
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_score(ref: List[str], hyp: List[str], n: int) -> float:
    import math
    if n > len(ref) or n > len(hyp):
        return 0.0
    hyp_ng = get_ngrams(hyp, n)
    ref_ng = get_ngrams(ref, n)
    if not hyp_ng:
        return 0.0
    matches = sum(min(hyp_ng[ng], max(ref_ng.get(ng, 0), 0)) for ng in hyp_ng)
    total = sum(hyp_ng.values())
    precision = matches / total if total > 0 else 0.0
    log_p = math.log(precision) if precision > 0 else float("-inf")
    avg_log_p = log_p / n
    hyp_len = len(hyp)
    ref_len = len(ref)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    return max(0.0, min(1.0, bp * math.exp(avg_log_p)))

def rouge_l(ref: List[str], hyp: List[str]) -> float:
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1].lower() == hyp[j-1].lower():
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    return 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

# ============================================================
# INFERENCE + EVAL
# ============================================================

print(f"\n🔄 Running inference on {len(rows)} images...")
eval_t0 = time.time()
infer_t0 = eval_t0

bleu1_vals, bleu2_vals, bleu3_vals, bleu4_vals = [], [], [], []
rouge_vals = []
results = []

for i, row in enumerate(tqdm(rows, desc="Eval", unit="img", disable=_args.quiet)):
    img_path = Path(row["image_path"])
    gt = row["caption"]

    try:
        img = Image.open(img_path).convert("RGB")
        model_inputs = cast(BlipProcessor, processor)(images=img, text="", return_tensors="pt")

        pixel_values = model_inputs["pixel_values"].to(DEVICE)
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")

        generate_inputs: Dict[str, Any] = {
            "pixel_values": pixel_values,
            **gen_kwargs,
        }
        if input_ids is not None:
            generate_inputs["input_ids"] = input_ids.to(DEVICE)
        if attention_mask is not None:
            generate_inputs["attention_mask"] = attention_mask.to(DEVICE)

        with torch.no_grad():
            output = cast(BlipForConditionalGeneration, model).generate(**generate_inputs)

        caption = cast(BlipProcessor, processor).decode(output[0], skip_special_tokens=True).strip()

    except Exception as e:
        caption = ""
        print(f"⚠️  Error on {row['image']}: {e}")

    # Tokenize (optional no-accent normalization for fair Vietnamese matching)
    gt_for_eval = normalize_text(gt)
    caption_for_eval = normalize_text(caption) if caption else ""
    ref_tokens = tokenize(gt_for_eval)
    hyp_tokens = tokenize(caption_for_eval) if caption_for_eval else []

    b1 = bleu_score(ref_tokens, hyp_tokens, 1)
    b2 = bleu_score(ref_tokens, hyp_tokens, 2)
    b3 = bleu_score(ref_tokens, hyp_tokens, 3)
    b4 = bleu_score(ref_tokens, hyp_tokens, 4)
    rl = rouge_l(ref_tokens, hyp_tokens)

    bleu1_vals.append(b1)
    bleu2_vals.append(b2)
    bleu3_vals.append(b3)
    bleu4_vals.append(b4)
    rouge_vals.append(rl)

    results.append({
        "image": row["image"],
        "ground_truth": gt,
        "prediction": caption,
        "bleu1": round(b1, 4),
        "bleu2": round(b2, 4),
        "bleu3": round(b3, 4),
        "bleu4": round(b4, 4),
        "rouge_l": round(rl, 4),
    })

    if (i + 1) % 100 == 0:
        elapsed = time.time() - infer_t0
        avg_b1 = sum(bleu1_vals) / len(bleu1_vals)
        eta = (elapsed / (i + 1)) * (len(rows) - i - 1)
        print(f"   [{i+1}/{len(rows)}] BLEU-1: {avg_b1:.4f} | ETA: {eta/60:.1f}min")

infer_elapsed = time.time() - infer_t0

# ============================================================
# METRICS SUMMARY
# ============================================================

def avg(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0

stats = {
    "bleu1": round(avg(bleu1_vals), 4),
    "bleu2": round(avg(bleu2_vals), 4),
    "bleu3": round(avg(bleu3_vals), 4),
    "bleu4": round(avg(bleu4_vals), 4),
    "rouge_l": round(avg(rouge_vals), 4),
}

# ============================================================
# SBERT SEMANTIC SIMILARITY
# ============================================================
print(f"\n🔄 Computing SBERT semantic similarity...")

sbert_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
sbert_device = "mps" if torch.backends.mps.is_available() else "cpu"
sbert = SentenceTransformer(sbert_model_name, device=sbert_device)

refs_list = [r["ground_truth"] for r in results]
preds_list = [r["prediction"] for r in results]

# Original SBERT (with accents)
e_refs = sbert.encode(refs_list, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
e_preds = sbert.encode(preds_list, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
sbert_cosine = (e_refs * e_preds).sum(dim=1).cpu().numpy()

# No-accent SBERT (normalize both sides before embedding)
def rm_accent(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

refs_na = [rm_accent(r) for r in refs_list]
preds_na = [rm_accent(p) for p in preds_list]
e_refs_na = sbert.encode(refs_na, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
e_preds_na = sbert.encode(preds_na, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
sbert_cosine_na = (e_refs_na * e_preds_na).sum(dim=1).cpu().numpy()

stats["sbert_cosine"] = round(float(sbert_cosine.mean()), 4)
stats["sbert_cosine_na"] = round(float(sbert_cosine_na.mean()), 4)

for i, r in enumerate(results):
    r["sbert_cosine"] = round(float(sbert_cosine[i]), 4)
    r["sbert_cosine_na"] = round(float(sbert_cosine_na[i]), 4)

print(f"  SBERT (original):   {stats['sbert_cosine']:.4f}")
print(f"  SBERT (no-accent):  {stats['sbert_cosine_na']:.4f}")

total_elapsed = time.time() - eval_t0

print()
print("=" * 60)
print("📊 KẾT QUẢ EVAL TRÊN TEST SET")
print("=" * 60)
print(f"Model:  {MODEL_PATH}")
print(f"Test:   {len(rows)} samples")
time_per_img = (total_elapsed / len(rows)) if rows else 0.0
print(f"Time:   {total_elapsed/60:.1f} min ({time_per_img:.1f}s/img)")
print(f"  - Inference: {infer_elapsed/60:.1f} min")
print(f"  - SBERT:     {(total_elapsed-infer_elapsed)/60:.1f} min")
print()
print(f"  BLEU-1:  {stats['bleu1']:.4f}  ({stats['bleu1']*100:.2f}%)")
print(f"  BLEU-2:  {stats['bleu2']:.4f}  ({stats['bleu2']*100:.2f}%)")
print(f"  BLEU-3:  {stats['bleu3']:.4f}  ({stats['bleu3']*100:.2f}%)")
print(f"  BLEU-4:  {stats['bleu4']:.4f}  ({stats['bleu4']*100:.2f}%)")
print(f"  ROUGE-L: {stats['rouge_l']:.4f}  ({stats['rouge_l']*100:.2f}%)")
print(f"  SBERT (original):   {stats['sbert_cosine']:.4f}  ({stats['sbert_cosine']*100:.2f}%)")
print(f"  SBERT (no-accent): {stats['sbert_cosine_na']:.4f}  ({stats['sbert_cosine_na']*100:.2f}%)")

# BLEU-4 distribution
print(f"\n  BLEU-4 distribution:")
ranges = [("=0", 0.0, 0.0), ("0-0.1", 0.001, 0.1), ("0.1-0.2", 0.1, 0.2),
          ("0.2-0.3", 0.2, 0.3), ("0.3-0.4", 0.3, 0.4), (">0.4", 0.4, 99.0)]
for label, lo, hi in ranges:
    if label == "=0":
        count = sum(1 for v in bleu4_vals if v == 0.0)
    else:
        count = sum(1 for v in bleu4_vals if lo <= v < hi)
    pct = 100 * count / len(bleu4_vals) if bleu4_vals else 0.0
    bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
    print(f"    {label:<8}|{bar}| {count:>3} ({pct:.1f}%)")

# Sample predictions
print(f"\n  Sample predictions:")
for r in results[:3]:
    print(f"    GT:      {r['ground_truth'][:55]}")
    print(f"    Pred:    {r['prediction'][:55]}")
    print(f"    B1={r['bleu1']:.3f} B4={r['bleu4']:.3f} R={r['rouge_l']:.3f}")
    print()

# ============================================================
# SAVE RESULTS
# ============================================================

# JSON summary
summary_path = OUTPUT_DIR_P / "test_eval_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": str(MODEL_PATH),
        "test_csv": str(TEST_CSV),
        "test_samples": len(rows),
        "elapsed_min": round(total_elapsed / 60, 2),
        "inference_elapsed_min": round(infer_elapsed / 60, 2),
        "sbert_elapsed_min": round((total_elapsed - infer_elapsed) / 60, 2),
        "generation_kwargs": gen_kwargs,
        "no_accent_eval": _args.no_accent,
        "metrics": stats,
    }, f, ensure_ascii=False, indent=2)

# CSV per-image
csv_path = OUTPUT_DIR_P / "test_eval_detail.csv"
fieldnames = ["image", "ground_truth", "prediction", "bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "sbert_cosine", "sbert_cosine_na"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Summary:  {summary_path}")
print(f"💾 Detail:   {csv_path}")
print("=" * 60)
