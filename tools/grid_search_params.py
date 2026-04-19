#!/usr/bin/env python3
"""
Grid Search Generation Parameters - KHỚP PIPELINE THẬT
Pipeline: BLIP → Accent Restoration → BLEU (có dấu)

Chạy nhanh trên 50 sample để tìm config tốt nhất,
sau đó chạy full test (1514 sample) với config thắng.

Cách dùng:
  python tools/grid_search_params.py
  python tools/grid_search_params.py --model models/blip_vietnamese_cleaned_v1 --max-samples 50
"""
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

BASE_DIR = Path(__file__).parent.parent

# CLI args
parser = argparse.ArgumentParser(description="Grid search generation params - khớp pipeline thật")
parser.add_argument("--model", type=str,
                    default="models/blip_vietnamese_cleaned_v1",
                    help="Đường dẫn model (relative to BASE_DIR)")
parser.add_argument("--test-csv", type=str,
                    default="data/test_20.csv",
                    help="File test CSV")
parser.add_argument("--image-dir", type=str,
                    default="data/images",
                    help="Thư mục ảnh")
parser.add_argument("--max-samples", type=int, default=50,
                    help="Số samples grid search (mặc định 50)")
parser.add_argument("--configs", type=int, default=None,
                    help="Số configs grid search muốn chạy (mặc định: tất cả)")
_args = parser.parse_args()

MODEL_PATH = BASE_DIR / _args.model
TEST_CSV = BASE_DIR / _args.test_csv
IMAGE_DIR = BASE_DIR / _args.image_dir

# Import pipeline
import sys
sys.path.insert(0, str(BASE_DIR))
from transformers import BlipProcessor, BlipForConditionalGeneration
from app.core.accent_restoration_loader import restore_accent


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path: Path):
    device = get_device()
    print(f"📱 Device: {device}")

    if model_path.exists() and any(model_path.iterdir()):
        print(f"📦 Load từ: {model_path}")
        processor = BlipProcessor.from_pretrained(str(model_path))
        model = BlipForConditionalGeneration.from_pretrained(str(model_path))
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.to(device)
    model.eval()
    return model, processor, device


# ============================================================
# GENERATION - khớp caption_service.py
# ============================================================

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
    return re.sub(r"\s+", " ", " ".join(merged_tokens)).strip()


def looks_broken(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or len(cleaned) <= 2:
        return True
    if "##" in cleaned:
        return True
    if not re.search(r"[A-Za-z0-9À-ỹ]", cleaned):
        return True
    return False


def generate_blip(model, processor, image: Image.Image, device: str,
                   gen_kwargs: Dict[str, Any]) -> str:
    """Generate caption từ BLIP với gen_kwargs tùy ý"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    if device == "mps":
        model_cpu = model.cpu()
        inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
    else:
        model_cpu = model
        inputs_cpu = inputs

    with torch.no_grad():
        output = model_cpu.generate(**inputs_cpu, **gen_kwargs)

    output_cpu = output.detach().cpu()
    caption = processor.decode(output_cpu[0], skip_special_tokens=True)
    caption = normalize_subword_text(caption)

    if looks_broken(caption):
        fallback_kwargs = dict(gen_kwargs)
        fallback_kwargs["num_beams"] = 1
        fallback_kwargs["repetition_penalty"] = 1.0
        fallback_kwargs.pop("no_repeat_ngram_size", None)
        fallback_kwargs.pop("length_penalty", None)

        with torch.no_grad():
            fallback_output = model_cpu.generate(**inputs_cpu, **fallback_kwargs)
        fallback_caption = processor.decode(fallback_output.detach().cpu()[0],
                                            skip_special_tokens=True)
        fallback_caption = normalize_subword_text(fallback_caption)
        if not looks_broken(fallback_caption):
            caption = fallback_caption

    if device == "mps":
        model.to(device)
        torch.mps.synchronize()

    del output, output_cpu, inputs, inputs_cpu
    return caption.strip()


def generate_with_pipeline(model, processor, image: Image.Image, device: str,
                           gen_kwargs: Dict[str, Any]) -> Tuple[str, str]:
    """Pipeline đầy đủ: BLIP → Accent Restoration"""
    caption_no_accent = generate_blip(model, processor, image, device, gen_kwargs)
    caption_with_accent = restore_accent(caption_no_accent)
    return caption_no_accent, caption_with_accent


# ============================================================
# BLEU SCORING - giống eval_new_model.py
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_bleu4(pred: str, ref: str) -> Tuple[float, float, float, float]:
    """Tính BLEU-1,2,3,4 cho 1 cặp prediction-reference"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        if not pred_tokens or not ref_tokens:
            return (0.0, 0.0, 0.0, 0.0)

        smoothing = SmoothingFunction().method1

        w1 = (1.0, 0.0, 0.0, 0.0)
        w2 = (0.5, 0.5, 0.0, 0.0)
        w3 = (0.33, 0.33, 0.34, 0.0)
        w4 = (0.25, 0.25, 0.25, 0.25)

        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=w1,
                               smoothing_function=smoothing)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=w2,
                               smoothing_function=smoothing)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=w3,
                               smoothing_function=smoothing)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=w4,
                               smoothing_function=smoothing)

        return (float(bleu1), float(bleu2), float(bleu3), float(bleu4))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


# ============================================================
# GRID SEARCH CONFIGS
# ============================================================

# Baseline hiện tại của eval_new_model.py
BASELINE = {
    "num_beams": 3,
    "max_new_tokens": 60,
    "repetition_penalty": 1.05,
    "length_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "min_length": 0,
}

ALL_CONFIGS = [
    # --- Baseline ---
    {"name": "baseline", **BASELINE},

    # --- Beam size ---
    {"name": "beam4",         "num_beams": 4,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "beam5",         "num_beams": 5,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},

    # --- Token length ---
    {"name": "len40",         "num_beams": 3,         "max_new_tokens": 40, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "len50",         "num_beams": 3,         "max_new_tokens": 50, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "len75",         "num_beams": 3,         "max_new_tokens": 75, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},

    # --- Repetition penalty ---
    {"name": "rep1.0",        "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.0,  "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "rep1.2",        "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.2,  "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "rep1.5",        "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.5,  "length_penalty": 1.1, "no_repeat_ngram_size": 3, "early_stopping": True},

    # --- Length penalty ---
    {"name": "lenpen0.8",     "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 0.8, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "lenpen1.0",     "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "lenpen1.5",     "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.5, "no_repeat_ngram_size": 3, "early_stopping": True},

    # --- no_repeat_ngram_size ---
    {"name": "ngram2",        "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 2, "early_stopping": True},
    {"name": "ngram4",        "num_beams": 3,         "max_new_tokens": 60, "repetition_penalty": 1.05, "length_penalty": 1.1, "no_repeat_ngram_size": 4, "early_stopping": True},

    # --- Combined best bets ---
    {"name": "beam5_len50",   "num_beams": 5,         "max_new_tokens": 50, "repetition_penalty": 1.05, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "beam4_len40",   "num_beams": 4,         "max_new_tokens": 40, "repetition_penalty": 1.05, "length_penalty": 0.9, "no_repeat_ngram_size": 3, "early_stopping": True},
    {"name": "beam5_len60_rep1.2", "num_beams": 5,    "max_new_tokens": 60, "repetition_penalty": 1.2,  "length_penalty": 1.0, "no_repeat_ngram_size": 3, "early_stopping": True},
]


# ============================================================
# RUN SINGLE CONFIG
# ============================================================

def run_experiment(
    model, processor, device: str,
    test_csv: Path,
    image_dir: Path,
    cfg: Dict[str, Any],
    max_samples: int,
) -> Dict[str, Any]:
    """Chạy 1 config, đánh giá trên BLEU có dấu"""

    gt_map: Dict[str, str] = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            caption = row.get("caption_vi") or row.get("caption") or ""
            if img and caption:
                gt_map[img] = caption.strip()

    # Build gen_kwargs
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg["max_new_tokens"],
        "num_beams": cfg["num_beams"],
        "early_stopping": cfg["early_stopping"],
        "repetition_penalty": cfg["repetition_penalty"],
        "length_penalty": cfg["length_penalty"],
        "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
    }
    if cfg.get("min_length", 0) > 0:
        gen_kwargs["min_length"] = cfg["min_length"]

    bleu_scores: List[float] = []
    bleu1_scores: List[float] = []
    processed = 0

    for img_name in gt_map:
        if processed >= max_samples:
            break

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path)
            _, caption_with_accent = generate_with_pipeline(
                model, processor, image, device, gen_kwargs
            )
            ref = gt_map[img_name]
            _, _, _, bleu4 = calculate_bleu4(caption_with_accent, ref)
            bleu1, _, _, _ = calculate_bleu4(caption_with_accent, ref)
            bleu_scores.append(bleu4)
            bleu1_scores.append(bleu1)
            processed += 1
        except Exception:
            continue

    return {
        "bleu4": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        "samples": processed,
    }


# ============================================================
# MAIN GRID SEARCH
# ============================================================

def main():
    # Chọn configs
    configs = ALL_CONFIGS[:_args.configs] if _args.configs else ALL_CONFIGS

    print("=" * 70)
    print("🔬 GRID SEARCH GENERATION - KHỚP PIPELINE THẬT")
    print("=" * 70)
    print(f"📦 Model:       {MODEL_PATH}")
    print(f"📊 Test CSV:    {TEST_CSV}")
    print(f"🖼️  Image dir:  {IMAGE_DIR}")
    print(f"📝 Samples:     {_args.max_samples}  |  Configs: {len(configs)}")
    print(f"⏱️  Est. time:  ~{len(configs) * _args.max_samples * 2 // 60} min")
    print()

    model, processor, device = load_model(MODEL_PATH)

    results: List[Dict[str, Any]] = []

    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        gen_info = (
            f"beam={cfg['num_beams']}, "
            f"tokens={cfg['max_new_tokens']}, "
            f"rep={cfg['repetition_penalty']}, "
            f"lp={cfg['length_penalty']}, "
            f"ngram={cfg['no_repeat_ngram_size']}"
        )
        print(f"[{i}/{len(configs)}] {name:<25} | {gen_info}")
        print(f"            ", end="", flush=True)

        start = time.time()
        result = run_experiment(
            model, processor, device,
            TEST_CSV, IMAGE_DIR,
            cfg, _args.max_samples,
        )
        elapsed = time.time() - start

        result["config_name"] = name
        result["params"] = {k: v for k, v in cfg.items() if k != "name"}
        results.append(result)

        print(f"BLEU4={result['bleu4']:.4f}  BLEU1={result['bleu1']:.4f}  "
              f"({result['samples']} samples, {elapsed:.1f}s)")

    # Sort by BLEU4 (có dấu)
    results.sort(key=lambda x: x["bleu4"], reverse=True)

    # Print ranking
    print()
    print("=" * 75)
    print("📊 KẾT QUẢ XẾP HẠNG (BLEU-4 có dấu)")
    print("=" * 75)
    print(f"{'Rank':<5} | {'Config':<28} | {'BLEU4':>7} | {'BLEU1':>7} | "
          f"{'Beams':>5} | {'Tokens':>6} | {'RepPen':>6} | {'LenPen':>6}")
    print("-" * 75)

    baseline_bleu4 = next((r["bleu4"] for r in results if r["config_name"] == "baseline"), 0)
    for i, r in enumerate(results, 1):
        p = r["params"]
        delta = r["bleu4"] - baseline_bleu4
        sign = "+" if delta >= 0 else ""
        marker = " ◀ BEST" if i == 1 else (" ★ baseline" if r["config_name"] == "baseline" else "")
        print(
            f" {i:3}  | {r['config_name']:<28} | {r['bleu4']:>7.4f} | {r['bleu1']:>7.4f} | "
            f" {p['num_beams']:3}  |   {p['max_new_tokens']:3}   |  "
            f" {p['repetition_penalty']:.2f}  |  {p['length_penalty']:.1f}   "
            f"{sign}{delta:.4f}{marker}"
        )

    # Best config
    best = results[0]
    print()
    print("=" * 75)
    print("🏆 CONFIG TỐT NHẤT")
    print("=" * 75)
    print(f"   Name:   {best['config_name']}")
    print(f"   BLEU-4: {best['bleu4']:.4f}  (baseline: {baseline_bleu4:.4f}, "
          f"delta: {best['bleu4']-baseline_bleu4:+.4f})")
    print(f"   BLEU-1: {best['bleu1']:.4f}")
    print(f"   Params:")
    for k, v in best["params"].items():
        print(f"      {k}: {v}")

    # Save results
    output_csv = BASE_DIR / "outputs" / "grid_search_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "config_name", "bleu4", "bleu1", "samples",
            "num_beams", "max_new_tokens", "repetition_penalty",
            "length_penalty", "no_repeat_ngram_size",
        ])
        writer.writeheader()
        for i, r in enumerate(results, 1):
            p = r["params"]
            writer.writerow({
                "rank": i,
                "config_name": r["config_name"],
                "bleu4": round(r["bleu4"], 4),
                "bleu1": round(r["bleu1"], 4),
                "samples": r["samples"],
                "num_beams": p["num_beams"],
                "max_new_tokens": p["max_new_tokens"],
                "repetition_penalty": p["repetition_penalty"],
                "length_penalty": p["length_penalty"],
                "no_repeat_ngram_size": p["no_repeat_ngram_size"],
            })

    print(f"\n💾 Đã lưu: {output_csv}")
    print()
    print("▶ Để chạy full test (1514 samples) với config tốt nhất:")
    print(f"   python tools/eval_new_model.py --model-path {_args.model}")
    print()
    print("▶ Hoặc muốn thử config cụ thể, cập nhật app/core/config.py:")


if __name__ == "__main__":
    main()
