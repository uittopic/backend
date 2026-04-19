#!/usr/bin/env python3
"""
Evaluation script - KHỚP PIPELINE THẬT (API production)
Pipeline: BLIP → Accent Restoration → BLEU score

Điểm khác với bản trước:
- Dùng Accent Restoration (không dấu → có dấu)
- Dùng đúng GENERATION_KWARGS từ config.py
- Tách rõ: BLEU không dấu vs BLEU có dấu

Cách dùng:
  python tools/eval_new_model.py
  python tools/eval_new_model.py --model-path models/blip_vietnamese_cleaned_v1
  python tools/eval_new_model.py --model-path models/blip_vietnamese_80_20 --output outputs/metrics_80_20.csv
"""
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path(__file__).parent.parent

# CLI args — cho phép override không cần sửa code
parser = argparse.ArgumentParser(description="Eval script khớp pipeline production")
parser.add_argument(
    "--model-path",
    default="models/blip_vietnamese_cleaned_v1",
    help="Đường dẫn model (relative to BASE_DIR, hoặc absolute). "
         "VD: models/blip_vietnamese_cleaned_v1  hoặc  models/blip_vietnamese_80_20",
)
parser.add_argument(
    "--test-csv",
    default="data/test_20.csv",
    help="Đường dẫn file test CSV (relative to BASE_DIR). Default: data/test_20.csv",
)
parser.add_argument(
    "--images-dir",
    default="data/images",
    help="Thư mục chứa ảnh (relative to BASE_DIR). Default: data/images",
)
parser.add_argument(
    "--output",
    default=None,
    help="File CSV output. Mặc định: outputs/metrics_<tên_model>.csv",
)
parser.add_argument(
    "--summary",
    default=None,
    help="File JSON summary. Mặc định: outputs/full_eval/evaluation_summary.json",
)
_args = parser.parse_args()

MODEL_PATH = BASE_DIR / _args.model_path
TEST_CSV = BASE_DIR / _args.test_csv
IMAGE_DIR = BASE_DIR / _args.images_dir

# Auto-generate output filenames nếu không truyền
model_slug = _args.model_path.rstrip("/").replace("/", "_")
_default_csv = BASE_DIR / "outputs" / f"metrics_{model_slug}.csv"
_default_summary = BASE_DIR / "outputs" / "full_eval" / f"evaluation_{model_slug}.json"

OUTPUT_CSV = (BASE_DIR / _args.output) if _args.output else _default_csv
SUMMARY_JSON = (BASE_DIR / _args.summary) if _args.summary else _default_summary

# Import config (để lấy đúng generation params)
import sys
sys.path.insert(0, str(BASE_DIR))
from app.core.config import (
    MAX_NEW_TOKENS, NUM_BEAMS, EARLY_STOPPING,
    NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY, LENGTH_PENALTY,
    get_device, synchronize_device, clear_device_cache,
)
from app.core.accent_restoration_loader import restore_accent


# ============================================================
# MODEL LOADING - giống API production
# ============================================================

def load_blip_model(model_path: Path):
    """Load BLIP model giống caption_service.py"""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = get_device()
    print(f"📱 Device: {device}")
    print(f"📦 Load model từ: {model_path}")

    processor = BlipProcessor.from_pretrained(str(model_path))  # type: ignore
    model = BlipForConditionalGeneration.from_pretrained(str(model_path))  # type: ignore

    model.to(device)  # type: ignore
    model.eval()  # type: ignore
    return model, processor, device


# ============================================================
# GENERATION - giống API production
# ============================================================

def get_generation_kwargs() -> Dict[str, Any]:
    """Lấy đúng generation config từ config.py"""
    kwargs: Dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS,
        "early_stopping": EARLY_STOPPING,
        "repetition_penalty": REPETITION_PENALTY,
        "length_penalty": LENGTH_PENALTY,
    }
    if NO_REPEAT_NGRAM_SIZE > 0:
        kwargs["no_repeat_ngram_size"] = NO_REPEAT_NGRAM_SIZE
    return kwargs


def normalize_subword_text(text: str) -> str:
    """Làm sạch token subword ##xxx - giống caption_service.py"""
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


def looks_broken(text: str) -> bool:
    """Kiểm tra caption có bị lỗi không - giống caption_service.py"""
    cleaned = text.strip()
    if not cleaned or len(cleaned) <= 2:
        return True
    if "##" in cleaned:
        return True
    if not re.search(r"[A-Za-z0-9À-ỹ]", cleaned):
        return True
    return False


def run_blip(model, processor, image: Image.Image, device: str) -> str:
    """Chạy BLIP - giống _run_blip trong caption_service.py"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore

    gen_kwargs = get_generation_kwargs()

    # MPS: chuyển model về CPU để generate
    if device == "mps":
        model_cpu = model.cpu()  # type: ignore
        inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
    else:
        model_cpu = model
        inputs_cpu = inputs

    with torch.no_grad():
        output = model_cpu.generate(**inputs_cpu, **gen_kwargs)  # type: ignore

    output_cpu = output.detach().cpu()
    caption = processor.decode(output_cpu[0], skip_special_tokens=True)  # type: ignore
    caption = normalize_subword_text(caption)

    # Fallback nếu caption bị lỗi
    if looks_broken(caption):
        fallback_kwargs = dict(gen_kwargs)
        fallback_kwargs["num_beams"] = 1
        fallback_kwargs["repetition_penalty"] = 1.0
        fallback_kwargs["do_sample"] = False
        fallback_kwargs.pop("no_repeat_ngram_size", None)

        with torch.no_grad():
            fallback_output = model_cpu.generate(**inputs_cpu, **fallback_kwargs)  # type: ignore
        fallback_output_cpu = fallback_output.detach().cpu()
        fallback_caption = processor.decode(fallback_output_cpu[0], skip_special_tokens=True)  # type: ignore
        fallback_caption = normalize_subword_text(fallback_caption)

        if not looks_broken(fallback_caption):
            caption = fallback_caption
        del fallback_output, fallback_output_cpu, fallback_caption

    # Clean up
    if device == "mps":
        model.to(device)  # type: ignore
        synchronize_device()
    else:
        synchronize_device()

    del output, output_cpu, inputs, inputs_cpu

    return caption.strip()


def generate_with_pipeline(model, processor, image: Image.Image, device: str) -> Tuple[str, str]:
    """
    Pipeline đầy đủ: BLIP → Accent Restoration
    Returns: (caption_khong_dau, caption_co_dau)
    """
    # Bước 1: BLIP sinh caption không dấu
    caption_no_accent = run_blip(model, processor, image, device)

    # Bước 2: Accent Restoration
    caption_with_accent = restore_accent(caption_no_accent)

    return caption_no_accent, caption_with_accent


# ============================================================
# BLEU SCORING
# ============================================================

def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi tính BLEU"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_bleu(pred: str, ref: str) -> Tuple[float, float, float, float, float]:
    """Calculate BLEU-1,2,3,4 scores"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        if not pred_tokens or not ref_tokens:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        smoothing = SmoothingFunction().method1  # type: ignore

        w1 = (1.0, 0.0, 0.0, 0.0)
        w2 = (0.5, 0.5, 0.0, 0.0)
        w3 = (0.33, 0.33, 0.34, 0.0)
        w4 = (0.25, 0.25, 0.25, 0.25)

        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=w1, smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=w2, smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=w3, smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=w4, smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)  # type: ignore[reportArgumentType]

        return (float(bleu1), float(bleu2), float(bleu3), float(bleu4), float(bleu))  # type: ignore[arg-type]
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, 0.0)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 ĐÁNH GIÁ MODEL - KHỚP PIPELINE THẬT")
    print("=" * 70)
    print()
    print(f"📦 Model:     {MODEL_PATH}")
    print(f"📊 Test CSV:  {TEST_CSV}")
    print(f"🖼️  Images:   {IMAGE_DIR}")
    print(f"💾 Output:    {OUTPUT_CSV}")
    print()
    print("Pipeline: BLIP → Accent Restoration → BLEU")
    print(f"Generation config: beams={NUM_BEAMS}, max_tokens={MAX_NEW_TOKENS}, "
          f"rep_penalty={REPETITION_PENALTY}")
    print()

    # Load model
    model, processor, device = load_blip_model(MODEL_PATH)

    # Load ground truth
    gt_map: Dict[str, str] = {}
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            caption = row.get("caption_vi") or row.get("caption") or ""
            if img and caption:
                gt_map[img] = caption.strip()

    print(f"📊 Ground truth: {len(gt_map)} samples")

    # Evaluate
    results: list = []
    bleu_no_accent: Dict[str, list] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}
    bleu_with_accent: Dict[str, list] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}
    all_words_no_accent: list = []
    all_words_with_accent: list = []

    start_time = time.time()
    total_samples = len(gt_map)

    for i, (img_name, gt) in enumerate(gt_map.items(), 1):
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start_time
            avg_bleu = sum(bleu_no_accent["bleu"]) / len(bleu_no_accent["bleu"]) if bleu_no_accent["bleu"] else 0
            eta = (elapsed / i) * (total_samples - i) if i > 0 else 0
            print(f"   [{i}/{total_samples}] BLEU: {avg_bleu:.4f} | ETA: {eta/60:.1f}min")

        img_path = IMAGE_DIR / img_name
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path)
            caption_no_accent, caption_with_accent = generate_with_pipeline(
                model, processor, image, device
            )

            # Tính BLEU cho cả 2 version
            scores_no_accent = calculate_bleu(caption_no_accent, gt)
            scores_with_accent = calculate_bleu(caption_with_accent, gt)

            for j, key in enumerate(["bleu1", "bleu2", "bleu3", "bleu4", "bleu"]):
                bleu_no_accent[key].append(scores_no_accent[j])
                bleu_with_accent[key].append(scores_with_accent[j])

            all_words_no_accent.extend(caption_no_accent.split())
            all_words_with_accent.extend(caption_with_accent.split())

            results.append({
                "image": img_name,
                "ground_truth": gt,
                "caption_no_accent": caption_no_accent,
                "caption_with_accent": caption_with_accent,
                "bleu1_no_accent": scores_no_accent[0],
                "bleu1_with_accent": scores_with_accent[0],
                "bleu2_no_accent": scores_no_accent[1],
                "bleu2_with_accent": scores_with_accent[1],
                "bleu3_no_accent": scores_no_accent[2],
                "bleu3_with_accent": scores_with_accent[2],
                "bleu4_no_accent": scores_no_accent[3],
                "bleu4_with_accent": scores_with_accent[3],
                "bleu_no_accent": scores_no_accent[4],
                "bleu_with_accent": scores_with_accent[4],
            })

            clear_device_cache()

        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")
            continue

    # Tính trung bình
    n = len(results)
    avg: Dict[str, Dict[str, float]] = {
        "no_accent": {k: sum(v) / n for k, v in bleu_no_accent.items()},
        "with_accent": {k: sum(v) / n for k, v in bleu_with_accent.items()},
    }

    avg_len_no_accent = len(all_words_no_accent) / n if n > 0 else 0
    avg_len_with_accent = len(all_words_with_accent) / n if n > 0 else 0

    # Lưu kết quả
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "ground_truth",
            "caption_no_accent", "caption_with_accent",
            "bleu1_no_accent", "bleu1_with_accent",
            "bleu2_no_accent", "bleu2_with_accent",
            "bleu3_no_accent", "bleu3_with_accent",
            "bleu4_no_accent", "bleu4_with_accent",
            "bleu_no_accent", "bleu_with_accent",
        ])
        writer.writeheader()
        writer.writerows(results)

    # Lưu evaluation_summary.json (dùng cho báo cáo)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(MODEL_PATH),
        "test_samples": n,
        "pipeline": "BLIP → Accent Restoration",
        "bleu_scores": {
            "bleu1_no_accent": round(avg["no_accent"]["bleu1"], 4),
            "bleu4_no_accent": round(avg["no_accent"]["bleu4"], 4),
            "bleu1_with_accent": round(avg["with_accent"]["bleu1"], 4),
            "bleu4_with_accent": round(avg["with_accent"]["bleu4"], 4),
        },
        "caption_length": {
            "no_accent": round(avg_len_no_accent, 1),
            "with_accent": round(avg_len_with_accent, 1),
        },
    }
    import json
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    # In kết quả
    print()
    print("=" * 70)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (KHỚP PIPELINE THẬT)")
    print("=" * 70)
    print(f"Total samples: {n}")
    print(f"Time: {elapsed/60:.1f}min ({elapsed/n:.1f}s/image)")
    print()
    print("BLEU SCORES (không dấu):")
    print(f"{'Metric':<10} | {'Value':<12}")
    print("-" * 25)
    for key in ["bleu1", "bleu2", "bleu3", "bleu4"]:
        print(f"{key.upper():<10} | {avg['no_accent'][key]:>10.4f}")
    print()
    print("BLEU SCORES (có dấu - Accent Restoration):")
    print(f"{'Metric':<10} | {'Value':<12}")
    print("-" * 25)
    for key in ["bleu1", "bleu2", "bleu3", "bleu4"]:
        print(f"{key.upper():<10} | {avg['with_accent'][key]:>10.4f}")
    print()
    print("CAPTION LENGTH:")
    print(f"  Không dấu: {avg_len_no_accent:.1f} words avg")
    print(f"  Có dấu:    {avg_len_with_accent:.1f} words avg")
    print()
    print(f"💾 Kết quả chi tiết: {OUTPUT_CSV}")
    print(f"📋 Summary JSON:       {SUMMARY_JSON}")
    print()
    print("=" * 70)
    print("SO SÁNH 2 MODEL:")
    print("=" * 70)
    print("Lần 1 (model A):")
    print(f"  python tools/eval_new_model.py --model-path models/blip_vietnamese_cleaned_v1")
    print()
    print("Lần 2 (model B):")
    print(f"  python tools/eval_new_model.py --model-path models/blip_vietnamese_80_20 --output outputs/metrics_80_20.csv")
    print()
    print("Sau đó so sánh 2 file summary JSON.")


if __name__ == "__main__":
    main()