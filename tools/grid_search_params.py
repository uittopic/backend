#!/usr/bin/env python3
"""
Thử nghiệm grid search các generation parameters
để tìm config tốt nhất cho BLEU
"""
import argparse
import csv
import io
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

# Set env trước khi import torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE_DIR = Path(__file__).parent.parent
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "images"
DEFAULT_TEST_CSV = BASE_DIR / "data" / "test_20.csv"
DEFAULT_GROUND_TRUTH_CSV = BASE_DIR / "outputs" / "ground_truth_simple.csv"


@dataclass
class GenParams:
    """Các parameters cho generation"""
    num_beams: int = 3
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    length_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    min_length: int = 5


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_text(text: str) -> str:
    """Chuẩn hóa text"""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(model_path: Path):
    """Load BLIP model"""
    device = get_device()
    print(f"📱 Device: {device}")

    loaded_model = None  # type: BlipForConditionalGeneration | None
    loaded_processor = None  # type: BlipProcessor | None

    if model_path.exists() and any(model_path.iterdir()):
        print(f"📦 Load từ: {model_path}")
        loaded_processor = BlipProcessor.from_pretrained(str(model_path))  # type: ignore
        loaded_model = BlipForConditionalGeneration.from_pretrained(str(model_path))  # type: ignore
    else:
        print("📦 Load pretrained BLIP (Salesforce/blip-image-captioning-base)")
        loaded_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore
        loaded_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore

    loaded_model.to(device)  # type: ignore
    loaded_model.eval()  # type: ignore
    return loaded_model, loaded_processor, device


def generate_caption(model, processor, image: Image.Image, params: GenParams, device: str) -> str:
    """Sinh caption với parameters cho trước"""
    import re

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": params.max_new_tokens,
        "num_beams": params.num_beams,
        "early_stopping": params.early_stopping,
        "repetition_penalty": params.repetition_penalty,
        "length_penalty": params.length_penalty,
        "no_repeat_ngram_size": params.no_repeat_ngram_size,
    }

    if params.min_length > 0:
        gen_kwargs["min_length"] = params.min_length

    with torch.no_grad():
        if device == "mps":
            model_cpu = model.cpu()  # type: ignore
            inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
            output = model_cpu.generate(**inputs_cpu, **gen_kwargs)  # type: ignore
            model.to(device)  # type: ignore
            torch.mps.synchronize()
        else:
            output = model.generate(**inputs, **gen_kwargs)  # type: ignore

    caption = processor.decode(output[0], skip_special_tokens=True)  # type: ignore

    tokens = caption.strip().split()
    merged = []
    for token in tokens:
        if token.startswith("##"):
            piece = token[2:]
            if piece and merged:
                merged[-1] = f"{merged[-1]}{piece}"
        else:
            merged.append(token)

    caption = " ".join(merged)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def calculate_bleu_nltk(pred: str, ref: str) -> float:
    """Tính BLEU-4 (chuan)"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        smoothing = SmoothingFunction().method1  # type: ignore
        bleu_score_raw = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        return float(bleu_score_raw) if bleu_score_raw is not None else 0.0  # type: ignore[reportArgumentType]
    except Exception:
        return 0.0


def run_experiment(
    model, processor, device: str,
    test_csv: Path,
    image_dir: Path,
    params: GenParams,
    max_samples: int = 50
) -> Dict[str, float]:
    """Chạy thí nghiệm với 1 config"""

    # Load ground truth
    gt_map: Dict[str, str] = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            caption = row.get("caption_vi") or row.get("caption") or ""
            if img and caption:
                gt_map[img] = caption.strip()

    bleu_scores: List[float] = []
    processed = 0
    errors = 0

    for img_name in gt_map:
        if processed >= max_samples:
            break

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path)
            pred = generate_caption(model, processor, image, params, device)
            ref = gt_map[img_name]
            bleu = calculate_bleu_nltk(pred, ref)
            bleu_scores.append(bleu)
            processed += 1
        except Exception:
            errors += 1
            continue

    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "samples": processed,
        "errors": errors
    }


def grid_search(
    model, processor, device: str,
    test_csv: Path,
    image_dir: Path,
    max_samples: int = 50
) -> List[Dict[str, Any]]:
    """Grid search để tìm config tốt nhất"""

    # Grid parameters
    configs = [
        {"name": "baseline", "num_beams": 3, "max_new_tokens": 50,
         "repetition_penalty": 1.2, "length_penalty": 1.1, "no_repeat_ngram_size": 3},
        {"name": "beam5", "num_beams": 5, "max_new_tokens": 50,
         "repetition_penalty": 1.2, "length_penalty": 1.1, "no_repeat_ngram_size": 3},
        {"name": "len75", "num_beams": 3, "max_new_tokens": 75,
         "repetition_penalty": 1.2, "length_penalty": 1.1, "no_repeat_ngram_size": 3},
        {"name": "lenpen1.5", "num_beams": 3, "max_new_tokens": 50,
         "repetition_penalty": 1.2, "length_penalty": 1.5, "no_repeat_ngram_size": 3},
        {"name": "reppel2.0", "num_beams": 3, "max_new_tokens": 50,
         "repetition_penalty": 2.0, "length_penalty": 1.1, "no_repeat_ngram_size": 3},
        {"name": "beam5_len75", "num_beams": 5, "max_new_tokens": 75,
         "repetition_penalty": 1.2, "length_penalty": 1.2, "no_repeat_ngram_size": 3},
        {"name": "beam5_len100_rep2.0", "num_beams": 5, "max_new_tokens": 100,
         "repetition_penalty": 2.0, "length_penalty": 1.5, "no_repeat_ngram_size": 3},
        {"name": "sample", "num_beams": 1, "max_new_tokens": 75,
         "repetition_penalty": 1.5, "length_penalty": 1.0, "no_repeat_ngram_size": 0},
        {"name": "minlen20", "num_beams": 3, "max_new_tokens": 75,
         "repetition_penalty": 1.2, "length_penalty": 1.2, "no_repeat_ngram_size": 3, "min_length": 20},
    ]

    results: List[Dict[str, Any]] = []
    total = len(configs)

    print("=" * 70)
    print("🔬 GRID SEARCH: TIM CONFIG TOT NHAT")
    print("=" * 70)
    print(f"📊 Thu {total} configs tren {max_samples} samples")
    print()

    for i, cfg in enumerate(configs, 1):
        params = GenParams(
            num_beams=cfg.get("num_beams", 3),
            max_new_tokens=cfg.get("max_new_tokens", 50),
            repetition_penalty=cfg.get("repetition_penalty", 1.2),
            length_penalty=cfg.get("length_penalty", 1.1),
            no_repeat_ngram_size=cfg.get("no_repeat_ngram_size", 3),
            min_length=cfg.get("min_length", 5),
        )

        print(f"[{i}/{total}] {cfg['name']}: ", end="", flush=True)

        start = time.time()
        result: Dict[str, Any] = dict(run_experiment(model, processor, device, test_csv, image_dir, params, max_samples))
        result["config_name"] = cfg["name"]
        result["params"] = cfg
        results.append(result)

        elapsed = time.time() - start
        print(f"BLEU={result['bleu']:.4f} ({result['samples']} samples, {elapsed:.1f}s)")

    results.sort(key=lambda x: x["bleu"], reverse=True)

    print()
    print("=" * 70)
    print("KET QUA XEP HANG:")
    print("=" * 70)
    print("Rank | Config               | BLEU   | Beams | Len | RepPen | LenPen | Tokens")
    for i, r in enumerate(results, 1):
        p = r["params"]
        print(f" {i:2} | {p['name']:<20} | {r['bleu']:.4f} |   {p['num_beams']:2}  | {p['max_new_tokens']:3} |   {p['repetition_penalty']:.1f}  |   {p['length_penalty']:.1f}   |   {p['num_beams']*p['max_new_tokens']//100:2}   |")

    print()
    best = results[0]
    print("=" * 70)
    print("CONFIG TOT NHAT:")
    print("=" * 70)
    print(f"   Name: {best['config_name']}")
    print(f"   BLEU: {best['bleu']:.4f}")
    print(f"   Params:")
    for k, v in best['params'].items():
        print(f"      - {k}: {v}")

    # Lưu kết quả
    output_csv = BASE_DIR / "outputs" / "grid_search_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "config_name", "bleu", "samples", "params"])
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "rank": i,
                "config_name": r["config_name"],
                "bleu": r["bleu"],
                "samples": r["samples"],
                "params": str(r["params"])
            })
    print(f"\nDa luu: {output_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Grid search generation parameters")
    parser.add_argument("--model", type=Path,
                        default=BASE_DIR / "models" / "blip_vietnamese",
                        help="Đường dẫn model")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV,
                        help="File test CSV")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="Thư mục ảnh")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Số samples để thử nghiệm (mặc định 50)")

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 GRID SEARCH: CAI THIEN BLEU")
    print("=" * 70)

    # Load model
    model, processor, device = load_model(args.model)

    # Run grid search
    results = grid_search(
        model, processor, device,
        args.test_csv,
        args.image_dir,
        args.max_samples
    )


if __name__ == "__main__":
    main()
