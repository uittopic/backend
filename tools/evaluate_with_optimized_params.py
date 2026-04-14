#!/usr/bin/env python3
"""
Đánh giá lại BLEU với config mới (num_beams=5)
và xuất báo cáo đầy đủ BLEU-1→4
"""
import argparse
import csv
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# === CONFIG ===
BASE_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "blip_vietnamese_80_20"
DEFAULT_TEST_CSV = BASE_DIR / "data" / "test_20.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "images"
DEFAULT_OUTPUT_CSV = BASE_DIR / "outputs" / "comparison_test_20_optimized.csv"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi đo BLEU"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class GenParams:
    num_beams: int = 5
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    length_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    min_length: int = 5


def load_model(model_path: Path):
    """Load BLIP model - ưu tiên model chính (root), fallback sang checkpoint lớn nhất"""
    device = get_device()
    print(f"📱 Device: {device}")

    # Kiểm tra model chính ở root trước
    model_safetensors = model_path / "model.safetensors"
    model_bin = model_path / "pytorch_model.bin"

    loaded_model = None  # type: BlipForConditionalGeneration | None
    loaded_processor = None  # type: BlipProcessor | None

    if model_safetensors.exists() or model_bin.exists():
        # Có model chính → dùng luôn
        print(f"📦 Load model chính từ: {model_path}")
        loaded_processor = BlipProcessor.from_pretrained(str(model_path))  # type: ignore
        loaded_model = BlipForConditionalGeneration.from_pretrained(str(model_path))  # type: ignore
    else:
        # Không có model chính → tìm checkpoint với step lớn nhất
        checkpoints: List[tuple] = []
        for d in model_path.iterdir():
            if d.name.startswith("checkpoint-"):
                try:
                    step = int(d.name.split("-")[1])
                    checkpoints.append((step, d))
                except Exception:
                    pass
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            step, checkpoint_path = checkpoints[-1]
            print(f"📦 Sử dụng checkpoint: {checkpoint_path.name} (step {step})")
            loaded_processor = BlipProcessor.from_pretrained(str(model_path))  # type: ignore
            loaded_model = BlipForConditionalGeneration.from_pretrained(str(checkpoint_path))  # type: ignore
        else:
            print("⚠️  Model không tìm thấy, load pretrained BLIP")
            loaded_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore
            loaded_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # type: ignore

    loaded_model.to(device)  # type: ignore
    loaded_model.eval()  # type: ignore
    return loaded_model, loaded_processor, device


def generate_caption(model, processor, image: Image.Image, params: GenParams, device: str) -> str:
    """Sinh caption với parameters đã tối ưu"""
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
            model.to(torch.device(device))  # type: ignore
            torch.mps.synchronize()
        else:
            output = model.generate(**inputs, **gen_kwargs)  # type: ignore

    caption = processor.decode(output[0], skip_special_tokens=True)  # type: ignore

    # Normalize subword
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


def calculate_all_bleu(pred: str, ref: str) -> Dict[str, float]:
    """Tính BLEU-1, 2, 3, 4"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        if not pred_tokens or not ref_tokens:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}

        smoothing = SmoothingFunction().method1  # type: ignore

        bleu1_raw = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu2_raw = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu3_raw = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu4_raw = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)  # type: ignore[reportArgumentType]
        bleu_raw = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)  # type: ignore[reportArgumentType]

        return {
            "bleu1": float(bleu1_raw),  # type: ignore[reportArgumentType]
            "bleu2": float(bleu2_raw),  # type: ignore[reportArgumentType]
            "bleu3": float(bleu3_raw),  # type: ignore[reportArgumentType]
            "bleu4": float(bleu4_raw),  # type: ignore[reportArgumentType]
            "bleu": float(bleu_raw)  # type: ignore[reportArgumentType]
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}


def run_evaluation(
    model, processor, device: str,
    test_csv: Path,
    image_dir: Path,
    params: GenParams,
    output_csv: Path,
    max_samples: int = 0
) -> Dict[str, float]:
    """Đánh giá trên toàn bộ test set"""

    # Load ground truth
    gt_map: Dict[str, str] = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            caption = row.get("caption_vi") or row.get("caption") or ""
            if img and caption:
                gt_map[img] = caption.strip()

    print(f"📊 Ground truth: {len(gt_map)} samples")

    # Evaluate
    results: List[Dict[str, Any]] = []
    all_bleu: Dict[str, List[float]] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}
    processed = 0
    errors = 0
    total = len(gt_map)

    if max_samples > 0:
        total = min(max_samples, total)

    print(f"🔄 Bắt đầu đánh giá {total} samples...")
    start_time = time.time()

    for i, (img_name, gt) in enumerate(gt_map.items(), 1):
        if max_samples > 0 and i > max_samples:
            break

        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i) if i > 0 else 0
            current_bleu = sum(all_bleu['bleu'])/len(all_bleu['bleu']) if all_bleu['bleu'] else 0
            print(f"   [{i}/{total}] BLEU: {current_bleu:.4f} | ETA: {eta/60:.1f}min")

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path)
            pred = generate_caption(model, processor, image, params, device)
            bleu_scores = calculate_all_bleu(pred, gt)

            for k, v in bleu_scores.items():
                all_bleu[k].append(v)

            results.append({
                "image": img_name,
                "ground_truth": gt,
                "prediction": pred,
                **bleu_scores
            })
            processed += 1
        except Exception:
            errors += 1
            continue

    # Tính trung bình
    avg: Dict[str, float] = {}
    for k, v in all_bleu.items():
        avg[k] = sum(v) / len(v) if v else 0.0

    # Lưu kết quả
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction",
                                                "bleu1", "bleu2", "bleu3", "bleu4", "bleu"])
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"✅ Hoàn thành: {processed} samples, {errors} lỗi, {elapsed/60:.1f}min")

    return avg


def main():
    parser = argparse.ArgumentParser(description="Đánh giá với config tối ưu (beam=5)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Đường dẫn model")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV,
                        help="File test CSV")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="Thư mục ảnh")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Số samples tối đa (0 = all)")

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 ĐÁNH GIÁ VỚI CONFIG TỐI ƯU (NUM_BEAMS=5)")
    print("=" * 70)

    # Load model
    model, processor, device = load_model(args.model)

    params = GenParams()

    # Run evaluation
    avg = run_evaluation(
        model, processor, device,
        args.test_csv,
        args.image_dir,
        params,
        args.output,
        args.max_samples
    )

    # In kết quả
    print()
    print("=" * 70)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (SAU TỐI ƯU)")
    print("=" * 70)
    print()
    print("BLEU METRICS (Normalized - lowercase, clean)")
    print(f"BLEU-1: {avg['bleu1']:.4f}")
    print(f"BLEU-2: {avg['bleu2']:.4f}")
    print(f"BLEU-3: {avg['bleu3']:.4f}")
    print(f"BLEU-4: {avg['bleu4']:.4f}")
    print(f"BLEU:   {avg['bleu']:.4f}")
    print()

    # So sánh với baseline
    baseline = {
        "bleu1": 0.0896,
        "bleu2": 0.0581,
        "bleu3": 0.0403,
        "bleu4": 0.0315,
        "bleu": 0.0315
    }

    print("SO SÁNH VỚI BASELINE (num_beams=3):")
    print()
    print("Metric | Baseline | Optimized | Improve")
    for k in ["bleu1", "bleu2", "bleu3", "bleu4", "bleu"]:
        diff = avg[k] - baseline[k]
        pct = (diff / baseline[k] * 100) if baseline[k] > 0 else 0
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{k.upper():<8} | {baseline[k]:.4f} | {avg[k]:.4f} | {arrow}{abs(pct):.1f}%")
    print()
    print(f"Ket qua chi tiet: {args.output}")


if __name__ == "__main__":
    main()
