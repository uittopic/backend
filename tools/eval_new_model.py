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
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import unicodedata

# Download NLTK data nếu chưa có
for resource in ["wordnet", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# === CONFIG ===
BASE_DIR = Path(__file__).parent.parent

# CLI args — cho phép override không cần sửa code
parser = argparse.ArgumentParser(description="Eval script khớp pipeline production")
parser.add_argument(
    "--model-path",
    default="models/blip_shopee_vicaps_v1_0_1_mps",
    help="Đường dẫn model (relative to BASE_DIR, hoặc absolute). "
         "VD: models/blip_shopee_vicaps_v1_0_1_mps  hoặc  models/blip_vietnamese_80_20",
)
parser.add_argument(
    "--test-csv",
    default="data/test_clean_v3_final_noleak_filtered.csv",
    help="Đường dẫn file test CSV (relative to BASE_DIR). Default: data/test_clean_v3_final_noleak_filtered.csv",
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
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Giới hạn số mẫu test (mặc định: full test set)",
)
parser.add_argument(
    "--folder-mode",
    action="store_true",
    help="Chạy trên tất cả ảnh trong thư mục, không cần ground truth. "
         "Chỉ sinh caption, không tính BLEU/ROUGE/METEOR.",
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


# ROUGE scorer — word-level regex \w+
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def calculate_rouge_l(pred: str, ref: str) -> float:
    """Calculate ROUGE-L (LCS-based longest common subsequence overlap)"""
    try:
        score = _rouge_scorer.score(ref, pred)
        return float(score["rougeL"].fmeasure)
    except Exception:
        return 0.0


def calculate_meteor(pred: str, ref: str) -> float:
    """Calculate METEOR score using NLTK meteor_metric"""
    try:
        # Tokenize word-level giống BLEU: \w+
        pred_tokens = re.findall(r'\w+', pred.lower())
        ref_tokens = re.findall(r'\w+', ref.lower())
        if not pred_tokens or not ref_tokens:
            return 0.0
        # meteor_score expects list of reference token lists
        score = meteor_score([ref_tokens], pred_tokens)
        return float(score)
    except Exception:
        return 0.0


def calculate_bleu(pred: str, ref: str) -> Tuple[float, float, float, float, float]:
    """Calculate BLEU-1,2,3,4 scores — word-level tokenization giống full_batch_test.py"""
    try:
        import math
        from collections import Counter

        # Word-level tokenization giống full_batch_test.py
        def tokenize(text):
            import re
            return re.findall(r'\w+', text.lower())

        def get_ngrams(tokens, n):
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        if not pred_tokens or not ref_tokens:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

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

        bleu1 = bleu_n(ref_tokens, pred_tokens, 1)
        bleu2 = bleu_n(ref_tokens, pred_tokens, 2)
        bleu3 = bleu_n(ref_tokens, pred_tokens, 3)
        bleu4 = bleu_n(ref_tokens, pred_tokens, 4)

        # BLEU tổng (n=4)
        log_precisions = []
        for n in range(1, 5):
            if n > len(ref_tokens) or n > len(pred_tokens):
                log_precisions.append(float('-inf'))
            else:
                hyp_ng = get_ngrams(pred_tokens, n)
                ref_ng = get_ngrams(ref_tokens, n)
                matches = sum(min(hyp_ng[ng], max(ref_ng.get(ng, 0), 0)) for ng in hyp_ng)
                total = sum(hyp_ng.values())
                precision = matches / total if total > 0 else 0.0
                log_precisions.append(math.log(precision) if precision > 0 else float('-inf'))
        avg_log_p = sum(log_precisions) / 4
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens)) if len(pred_tokens) > 0 else 0.0
        bleu = max(0.0, min(1.0, bp * math.exp(avg_log_p)))

        return (float(bleu1), float(bleu2), float(bleu3), float(bleu4), float(bleu))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, 0.0)


# ============================================================
# FOLDER MODE — run on all images without ground truth
# ============================================================

def _run_folder_mode(model, processor, device):
    """Run model inference on all images in IMAGE_DIR, save captions only."""
    print("📁 FOLDER MODE — không cần ground truth, chỉ sinh caption")
    print()

    import os
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    total = len(image_files)
    print(f"🖼️  Tìm thấy {total} ảnh trong {IMAGE_DIR}")
    print()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "caption_no_accent", "caption_with_accent"]
    results = []

    start_time = time.time()
    for i, img_name in enumerate(image_files, 1):
        if i % 100 == 0 or i == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i) if i > 0 else 0
            print(f"   [{i}/{total}] | ETA: {eta/60:.1f}min")

        img_path = IMAGE_DIR / img_name
        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path)
            caption_no_accent, caption_with_accent = generate_with_pipeline(
                model, processor, image, device
            )
            results.append({
                "image": img_name,
                "caption_no_accent": caption_no_accent,
                "caption_with_accent": caption_with_accent,
            })
            clear_device_cache()
        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")
            continue

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print()
    print(f"✅ Hoàn thành! {len(results)}/{total} ảnh")
    print(f"⏱️  Thời gian: {elapsed/60:.1f} phút")
    print(f"💾 CSV: {OUTPUT_CSV}")
    print(f"📋 Note: Không có BLEU/ROUGE/METEOR vì không có ground truth")


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

    if _args.folder_mode:
        _run_folder_mode(model, processor, device)
        return

    # Load ground truth — auto-detect delimiter + caption column
    gt_map: Dict[str, str] = {}
    with open(TEST_CSV, "r", encoding="utf-8-sig") as f:
        header = f.readline()
        delimiter = ";" if ";" in header else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            img = row.get("image", "").strip()
            caption = (
                row.get("caption_vi_clean")
                or row.get("caption_vi")
                or row.get("caption")
                or ""
            ).strip()
            if img and caption:
                gt_map[img] = caption

    print(f"📊 Ground truth: {len(gt_map)} samples (delimiter='{delimiter}')")

    # Evaluate
    results: list = []
    bleu_semantic: Dict[str, list] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}
    bleu_accented: Dict[str, list] = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "bleu": []}
    rouge_l_semantic_list: List[float] = []
    rouge_l_accented_list: List[float] = []
    meteor_semantic_list: List[float] = []
    meteor_accented_list: List[float] = []
    all_words_no_accent: list = []
    all_words_with_accent: list = []

    # Normalize ground truth sang no-accent để so sánh công bằng
    # GT dùng Shopee-style accents; prediction dùng VNCOMMON style
    # → so với no-accent GT để đo semantic chứ không đo accent style
    def strip_accents(s):
        s = unicodedata.normalize("NFD", s)
        return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    gt_no_accent_map = {img: strip_accents(cap) for img, cap in gt_map.items()}

    start_time = time.time()
    total_samples = len(gt_map)

    images_to_process = list(gt_map.items())
    if _args.limit:
        images_to_process = images_to_process[:_args.limit]
        total_samples = len(images_to_process)

    for i, (img_name, gt) in enumerate(images_to_process, 1):
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start_time
            avg_bleu = sum(bleu_semantic["bleu"]) / len(bleu_semantic["bleu"]) if bleu_semantic["bleu"] else 0
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

            # Semantic no-accent: strip accent from pred + strip accent from ref
            # → đo semantic understanding của model, không bị ảnh hưởng bởi dấu tiếng Việt
            gt_na = gt_no_accent_map[img_name]
            scores_semantic = calculate_bleu(caption_no_accent, gt_na)

            # Accented exact: restored pred (có dấu) vs ground truth gốc (có dấu)
            # → đo chất lượng accent restoration khi cùng style với GT
            gt_raw = gt_map[img_name]
            scores_accented = calculate_bleu(caption_with_accent, gt_raw)

            # Semantic: so caption_no_accent vs gt_na (stripped)
            # Accented: so caption_with_accent vs gt_raw (raw)
            rouge_l_semantic = calculate_rouge_l(caption_no_accent, gt_na)
            rouge_l_accented = calculate_rouge_l(caption_with_accent, gt_raw)
            meteor_semantic = calculate_meteor(caption_no_accent, gt_na)
            meteor_accented = calculate_meteor(caption_with_accent, gt_raw)

            for j, key in enumerate(["bleu1", "bleu2", "bleu3", "bleu4", "bleu"]):
                bleu_semantic[key].append(scores_semantic[j])
                bleu_accented[key].append(scores_accented[j])

            rouge_l_semantic_list.append(rouge_l_semantic)
            rouge_l_accented_list.append(rouge_l_accented)
            meteor_semantic_list.append(meteor_semantic)
            meteor_accented_list.append(meteor_accented)

            all_words_no_accent.extend(caption_no_accent.split())
            all_words_with_accent.extend(caption_with_accent.split())

            results.append({
                "image": img_name,
                "ground_truth": gt_raw,
                "caption_no_accent": caption_no_accent,
                "caption_with_accent": caption_with_accent,
                "bleu1_semantic_no_accent": scores_semantic[0],
                "bleu1_accented_exact": scores_accented[0],
                "bleu2_semantic_no_accent": scores_semantic[1],
                "bleu2_accented_exact": scores_accented[1],
                "bleu3_semantic_no_accent": scores_semantic[2],
                "bleu3_accented_exact": scores_accented[2],
                "bleu4_semantic_no_accent": scores_semantic[3],
                "bleu4_accented_exact": scores_accented[3],
                "bleu_semantic_no_accent": scores_semantic[4],
                "bleu_accented_exact": scores_accented[4],
                "rouge_l_semantic_no_accent": rouge_l_semantic,
                "rouge_l_accented_exact": rouge_l_accented,
                "meteor_semantic_no_accent": meteor_semantic,
                "meteor_accented_exact": meteor_accented,
            })

            clear_device_cache()

        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")
            continue

    # ============================================================
    # SBERT SEMANTIC SIMILARITY — chạy sau khi inference xong
    # ============================================================
    n = len(results)
    if n == 0:
        print("\n❌ Không có sample hợp lệ để tính metric. Dừng đánh giá.")
        return

    print(f"\n🔄 Computing SBERT semantic similarity ({n} samples)...")
    sbert_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    sbert_device = "mps" if torch.backends.mps.is_available() else "cpu"
    sbert = SentenceTransformer(sbert_name, device=sbert_device)

    # refs = ground truth RAW (dùng so sánh accented_exact)
    # refs_semantic = ground truth STRIPPED (dùng so sánh semantic_no_accent)
    # preds_na = prediction KHÔNG DẤU → strip → so với refs_semantic
    # preds_a = prediction CÓ DẤU → so với refs RAW
    refs = [r["ground_truth"] for r in results]
    preds_na = [r["caption_no_accent"] for r in results]
    preds_a = [r["caption_with_accent"] for r in results]

    def rm_accent(s):
        s = str(s).replace("đ", "d").replace("Đ", "D")
        return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

    # Semantic no-accent: strip accent từ cả pred + ref
    refs_semantic = [rm_accent(r) for r in refs]
    preds_semantic = [rm_accent(p) for p in preds_na]

    e_refs_sem = sbert.encode(refs_semantic, batch_size=64, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)
    e_preds_sem = sbert.encode(preds_semantic, batch_size=64, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)
    sbert_semantic = (e_refs_sem * e_preds_sem).sum(dim=1).cpu().numpy()

    # Accented exact: pred có dấu vs ref raw có dấu
    e_refs_raw = sbert.encode(refs, batch_size=64, convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False)
    e_preds_a = sbert.encode(preds_a, batch_size=64, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
    sbert_accented = (e_refs_raw * e_preds_a).sum(dim=1).cpu().numpy()

    for i, r in enumerate(results):
        r["sbert_semantic_no_accent"] = round(float(sbert_semantic[i]), 4)
        r["sbert_accented_exact"] = round(float(sbert_accented[i]), 4)

    avg_sbert_semantic = round(float(sbert_semantic.mean()), 4)
    avg_sbert_accented = round(float(sbert_accented.mean()), 4)
    print(f"  SBERT (semantic no-accent): {avg_sbert_semantic:.4f}")
    print(f"  SBERT (accented exact):     {avg_sbert_accented:.4f}")

    # Tính trung bình
    avg_semantic: Dict[str, float] = {k: sum(v) / n for k, v in bleu_semantic.items()}
    avg_accented: Dict[str, float] = {k: sum(v) / n for k, v in bleu_accented.items()}

    avg_rouge_l_semantic = sum(rouge_l_semantic_list) / n if n > 0 else 0.0
    avg_rouge_l_accented = sum(rouge_l_accented_list) / n if n > 0 else 0.0
    avg_meteor_semantic = sum(meteor_semantic_list) / n if n > 0 else 0.0
    avg_meteor_accented = sum(meteor_accented_list) / n if n > 0 else 0.0

    avg_len_no_accent = len(all_words_no_accent) / n if n > 0 else 0
    avg_len_with_accent = len(all_words_with_accent) / n if n > 0 else 0

    # Lưu kết quả
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "ground_truth",
        "caption_no_accent", "caption_with_accent",
        "bleu1_semantic_no_accent", "bleu1_accented_exact",
        "bleu2_semantic_no_accent", "bleu2_accented_exact",
        "bleu3_semantic_no_accent", "bleu3_accented_exact",
        "bleu4_semantic_no_accent", "bleu4_accented_exact",
        "bleu_semantic_no_accent", "bleu_accented_exact",
        "rouge_l_semantic_no_accent", "rouge_l_accented_exact",
        "meteor_semantic_no_accent", "meteor_accented_exact",
        "sbert_semantic_no_accent", "sbert_accented_exact",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Lưu evaluation_summary.json (dùng cho báo cáo)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(MODEL_PATH),
        "test_samples": n,
        "pipeline": "BLIP → Accent Restoration",
        "tokenization": "word-level regex \\w+",
        "metrics_reported": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "SBERT"],
        "bleu_scores": {
            "bleu1_semantic_no_accent": round(avg_semantic["bleu1"], 4),
            "bleu2_semantic_no_accent": round(avg_semantic["bleu2"], 4),
            "bleu3_semantic_no_accent": round(avg_semantic["bleu3"], 4),
            "bleu4_semantic_no_accent": round(avg_semantic["bleu4"], 4),
            "bleu1_accented_exact": round(avg_accented["bleu1"], 4),
            "bleu2_accented_exact": round(avg_accented["bleu2"], 4),
            "bleu3_accented_exact": round(avg_accented["bleu3"], 4),
            "bleu4_accented_exact": round(avg_accented["bleu4"], 4),
        },
        "rouge_l": {
            "semantic_no_accent": round(avg_rouge_l_semantic, 4),
            "accented_exact": round(avg_rouge_l_accented, 4),
        },
        "meteor": {
            "semantic_no_accent": round(avg_meteor_semantic, 4),
            "accented_exact": round(avg_meteor_accented, 4),
        },
        "sbert": {
            "semantic_no_accent": avg_sbert_semantic,
            "accented_exact": avg_sbert_accented,
        },
        "caption_length": {
            "no_accent": round(avg_len_no_accent, 1),
            "with_accent": round(avg_len_with_accent, 1),
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    # In kết quả
    print()
    print("=" * 70)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (KHỚP PIPELINE THẬT)")
    print("=" * 70)
    print(f"Total samples: {n}  |  Time: {elapsed/60:.1f}min ({elapsed/n:.1f}s/img)")
    print(f"Tokenization: word-level regex \\w+")
    print()
    print(f"{'':25} {'Semantic (no-acc)':>20} {'Accented (exact)':>20}")
    print("-" * 68)
    print(f"{'BLEU-1':<25} {avg_semantic['bleu1']:>20.4f} {avg_accented['bleu1']:>20.4f}")
    print(f"{'BLEU-2':<25} {avg_semantic['bleu2']:>20.4f} {avg_accented['bleu2']:>20.4f}")
    print(f"{'BLEU-3':<25} {avg_semantic['bleu3']:>20.4f} {avg_accented['bleu3']:>20.4f}")
    print(f"{'BLEU-4':<25} {avg_semantic['bleu4']:>20.4f} {avg_accented['bleu4']:>20.4f}")
    print(f"{'ROUGE-L':<25} {avg_rouge_l_semantic:>20.4f} {avg_rouge_l_accented:>20.4f}")
    print(f"{'METEOR':<25} {avg_meteor_semantic:>20.4f} {avg_meteor_accented:>20.4f}")
    print(f"{'SBERT':<25} {avg_sbert_semantic:>20.4f} {avg_sbert_accented:>20.4f}")
    print()
    print("CAPTION LENGTH:")
    print(f"  Không dấu: {avg_len_no_accent:.1f} words avg")
    print(f"  Có dấu:    {avg_len_with_accent:.1f} words avg")
    print()
    print(f"💾 CSV: {OUTPUT_CSV}")
    print(f"📋 JSON: {SUMMARY_JSON}")
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