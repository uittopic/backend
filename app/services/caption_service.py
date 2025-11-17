"""
Caption service helpers.

Tách logic generate caption ra khỏi lớp route để dễ tái sử dụng và tối ưu bộ nhớ
cho macOS (MPS) cũng như các backend khác.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from PIL import Image

from app.core.accent_restoration_loader import accent_model, restore_accent
from app.core.config import (
    CACHE_TTL,
    ENABLE_CACHE,
    MAX_IMAGE_SIZE,
    MAX_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
    REPETITION_PENALTY,
    EARLY_STOPPING,
    clear_device_cache,
    get_device,
    synchronize_device,
)
from app.core.model_loader import model, processor
from app.utils.cache import get_cached_caption, get_image_hash, set_cached_caption

# Pillow 10 đổi tên constant, fallback để tránh crash khi chạy trên mac cũ
RESAMPLING = getattr(Image, "Resampling", Image)

GENERATION_KWARGS: Dict[str, object] = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": NUM_BEAMS,
    "early_stopping": EARLY_STOPPING,
    "repetition_penalty": REPETITION_PENALTY,
    "length_penalty": 1.1,
}
if NO_REPEAT_NGRAM_SIZE > 0:
    GENERATION_KWARGS["no_repeat_ngram_size"] = NO_REPEAT_NGRAM_SIZE


@dataclass
class CaptionResult:
    caption_vi: str
    device: str
    cached: bool = False
    caption_vi_no_accent: Optional[str] = None
    accent_restored: bool = False

    def to_payload(self) -> Dict[str, object]:
        payload = asdict(self)
        # Không expose key None để response gọn hơn
        if self.caption_vi_no_accent is None:
            payload.pop("caption_vi_no_accent", None)
        return payload


def load_image_from_bytes(contents: bytes) -> Image.Image:
    """Chuẩn hóa ảnh từ bytes input."""
    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _prepare_image(image: Image.Image) -> Image.Image:
    """Resize ảnh nếu quá lớn để tiết kiệm VRAM (đặc biệt cho MPS)."""
    if MAX_IMAGE_SIZE and max(image.size) > MAX_IMAGE_SIZE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), RESAMPLING.LANCZOS)
    return image


def _run_blip(image: Image.Image) -> str:
    """Chạy BLIP model để sinh caption không dấu."""
    device = get_device()
    prepared_image = _prepare_image(image)
    inputs = processor(images=prepared_image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, **GENERATION_KWARGS)
        synchronize_device()

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        caption = processor.decode(output[0], skip_special_tokens=True)

    # Cleanup tensors: move về CPU rồi release
    output = output.detach().cpu()
    del output
    inputs = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in inputs.items()}
    del inputs

    return caption.strip()


def generate_caption_for_image(image: Image.Image, use_cache: bool = True) -> CaptionResult:
    """Sinh caption tiếng Việt không dấu với cache optional."""
    device = get_device()
    cache_key = get_image_hash(image) if (use_cache and ENABLE_CACHE) else None

    if cache_key:
        cached = get_cached_caption(cache_key)
        if cached:
            return CaptionResult(
                caption_vi=cached,
                device=device,
                cached=True,
            )

    caption = _run_blip(image)
    clear_device_cache()

    if cache_key:
        set_cached_caption(cache_key, caption, CACHE_TTL)

    return CaptionResult(
        caption_vi=caption,
        device=device,
        cached=False,
    )


def generate_caption_with_accent(image: Image.Image, use_cache: bool = True) -> CaptionResult:
    """Sinh caption có dấu bằng pipeline BLIP + Accent restoration."""
    base_result = generate_caption_for_image(image, use_cache=use_cache)

    if accent_model is None:
        return CaptionResult(
            caption_vi=base_result.caption_vi,
            caption_vi_no_accent=base_result.caption_vi,
            accent_restored=False,
            device=base_result.device,
            cached=base_result.cached,
        )

    caption_with_accent = restore_accent(base_result.caption_vi)
    return CaptionResult(
        caption_vi=caption_with_accent,
        caption_vi_no_accent=base_result.caption_vi,
        accent_restored=True,
        device=base_result.device,
        cached=base_result.cached,
    )

