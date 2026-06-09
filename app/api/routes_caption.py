"""
API Routes cho Image Captioning.

Endpoints:
- POST /api/caption
- POST /api/caption/batch
- POST /api/caption_full
- POST /api/caption_full/batch
- POST /api/accent/restore
- GET /api/health
- POST /api/cache/clear
"""
from __future__ import annotations

import time
from typing import List, Optional, Union
from PIL import Image

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.accent_restoration_loader import accent_model, restore_accent
from app.core.config import MAX_BATCH_SIZE, ENABLE_CACHE, clear_device_cache, get_device
from app.core.model_loader import model
from app.services.caption_service import (
    generate_caption_for_image,
    generate_caption_with_accent,
    load_image_from_bytes,
)
from app.utils.cache import clear_cache, get_cache_stats
from app.utils.rate_limit import check_rate_limit

router = APIRouter(tags=["Caption"])


class AccentRestoreRequest(BaseModel):
    text: str


async def _read_upload_image(file: UploadFile) -> Image.Image:
    """Đọc UploadFile và convert thành PIL image."""
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="File ảnh rỗng. Vui lòng chọn file ảnh hợp lệ.",
        )
    return load_image_from_bytes(contents)


def _validate_batch(files: List[UploadFile]) -> None:
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Số lượng ảnh vượt quá giới hạn. Tối đa {MAX_BATCH_SIZE} ảnh mỗi request.",
        )
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng gửi ít nhất 1 ảnh.",
        )


@router.post(
    "/caption",
    summary="Tạo caption không dấu (1 ảnh)",
    description="""
Tạo caption tiếng Việt **không dấu** cho 1 ảnh sản phẩm.

**Pipeline:** `Image → BLIP → Caption không dấu`

**Output:** Caption tiếng Việt không dấu (ví dụ: `san pham thoi trang mau hong`)

**Cache:** Kết quả được cache theo MD5 hash của ảnh (TTL 24h). Ảnh đã gửi trước đó → trả kết quả tức thì.

**Metric chất lượng:** BLEU-4 = 0.5646, SBERT = 0.8109 (no-accent)
    """,
    response_description="Caption không dấu thành công",
    responses={
        200: {"description": "Thành công, trả về caption"},
        400: {"description": "File ảnh rỗng hoặc không hợp lệ"},
        429: {"description": "Vượt rate limit (30 req/phút)"},
        500: {"description": "Lỗi xử lý ảnh"},
    },
)
async def generate_caption(
    request: Request,
    file: UploadFile = File(...),
):
    """Tạo caption tiếng Việt không dấu cho 1 ảnh."""
    check_rate_limit(request)
    try:
        image = await _read_upload_image(file)
        start_time = time.time()
        result = generate_caption_for_image(image=image, use_cache=True)
        return {
            "success": True,
            "caption_vi": result.caption_vi,
            "device": result.device,
            "cached": result.cached,
            "processing_time": round(time.time() - start_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")


@router.post(
    "/caption/batch",
    summary="Tạo caption không dấu (nhiều ảnh)",
    description="""
Tạo caption tiếng Việt **không dấu** cho nhiều ảnh cùng lúc.

**Pipeline:** `Image → BLIP → Caption không dấu`

**Giới hạn:** Tối đa **5 ảnh/request**

**Cache:** Mỗi ảnh được cache riêng theo MD5 hash (TTL 24h).

**Metric chất lượng:** BLEU-4 = 0.5646, SBERT = 0.8109 (no-accent)
    """,
    response_description="Danh sách caption không dấu",
    responses={
        200: {"description": "Thành công, trả về danh sách caption"},
        400: {"description": "Số ảnh vượt quá giới hạn (max 5) hoặc không có ảnh"},
        429: {"description": "Vượt rate limit (30 req/phút)"},
        500: {"description": "Lỗi xử lý batch"},
    },
)
async def generate_caption_batch(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """Tạo caption tiếng Việt không dấu cho nhiều ảnh."""
    check_rate_limit(request)
    _validate_batch(files)

    try:
        start_time = time.time()
        results = []

        for index, file in enumerate(files):
            try:
                image = await _read_upload_image(file)
                result = generate_caption_for_image(image=image, use_cache=True)
                results.append(
                    {
                        "index": index,
                        "filename": file.filename or f"image_{index}.jpg",
                        "caption_vi": result.caption_vi,
                        "success": True,
                        "cached": result.cached,
                    }
                )

                if index % 5 == 0:
                    clear_device_cache()
            except Exception as e:
                results.append(
                    {
                        "index": index,
                        "filename": file.filename or f"image_{index}.jpg",
                        "success": False,
                        "error": str(e),
                    }
                )

        clear_device_cache()
        return {
            "success": True,
            "total": len(files),
            "results": results,
            "device": get_device(),
            "processing_time": round(time.time() - start_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý batch: {str(e)}")


@router.post(
    "/caption_full",
    summary="Tạo caption có dấu (1 ảnh)",
    description="""
Tạo caption tiếng Việt **có dấu** cho 1 ảnh sản phẩm.

**Pipeline:** `Image → BLIP → Caption không dấu → Accent Restoration → Caption có dấu`

**Output:** 
- `caption_vi`: Caption tiếng Việt có dấu (ví dụ: `sản phẩm thời trang màu hồng`)
- `caption_vi_no_accent`: Caption không dấu gốc từ BLIP
- `accent_restored`: True nếu accent model thành công

**Chú ý:** Accent restoration dùng XLM-RoBERTa pretrained trên VNCOMMON dataset. Kết quả có thể khác annotation standard của Shopee.

**Metric chất lượng:** SBERT (no-accent) = 0.8109, SBERT (with-accent) = 0.361
    """,
    response_description="Caption có dấu thành công",
    responses={
        200: {"description": "Thành công, trả về caption có dấu"},
        400: {"description": "File ảnh rỗng hoặc không hợp lệ"},
        429: {"description": "Vượt rate limit (30 req/phút)"},
        500: {"description": "Lỗi xử lý ảnh"},
    },
)
async def generate_caption_full(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Tạo caption tiếng Việt có dấu cho 1 ảnh.
    Pipeline: BLIP (không dấu) -> Accent restoration (có dấu)
    """
    check_rate_limit(request)
    try:
        if not file or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng gửi file ảnh. Field name phải là 'file' và type phải là 'File' (không phải 'Text').",
            )

        image = await _read_upload_image(file)
        start_time = time.time()
        result = generate_caption_with_accent(image=image, use_cache=True)
        return {
            "success": True,
            "caption_vi": result.caption_vi,
            "caption_vi_no_accent": result.caption_vi_no_accent or "",
            "accent_restored": result.accent_restored,
            "device": result.device,
            "cached": result.cached,
            "processing_time": round(time.time() - start_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")


@router.post(
    "/caption_full/batch",
    summary="Tạo caption có dấu (nhiều ảnh)",
    description="""
Tạo caption tiếng Việt **có dấu** cho nhiều ảnh cùng lúc.

**Pipeline:** `Image → BLIP → Caption không dấu → Accent Restoration → Caption có dấu`

**Giới hạn:** Tối đa **5 ảnh/request**

**Chú ý:** Accent restoration dùng XLM-RoBERTa pretrained trên VNCOMMON dataset. Kết quả có thể khác annotation standard của Shopee.
    """,
    response_description="Danh sách caption có dấu",
    responses={
        200: {"description": "Thành công, trả về danh sách caption"},
        400: {"description": "Số ảnh vượt quá giới hạn (max 5) hoặc không có ảnh"},
        429: {"description": "Vượt rate limit (30 req/phút)"},
        500: {"description": "Lỗi xử lý batch"},
    },
)
async def generate_caption_full_batch(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """
    Tạo caption tiếng Việt có dấu cho nhiều ảnh.
    Pipeline: BLIP (không dấu) -> Accent restoration (có dấu)
    """
    check_rate_limit(request)
    _validate_batch(files)

    try:
        start_time = time.time()
        results = []

        for index, file in enumerate(files):
            try:
                image = await _read_upload_image(file)
                result = generate_caption_with_accent(image=image, use_cache=True)
                results.append(
                    {
                        "index": index,
                        "filename": file.filename or f"image_{index}.jpg",
                        "caption_vi": result.caption_vi,
                        "caption_vi_no_accent": result.caption_vi_no_accent or "",
                        "accent_restored": result.accent_restored,
                        "success": True,
                        "cached": result.cached,
                    }
                )

                if index % 5 == 0:
                    clear_device_cache()
            except Exception as e:
                results.append(
                    {
                        "index": index,
                        "filename": file.filename or f"image_{index}.jpg",
                        "success": False,
                        "error": str(e),
                    }
                )

        clear_device_cache()
        return {
            "success": True,
            "total": len(files),
            "results": results,
            "device": get_device(),
            "processing_time": round(time.time() - start_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý batch: {str(e)}")


@router.post(
    "/accent/restore",
    summary="Gán dấu tiếng Việt cho text",
    description="""
Gán dấu tiếng Việt cho text không dấu.

**Model:** XLM-RoBERTa (peterhung/vietnamese-accent-marker-xlm-roberta)

**Ví dụ:**
- Input: `san pham thoi trang mau hong`
- Output: `sản phẩm thời trang màu hồng`

**Use case:** Test accent restoration trước khi dùng trong pipeline.

**Chú ý:** Accent model pretrained trên VNCOMMON dataset. Kết quả có thể khác annotation standard của Shopee.
    """,
    response_description="Text đã được gán dấu",
    responses={
        200: {"description": "Thành công"},
        400: {"description": "Text rỗng"},
        429: {"description": "Vượt rate limit (30 req/phút)"},
        500: {"description": "Lỗi accent restoration"},
    },
)
async def restore_accent_text(
    request: Request,
    body: AccentRestoreRequest = Body(...),
):
    """Restore dấu cho text tiếng Việt không dấu."""
    check_rate_limit(request)
    try:
        if not body.text or not body.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp text cần restore accent",
            )

        start_time = time.time()
        text_no_accent = body.text.strip()
        text_with_accent = restore_accent(text_no_accent)
        return {
            "success": True,
            "text_no_accent": text_no_accent,
            "text_with_accent": text_with_accent,
            "device": get_device(),
            "accent_model_loaded": accent_model is not None,
            "processing_time": round(time.time() - start_time, 3),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi restore accent: {str(e)}")


@router.get(
    "/health",
    summary="Kiểm tra trạng thái API",
    description="""
Kiểm tra trạng thái hoạt động của API và các thành phần:

- **BLIP model**: Đã load hay chưa
- **Accent model**: Đã load hay chưa
- **Cache**: Số entries, expired entries
- **Device**: MPS/CPU/CUDA

**Dùng để:** Health check, monitoring, debug
    """,
    response_description="Trạng thái API",
    responses={
        200: {"description": "API đang hoạt động"},
    },
)
async def health_check():
    """Kiểm tra trạng thái API và model."""
    return {
        "status": "healthy",
        "device": get_device(),
        "blip_model_loaded": model is not None,
        "accent_model_loaded": accent_model is not None,
        "cache_enabled": ENABLE_CACHE,
        "cache_stats": get_cache_stats() if ENABLE_CACHE else None,
    }


@router.post(
    "/cache/clear",
    summary="Xóa toàn bộ cache",
    description="""
Xóa tất cả cached captions.

**Dùng để:** 
- Reset cache khi model được cập nhật
- Debug cache behavior
- Free memory

**Cache info:** 
- TTL: 24 giờ
- Key: MD5 hash của ảnh
- Max entries: Unlimited (tùy memory)
    """,
    response_description="Cache đã được xóa",
    responses={
        200: {"description": "Xóa cache thành công"},
    },
)
async def clear_cache_endpoint():
    """Xóa toàn bộ cache."""
    clear_cache()
    return {
        "success": True,
        "message": "Cache đã được xóa",
    }
