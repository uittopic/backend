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


@router.post("/caption")
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


@router.post("/caption/batch")
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


@router.post("/caption_full")
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


@router.post("/caption_full/batch")
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


@router.post("/accent/restore")
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


@router.get("/health")
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


@router.post("/cache/clear")
async def clear_cache_endpoint():
    """Xóa toàn bộ cache."""
    clear_cache()
    return {
        "success": True,
        "message": "Cache đã được xóa",
    }
