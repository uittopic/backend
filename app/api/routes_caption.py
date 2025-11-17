"""
API Routes cho Image Captioning
Endpoints: 
- POST /api/caption - Tạo caption cho 1 ảnh
- POST /api/caption/batch - Tạo caption cho nhiều ảnh
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Body
from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent, accent_model
from app.core.config import (
    get_device, MAX_NEW_TOKENS, NUM_BEAMS, EARLY_STOPPING,
    NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY,
    ENABLE_CACHE, CACHE_TTL, MAX_BATCH_SIZE,
    ENABLE_RATE_LIMIT, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR,
    synchronize_device, clear_device_cache
)
from app.utils.cache import (
    get_image_hash, get_cached_caption, set_cached_caption
)
from app.utils.rate_limit import check_rate_limit
from PIL import Image
import torch
import io
import time
from typing import List
from pydantic import BaseModel

router = APIRouter(tags=["Caption"])

# Pydantic models cho request body
class AccentRestoreRequest(BaseModel):
    text: str
    """Text tiếng Việt không dấu cần restore accent"""

def _generate_caption_for_image(image: Image.Image, use_cache: bool = True) -> dict:
    """
    Helper function để generate caption cho một ảnh
    Có tích hợp cache
    
    Args:
        image: PIL Image object
        use_cache: Có sử dụng cache không
    
    Returns:
        Dict với caption và thông tin
    """
    device = get_device()
    # Generation params cho BLIP fine-tuned tiếng Việt
    # Giữ caption tự nhiên, hạn chế lặp từ
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS,
        "early_stopping": EARLY_STOPPING,
        "repetition_penalty": REPETITION_PENALTY,
        "length_penalty": 1.1,  # Khuyến khích độ dài vừa phải
    }

    if NO_REPEAT_NGRAM_SIZE > 0:
        generation_kwargs["no_repeat_ngram_size"] = NO_REPEAT_NGRAM_SIZE

    def _generate_caption() -> str:
        # Tối ưu: Resize ảnh nếu quá lớn (giảm memory usage cho MPS)
        max_size = 512  # Max dimension
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                **generation_kwargs
            )
            # Synchronize device sau khi generate (quan trọng cho MPS)
            synchronize_device()
        
        # Decode bằng tokenizer trực tiếp
        # processor.decode() có thể không hoạt động đúng với tokenizer tùy chỉnh
        if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
            caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            caption = processor.decode(output[0], skip_special_tokens=True)
        
        # Cleanup: Move tensors to CPU trước khi delete để giải phóng memory tốt hơn
        if hasattr(output, 'cpu'):
            output = output.cpu()
        if hasattr(inputs, 'to'):
            inputs = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in inputs.items()}
        del inputs, output
        return caption

    # Kiểm tra cache nếu bật
    if use_cache and ENABLE_CACHE:
        image_hash = get_image_hash(image)
        cached_caption = get_cached_caption(image_hash)

        if cached_caption:
            return {
                "caption_vi": cached_caption,
                "device": device,
                "cached": True
            }

        # Không có trong cache, generate mới
        caption = _generate_caption()
        
        # Cleanup memory sau mỗi inference (quan trọng cho MPS)
        clear_device_cache()

        # Lưu vào cache
        set_cached_caption(image_hash, caption, CACHE_TTL)

        return {
            "caption_vi": caption,
            "device": device,
            "cached": False
        }

    # Không dùng cache
    caption = _generate_caption()
    
    # Cleanup memory sau mỗi inference
    clear_device_cache()

    return {
        "caption_vi": caption,
        "device": device,
        "cached": False
    }

@router.post("/caption")
async def generate_caption(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Tạo caption tiếng Việt cho ảnh sản phẩm (1 ảnh)
    
    Args:
        file: File ảnh (jpg, png, etc.)
    
    Returns:
        JSON với caption tiếng Việt
    """
    # Check rate limit
    check_rate_limit(request)
    
    try:
        # Generate caption (có cache)
        start_time = time.time()
        result = _generate_caption_for_image(image)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "caption_vi": result["caption_vi"],
            "device": result["device"],
            "cached": result.get("cached", False),
            "processing_time": round(processing_time, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

@router.post("/caption/batch")
async def generate_caption_batch(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    Tạo caption tiếng Việt cho nhiều ảnh sản phẩm cùng lúc
    
    Args:
        files: Danh sách file ảnh (tối đa MAX_BATCH_SIZE)
    
    Returns:
        JSON với danh sách caption
    """
    # Check rate limit
    check_rate_limit(request)
    
    try:
        # Kiểm tra số lượng ảnh
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Số lượng ảnh vượt quá giới hạn. Tối đa {MAX_BATCH_SIZE} ảnh mỗi request."
            )
        
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng gửi ít nhất 1 ảnh."
            )
        
        start_time = time.time()
        results = []
        
        # Xử lý từng ảnh
        for index, file in enumerate(files):
            try:
                # Đọc file ảnh
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                
                # Generate caption
                result = _generate_caption_for_image(image)
                
                results.append({
                    "index": index,
                    "filename": file.filename or f"image_{index}.jpg",
                    "caption_vi": result["caption_vi"],
                    "success": True,
                    "cached": result.get("cached", False)
                })
                
                # Cleanup memory sau mỗi ảnh trong batch (quan trọng cho MPS)
                del image, contents
                if index % 5 == 0:  # Clear cache mỗi 5 ảnh
                    clear_device_cache()
                    
            except Exception as e:
                # Nếu một ảnh lỗi, vẫn tiếp tục xử lý ảnh khác
                results.append({
                    "index": index,
                    "filename": file.filename or f"image_{index}.jpg",
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        device = get_device()
        
        # Final cleanup sau khi xử lý xong batch
        clear_device_cache()
        
        return {
            "success": True,
            "total": len(files),
            "results": results,
            "device": device,
            "processing_time": round(processing_time, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý batch: {str(e)}")

def _generate_caption_with_accent(image: Image.Image, use_cache: bool = True) -> dict:
    """
    Helper function để generate caption CÓ DẤU cho một ảnh
    Pipeline: BLIP (không dấu) → ViT5 (có dấu)
    
    Args:
        image: PIL Image object
        use_cache: Có sử dụng cache không
    
    Returns:
        Dict với caption có dấu và thông tin
    """
    device = get_device()
    
    # Bước 1: Generate caption không dấu bằng BLIP
    blip_result = _generate_caption_for_image(image, use_cache=use_cache)
    caption_no_accent = blip_result["caption_vi"]
    
    # Bước 2: Restore accent bằng ViT5
    if accent_model is None:
        # Nếu accent model chưa load được, trả về caption không dấu
        return {
            "caption_vi": caption_no_accent,
            "caption_vi_no_accent": caption_no_accent,
            "accent_restored": False,
            "device": device,
            "cached": blip_result.get("cached", False)
        }
    
    # Restore accent
    caption_with_accent = restore_accent(caption_no_accent)
    
    return {
        "caption_vi": caption_with_accent,
        "caption_vi_no_accent": caption_no_accent,
        "accent_restored": True,
        "device": device,
        "cached": blip_result.get("cached", False)
    }

@router.post("/caption_full")
async def generate_caption_full(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Tạo caption tiếng Việt CÓ DẤU cho ảnh sản phẩm (1 ảnh)
    Pipeline: BLIP (không dấu) → ViT5 Accent Restoration (có dấu)
    
    Args:
        file: File ảnh (jpg, png, etc.)
    
    Returns:
        JSON với caption tiếng Việt có dấu
    """
    # Check rate limit
    check_rate_limit(request)
    
    try:
        # Đọc file ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate caption có dấu (có cache cho BLIP)
        start_time = time.time()
        result = _generate_caption_with_accent(image)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "caption_vi": result["caption_vi"],
            "caption_vi_no_accent": result.get("caption_vi_no_accent", ""),
            "accent_restored": result.get("accent_restored", False),
            "device": result["device"],
            "cached": result.get("cached", False),
            "processing_time": round(processing_time, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

@router.post("/caption_full/batch")
async def generate_caption_full_batch(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    Tạo caption tiếng Việt CÓ DẤU cho nhiều ảnh sản phẩm cùng lúc
    Pipeline: BLIP (không dấu) → ViT5 Accent Restoration (có dấu)
    
    Args:
        files: Danh sách file ảnh (tối đa MAX_BATCH_SIZE)
    
    Returns:
        JSON với danh sách caption có dấu
    """
    # Check rate limit
    check_rate_limit(request)
    
    try:
        # Kiểm tra số lượng ảnh
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Số lượng ảnh vượt quá giới hạn. Tối đa {MAX_BATCH_SIZE} ảnh mỗi request."
            )
        
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng gửi ít nhất 1 ảnh."
            )
        
        start_time = time.time()
        results = []
        
        # Xử lý từng ảnh
        for index, file in enumerate(files):
            try:
                # Đọc file ảnh
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                
                # Generate caption có dấu
                result = _generate_caption_with_accent(image)
                
                results.append({
                    "index": index,
                    "filename": file.filename or f"image_{index}.jpg",
                    "caption_vi": result["caption_vi"],
                    "caption_vi_no_accent": result.get("caption_vi_no_accent", ""),
                    "accent_restored": result.get("accent_restored", False),
                    "success": True,
                    "cached": result.get("cached", False)
                })
                
                # Cleanup memory sau mỗi ảnh trong batch (quan trọng cho MPS)
                del image, contents
                if index % 5 == 0:  # Clear cache mỗi 5 ảnh
                    clear_device_cache()
                    
            except Exception as e:
                # Nếu một ảnh lỗi, vẫn tiếp tục xử lý ảnh khác
                results.append({
                    "index": index,
                    "filename": file.filename or f"image_{index}.jpg",
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        device = get_device()
        
        # Final cleanup sau khi xử lý xong batch
        clear_device_cache()
        
        return {
            "success": True,
            "total": len(files),
            "results": results,
            "device": device,
            "processing_time": round(processing_time, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý batch: {str(e)}")

@router.post("/accent/restore")
async def restore_accent_text(
    request: Request,
    body: AccentRestoreRequest = Body(...)
):
    """
    Restore accent cho text tiếng Việt không dấu
    Endpoint này dùng để test accent restoration model trực tiếp (không cần ảnh)
    
    Args:
        body: JSON với field "text" (text không dấu)
    
    Returns:
        JSON với text đã được restore accent
    
    Example:
        Input: {"text": "ao khoac the thao mau den"}
        Output: {"text": "áo khoác thể thao màu đen"}
    """
    # Check rate limit
    check_rate_limit(request)
    
    try:
        if not body.text or not body.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp text cần restore accent"
            )
        
        # Restore accent
        start_time = time.time()
        text_with_accent = restore_accent(body.text.strip())
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "text_no_accent": body.text.strip(),
            "text_with_accent": text_with_accent,
            "device": get_device(),
            "accent_model_loaded": accent_model is not None,
            "processing_time": round(processing_time, 3)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi restore accent: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Kiểm tra trạng thái API và model"""
    from app.utils.cache import get_cache_stats
    
    device = get_device()
    cache_stats = get_cache_stats() if ENABLE_CACHE else None
    
    return {
        "status": "healthy",
        "device": device,
        "blip_model_loaded": model is not None,
        "accent_model_loaded": accent_model is not None,
        "cache_enabled": ENABLE_CACHE,
        "cache_stats": cache_stats
    }

