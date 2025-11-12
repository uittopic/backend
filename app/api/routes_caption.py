"""
API Routes cho Image Captioning
Endpoint: POST /api/caption
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.model_loader import model, processor
from app.core.config import (
    get_device, MAX_NEW_TOKENS, NUM_BEAMS, EARLY_STOPPING,
    TEMPERATURE, REPETITION_PENALTY, LENGTH_PENALTY
)
from PIL import Image
import torch
import io

router = APIRouter(tags=["Caption"])

@router.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Tạo caption tiếng Việt cho ảnh sản phẩm
    
    Args:
        file: File ảnh (jpg, png, etc.)
    
    Returns:
        JSON với caption tiếng Việt
    """
    try:
        # Đọc file ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Xử lý ảnh với processor
        device = get_device()
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate caption với parameters tối ưu cho tiếng Việt có dấu
        # Sử dụng beam search với các penalties để cải thiện chất lượng output
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=EARLY_STOPPING,
                repetition_penalty=REPETITION_PENALTY,
                length_penalty=LENGTH_PENALTY,
                pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode caption
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return {
            "success": True,
            "caption_vi": caption,
            "device": device
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

@router.get("/health")
async def health_check():
    """Kiểm tra trạng thái API và model"""
    device = get_device()
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

