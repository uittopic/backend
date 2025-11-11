"""
Model Loader Module - Tối ưu cho macOS M1 Pro Max
Sử dụng MPS (Metal Performance Shaders) backend nếu có GPU
"""
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from app.core.config import (
    MODEL_PATH,
    PRETRAINED_MODEL,
    get_device
)

# Load processor và model
print(f"🔄 Đang load model từ {MODEL_PATH}...")
print(f"📱 Device: {get_device()}")

# Kiểm tra xem model đã được train chưa
if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
    processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
    model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    print("✅ Đã load model fine-tuned tiếng Việt")
else:
    # Fallback về pretrained model nếu chưa train
    print("⚠️  Chưa có model fine-tuned, đang load pretrained model...")
    processor = BlipProcessor.from_pretrained(PRETRAINED_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)
    print("✅ Đã load pretrained model (chưa fine-tune tiếng Việt)")

device = get_device()
model.to(device)
model.eval()

print(f"✅ Model đã được load và chuyển sang {device}")

