"""
Model Loader Module - BLIP Vietnamese Captioning (không dấu)
Tối ưu cho macOS M1 Pro Max - Sử dụng MPS (Metal Performance Shaders) backend nếu có GPU
Model này tạo caption tiếng Việt KHÔNG DẤU, sau đó sẽ được xử lý bởi Accent Restoration Model
"""
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from app.core.config import MODEL_PATH, PRETRAINED_MODEL, get_device, synchronize_device

# Type aliases
BlipModel = BlipForConditionalGeneration
BlipProcessorType = BlipProcessor

# Load processor và model BLIP (caption không dấu)
print(f"🔄 Đang load BLIP model từ {MODEL_PATH}...")
device = get_device()
print(f"📱 Device: {device}")

# Kiểm tra xem model đã được train chưa
if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
    # Load processor và model
    processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
    model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    print("✅ Đã load BLIP model fine-tuned tiếng Việt (không dấu)")
else:
    # Fallback về pretrained model nếu chưa train
    print("⚠️  Chưa có model fine-tuned, đang load pretrained model...")
    processor = BlipProcessor.from_pretrained(PRETRAINED_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)
    print("✅ Đã load pretrained model (chưa fine-tune tiếng Việt)")

# Chuyển model sang device và tối ưu
blip_model: BlipForConditionalGeneration = model  # type: ignore
blip_model.to(device)  # type: ignore
blip_model.eval()

# Tối ưu cho MPS: Set model to half precision nếu không phải MPS (MPS chưa hỗ trợ tốt fp16)
# Giữ nguyên float32 cho MPS để đảm bảo stability
if device != "mps":
    # Có thể dùng half precision cho CUDA nếu cần
    pass

# Synchronize device sau khi load model (quan trọng cho MPS)
synchronize_device()

print(f"✅ BLIP model đã được load và chuyển sang {device}")
print("📝 Model này tạo caption tiếng Việt KHÔNG DẤU")
print("💡 Sử dụng /api/caption_full để có caption CÓ DẤU (qua Accent Restoration)")

# Tối ưu: Compile model nếu PyTorch 2.0+ (tăng tốc inference)
try:
    if hasattr(torch, 'compile') and device != "mps":
        # torch.compile() chưa hỗ trợ tốt MPS, chỉ dùng cho CUDA/CPU
        print("⚡ Đang compile model để tăng tốc inference...")
        blip_model = torch.compile(blip_model, mode="reduce-overhead")  # type: ignore
        print("✅ Model đã được compile")
except Exception as e:
    # Nếu không compile được, vẫn dùng model bình thường
    pass

