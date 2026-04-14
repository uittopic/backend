"""
Configuration Module
Quản lý tất cả các cấu hình của ứng dụng
"""
import os
from pathlib import Path

# Load .env file nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Không bắt buộc phải có python-dotenv

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "Salesforce/blip-image-captioning-base")
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/blip_vietnamese_80_20")
PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", "Salesforce/blip-image-captioning-base")

# Accent Restoration Model Configuration
ACCENT_MODEL_NAME = os.getenv("ACCENT_MODEL_NAME", "peterhung/vietnamese-accent-marker-xlm-roberta")
ACCENT_MODEL_PATH = BASE_DIR / os.getenv("ACCENT_MODEL_PATH", "models/accent_restoration")

# Data Configuration
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "train_bilingual_clean_v2.csv"

# Training Configuration
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "2"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "500"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "77"))

# API Configuration
API_TITLE = os.getenv("API_TITLE", "BLIP Vietnamese Captioning API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Generation Configuration
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "5"))  # Tối ưu từ grid search (baseline 3 → 5 cải thiện BLEU +3.8%)
EARLY_STOPPING = os.getenv("EARLY_STOPPING", "true").lower() == "true"
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))

# Logging Configuration
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Device Configuration (tự động detect, cho phép override)
DEVICE_OVERRIDE = os.getenv("DEVICE", "cpu").lower()

def get_device() -> str:
    """
    Xác định device mặc định cho inference.
    Ưu tiên dùng giá trị override từ env (DEVICE).
    Nếu không override, fallback: MPS > CUDA > CPU.
    """
    import torch

    if DEVICE_OVERRIDE in {"cpu", "cuda", "mps"}:
        return DEVICE_OVERRIDE

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def clear_device_cache():
    """
    Xóa cache của device (MPS/CUDA) để giải phóng memory
    Tối ưu cho macOS M1 - MPS memory management
    """
    import torch
    device = get_device()
    if device == "mps":
        # MPS không có empty_cache() như CUDA, nhưng có thể dùng synchronize()
        try:
            torch.mps.synchronize()
        except AttributeError:
            # Fallback nếu không có synchronize
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def synchronize_device():
    """
    Đồng bộ device để đảm bảo tất cả operations đã hoàn thành
    Quan trọng cho MPS trên macOS
    """
    import torch
    device = get_device()
    if device == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
    elif device == "cuda":
        torch.cuda.synchronize()

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Batch Processing Configuration
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))

# Caching Configuration
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 giờ (giây)
REDIS_URL = os.getenv("REDIS_URL", None)  # None = dùng in-memory cache

# Authentication Configuration
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]  # Remove empty strings

# Rate Limiting Configuration
ENABLE_RATE_LIMIT = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

# Tạo thư mục nếu chưa có
MODEL_PATH.mkdir(parents=True, exist_ok=True)
ACCENT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
