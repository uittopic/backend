"""
Configuration Module
Quản lý tất cả các cấu hình của ứng dụng
"""
import os
from pathlib import Path
from typing import Optional

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
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/blip_vietnamese")
PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", "Salesforce/blip-image-captioning-base")

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
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "3"))
EARLY_STOPPING = os.getenv("EARLY_STOPPING", "true").lower() == "true"

# Logging Configuration
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Device Configuration (tự động detect)
def get_device() -> str:
    """
    Xác định device tối ưu
    Ưu tiên: MPS (macOS M1) > CUDA > CPU
    """
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Tạo thư mục nếu chưa có
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

