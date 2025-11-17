"""
FastAPI Main Entrypoint
BLIP Vietnamese Captioning API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api.routes_caption import router as caption_router
from app.core.config import (
    API_TITLE, API_VERSION, CORS_ORIGINS,
    ENABLE_AUTH, ENABLE_RATE_LIMIT,
    RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR
)

app = FastAPI(
    title=API_TITLE,
    description="API tạo caption tiếng Việt cho ảnh sản phẩm sử dụng BLIP model",
    version=API_VERSION
)

# CORS middleware để cho phép mobile app gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication middleware
if ENABLE_AUTH:
    from app.middleware.auth import AuthMiddleware
    app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(caption_router, prefix="/api")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "BLIP Vietnamese Captioning API is running 🚀",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/info")
def info():
    """Thông tin về API"""
    from app.core.config import (
        ENABLE_AUTH, ENABLE_CACHE, ENABLE_RATE_LIMIT,
        MAX_BATCH_SIZE, RATE_LIMIT_PER_MINUTE
    )
    
    return {
        "name": "BLIP Vietnamese Captioning API",
        "version": "2.0.0",
        "features": {
            "batch_processing": True,
            "caching": ENABLE_CACHE,
            "authentication": ENABLE_AUTH,
            "rate_limiting": ENABLE_RATE_LIMIT
        },
        "endpoints": {
            "generate_caption": "POST /api/caption (không dấu)",
            "generate_caption_batch": "POST /api/caption/batch (không dấu)",
            "generate_caption_full": "POST /api/caption_full (có dấu - BLIP + Accent Restoration)",
            "generate_caption_full_batch": "POST /api/caption_full/batch (có dấu)",
            "restore_accent": "POST /api/accent/restore (test accent restoration với text)",
            "health_check": "GET /api/health"
        },
        "limits": {
            "max_batch_size": MAX_BATCH_SIZE,
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE if ENABLE_RATE_LIMIT else None
        }
    }

