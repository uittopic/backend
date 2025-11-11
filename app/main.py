"""
FastAPI Main Entrypoint
BLIP Vietnamese Captioning API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_caption import router as caption_router
from app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS

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
    return {
        "name": "BLIP Vietnamese Captioning API",
        "version": "1.0.0",
        "endpoints": {
            "generate_caption": "POST /api/caption",
            "health_check": "GET /api/health"
        }
    }

