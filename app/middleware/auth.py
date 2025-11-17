"""
Authentication Middleware
Kiểm tra API key trong request header
"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import ENABLE_AUTH, API_KEYS

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware kiểm tra API key
    Chỉ áp dụng khi ENABLE_AUTH = true
    """
    
    async def dispatch(self, request: Request, call_next):
        # Bỏ qua authentication cho các endpoint công khai
        public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/api/health"]
        
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Nếu không bật auth, bỏ qua
        if not ENABLE_AUTH:
            return await call_next(request)
        
        # Kiểm tra API key
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Please provide X-API-Key header."
            )
        
        if api_key not in API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key."
            )
        
        # API key hợp lệ, tiếp tục request
        response = await call_next(request)
        return response

