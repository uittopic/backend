"""
Rate Limiting Utility
Helper functions để apply rate limiting
"""
from fastapi import Request
from app.core.config import ENABLE_RATE_LIMIT

def check_rate_limit(request: Request):
    """
    Kiểm tra rate limit cho request
    Với slowapi, rate limiting được handle tự động bởi exception handler
    Function này chỉ để đảm bảo limiter được setup đúng
    """
    if not ENABLE_RATE_LIMIT:
        return
    
    # Rate limiting được handle bởi slowapi exception handler
    # Nếu vượt quá, RateLimitExceeded sẽ được raise và handle bởi main.py
    pass

