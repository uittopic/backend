"""
Cache Utility Module
Cache kết quả caption để tăng tốc độ response
"""
import hashlib
import time
from typing import Optional, Dict, Any
from PIL import Image
import io

# In-memory cache (có thể thay bằng Redis)
_cache: Dict[str, Dict[str, Any]] = {}

def get_image_hash(image: Image.Image) -> str:
    """
    Tính hash của ảnh để dùng làm cache key
    
    Args:
        image: PIL Image object
    
    Returns:
        MD5 hash string
    """
    # Convert ảnh thành bytes để hash
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Tính MD5 hash
    hash_obj = hashlib.md5(img_bytes.read())
    return hash_obj.hexdigest()

def get_cached_caption(image_hash: str) -> Optional[str]:
    """
    Lấy caption từ cache nếu có
    
    Args:
        image_hash: Hash của ảnh
    
    Returns:
        Caption nếu có trong cache, None nếu không
    """
    if image_hash not in _cache:
        return None
    
    cache_entry = _cache[image_hash]
    
    # Kiểm tra TTL
    if time.time() > cache_entry['expires_at']:
        # Cache đã hết hạn, xóa đi
        del _cache[image_hash]
        return None
    
    return cache_entry['caption']

def set_cached_caption(image_hash: str, caption: str, ttl: int = 86400):
    """
    Lưu caption vào cache
    
    Args:
        image_hash: Hash của ảnh
        caption: Caption cần cache
        ttl: Time to live (giây), mặc định 24 giờ
    """
    _cache[image_hash] = {
        'caption': caption,
        'expires_at': time.time() + ttl,
        'created_at': time.time()
    }

def clear_cache():
    """Xóa toàn bộ cache"""
    global _cache
    _cache.clear()

def get_cache_stats() -> Dict[str, Any]:
    """
    Lấy thống kê về cache
    
    Returns:
        Dict với thông tin cache
    """
    now = time.time()
    valid_entries = sum(1 for entry in _cache.values() if entry['expires_at'] > now)
    expired_entries = len(_cache) - valid_entries
    
    # Xóa các entry đã hết hạn
    if expired_entries > 0:
        keys_to_delete = [
            key for key, entry in _cache.items()
            if entry['expires_at'] <= now
        ]
        for key in keys_to_delete:
            del _cache[key]
    
    return {
        'total_entries': len(_cache),
        'valid_entries': valid_entries,
        'expired_entries': expired_entries
    }

