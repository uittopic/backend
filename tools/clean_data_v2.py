"""
Data Cleaning Script v2 cho Vietnamese Product Caption
=========================================================

Cải tiến so với v1:
1. Loại SEO spam keywords mạnh hơn
2. Xử lý repetition liên tiếp (KHÔNG xóa toàn bộ từ trùng)
3. Keep important brands (joyko, casio, samsung, etc.)
4. Remove English stopwords nhưng giữ product-specific English
5. Chuẩn hóa category keywords
6. Filter captions nhiễu nặng

Usage:
    python tools/clean_data_v2.py \
        --input data/train_80_cleaned.csv \
        --output data/train_80_cleaned_v2.csv \
        --min-len 8 \
        --max-en-ratio 0.35
"""

import re
import csv
import argparse
import unicodedata
from pathlib import Path
from typing import List, Tuple, Set


# ============================================================
# SECTION 1: SEO SPAM KEYWORDS - CẦN LOẠI BỎ
# ============================================================
SEO_SPAM_KEYWORDS: Set[str] = {
    # Tiếng Anh
    'free', 'sale', 'hot', 'best', 'new', 'newest', 'latest',
    'original', 'import', 'authentic', 'genuine', 'premium', 'quality',
    'best seller', 'bestseller', 'best-seller', 'top', 'popular',
    'ready', 'available', 'in stock', 'limited', 'promo', 'promotion',
    'discount', 'cheap', 'affordable', 'murah', 'gratis',
    'ongkir', 'free ongkir', 'free-ship', 'freeship',
    ' COD ', 'cod', 'bayar', 'transfer',
    'terlaris', 'terbaik', 'terpercaya', 'terjamin',
    'official', 'store', 'shop', 'boutique',
    # Tiếng Việt
    'giá rẻ', 'miễn phí', 'khuyến mãi', 'giảm giá', 'hotsale',
    'bán chạy', 'nổi bật', 'yêu thích', 'hàng mới', 'về gấp',
    'đáng mua', 'siêu rẻ', 'cực rẻ', 'mới nhất',
    # Marketing
    'super', 'mega', 'ultra', 'plus', 'pro', 'max',
    'big sale', 'flash sale', 'clearance', 'outlet',
}


# ============================================================
# SECTION 2: ENGLISH STOPWORDS - CẦN LOẠI BỎ
# ============================================================
ENGLISH_STOPWORDS: Set[str] = {
    # Articles
    'a', 'an', 'the',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
    # Conjunctions
    'and', 'or', 'but', 'so', 'yet', 'nor',
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    # Verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'can', 'must', 'shall',
    # Others
    'this', 'that', 'these', 'those', 'here', 'there', 'when', 'where', 'which',
    'who', 'whom', 'whose', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'own', 'same', 'than', 'too', 'very', 'just', 'also', 'now', 'then',
}


# ============================================================
# SECTION 3: BRANDS & IMPORTANT ENGLISH WORDS - GIỮ LẠI
# ============================================================
IMPORTANT_BRANDS: Set[str] = {
    # Brands thường gặp trong e-commerce Việt Nam
    'joyko', 'casio', 'samsung', 'xiaomi', 'redmi', 'oppo', 'vivo', 'huawei',
    'apple', 'iphone', 'ipad', 'macbook', 'sony', 'lg', 'philips', 'panasonic',
    'sharp', 'toshiba', 'hitachi', 'crown', 'nivea', 'vaseline', 'ponds',
    'maybelline', 'mac', 'clinique', 'loreal', 'maybelline', 'innisfree',
    'the body shop', 'body shop',
    # Common English words trong sản phẩm Việt Nam - PRODUCT-SPECIFIC
    'set', 'combo', 'pack', 'mini', 'pro', 'max', 'plus', 'slim', 'wide',
    'auto', 'manual', 'electric', 'digital', 'smart', 'wireless', 'bluetooth',
    'led', 'lcd', 'oled', 'touch', 'screen', 'display',
    'waterproof', 'water resistant', 'splash',
    'baby', 'kids', 'children', 'men', 'women', 'unisex',
    'size', 'small', 'medium', 'large', 'xl', 'xxl', 'free size', 'one size',
    # Units
    'ml', 'mm', 'cm', 'kg', 'gram', 'g', 'l', 'liter', 'pcs', 'pc', 'set',
    'inch', 'volt', 'watt', 'amp', 'mah', 'gb', 'tb',
    # PRODUCT-SPECIFIC ENGLISH - Các từ thường xuất hiện trong caption thương mại
    'shirt', 'blouse', 'dress', 'skirt', 'pants', 'shorts', 'jeans',
    'jacket', 'hoodie', 'sweater', 'cardigan', 'vest',
    'sandal', 'slipper', 'heels', 'boots', 'sneakers', 'flat',
    'bag', 'wallet', 'backpack', 'clutch', 'tote', 'purse',
    'watch', 'necklace', 'bracelet', 'earring', 'ring', 'jewelry',
    'lipstick', 'mascara', 'foundation', 'powder', 'concealer', 'blush',
    'serum', 'cream', 'lotion', 'toner', 'cleanser', 'mask', 'scrub',
    'shampoo', 'conditioner', 'hair', 'wig',
    'brush', 'comb', 'mirror', 'organizer', 'container',
    'bottle', 'jar', 'spray', 'pump',
    'glasses', 'sunglasses', 'lens',
    'strap', 'cover', 'case', 'holder', 'stand',
    'cable', 'charger', 'adapter', 'usb', 'hdmi',
    'earphone', 'headphone', 'speaker', 'microphone',
    'powerbank', 'battery', 'cordless',
    'blender', 'mixer', 'grinder', 'chopper', 'slicer',
    'pan', 'pot', 'pot', 'steamer', 'kettle',
    'knife', 'scissors', 'cutlery',
    'mat', 'cover', 'blanket', 'pillow', 'cushion',
    'toy', 'game', 'puzzle',
    'book', 'notebook', 'diary', 'planner',
    'stick', 'tape', 'glue', ' sealant',
    'door', 'window', 'lock', 'hinge',
    'filter', 'cartridge', 'refill',
    'sensor', 'alarm', 'detector',
    'lamp', 'light', 'bulb', 'torch', 'flashlight',
    # Colors
    'gold', 'silver', 'black', 'white', 'red', 'blue', 'green', 'pink',
    'yellow', 'purple', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy',
    # Common adjectives
    'vintage', 'classic', 'modern', 'cute', 'sexy', 'elegant', 'casual',
    'formal', 'sporty', 'cozy', 'warm', 'cool', 'soft', 'hard',
    'thick', 'thin', 'long', 'short', 'round', 'square',
    'new', 'used', 'original', 'import', 'local',
    # Measurements & specs
    'ml', 'mm', 'cm', 'kg', 'g', 'l', 'pcs', 'pc', 'set', 'pair',
    'inch', 'volt', 'watt', 'amp', 'mah', 'gb', 'tb', 'rpm',
    # Technical terms
    'wifi', 'usb', 'hdmi', 'bluetooth', 'gps', 'fm', 'am',
    'dc', 'ac', 'led', 'lcd', 'oled', 'amoled',
    '3d', '4d', 'hd', 'full hd', '4k', '8k',
    # Health & Beauty specific
    'spf', 'uva', 'uvb', 'bpa free', 'organic', 'natural', 'herbal',
    # Numbers that appear in specs
    'full', 'half', 'quarter',
}


# ============================================================
# SECTION 4: CATEGORY KEYWORDS - CHUẨN HÓA
# ============================================================
CATEGORY_NORM: dict = {
    # Giày dép
    r'\bsandal\b': 'sandal',
    r'\bsendal\b': 'sandal',
    r'\bsepatu\b': 'giày',
    r'\b heels?\b': 'cao gót',
    r'\bboot\b': 'boots',
    r'\bslipper\b': 'dép',
    # Túi xách
    r'\btas\b': 'túi',
    r'\bbag\b': 'túi',
    r'\bbackpack\b': 'túi đeo lưng',
    r'\bsling\b': 'túi chéo',
    r'\bwallet\b': 'ví',
    r'\bdompet\b': 'ví',
    # Quần áo
    r'\bcelana\b': 'quần',
    r'\bpants?\b': 'quần',
    r'\blegging\b': 'quần',
    r'\bleggins\b': 'quần',
    r'\bdress\b': 'váy',
    r'\brobe\b': 'áo choàng',
    r'\bshirt\b': 'áo',
    r'\bt-shirt\b': 'áo thun',
    r'\bblouse\b': 'áo sơ mi nữ',
    r'\bkemeja\b': 'áo sơ mi',
    r'\bjacket\b': 'áo khoác',
    r'\bhoodie\b': 'áo hoodie',
    r'\bsweater\b': 'áo len',
    # Phụ kiện
    r'\bgelang\b': 'vòng tay',
    r'\bgelang\b': 'vòng tay',
    r'\bkalung\b': 'dây chuyền',
    r'\bjam\b': 'đồng hồ',
    r'\bjewelry\b': 'trang sức',
    # Làm đẹp
    r'\blipstick\b': 'son môi',
    r'\blip\b': 'môi',
    r'\bmascara\b': 'mascara',
    r'\bpowder\b': 'phấn',
    r'\bfoundation\b': 'kem nền',
    r'\bserum\b': 'tinh chất',
    r'\bcream\b': 'kem',
    r'\bwipes?\b': 'khăn ướt',
    r'\bsunscreen\b': 'kem chống nắng',
    # Nhà bếp
    r'\bspatula\b': 'xẻng',
    r'\bpan\b': 'chảo',
    r'\bpot\b': 'nồi',
    r'\bknife\b': 'dao',
    r'\bblender\b': 'máy xay',
    r'\bmixer\b': 'máy đánh trứng',
    # Khác
    r'\bmask\b': 'mặt nạ',
    r'\bglasses\b': 'kính',
    r'\bsunglasses\b': 'kính râm',
    r'\bstrap\b': 'dây đeo',
    r'\bcover\b': 'ốp lưng',
    r'\bcase\b': 'hộp',
}


# ============================================================
# SECTION 5: PRODUCT-SPECIFIC ENGLISH MAPPING
# ============================================================
ENGLISH_TO_VIETNAMESE: dict = {
    # Common product terms
    'full cover': 'ốp lưng',
    'tempered glass': 'kính cường lực',
    'air freshener': 'nước hoa không khí',
    'air purifier': 'máy lọc không khí',
    'air conditioner': 'điều hòa',
    'vacuum cleaner': 'máy hút bụi',
    'mosquito': 'muỗi',
    'insect': 'côn trùng',
    'baby': 'trẻ em',
    'children': 'trẻ em',
    'kids': 'trẻ em',
    'facial wash': 'sữa rửa mặt',
    'body lotion': 'sữa dưỡng thể',
    'body scrub': 'tẩy tế bào chết',
    'shampoo': 'dầu gội',
    'conditioner': 'dầu xả',
    'moisturizer': 'kem dưỡng ẩm',
    'perfume': 'nước hoa',
    'cologne': 'nước hoa',
    'watch': 'đồng hồ',
    'clock': 'đồng hồ',
    'wall clock': 'đồng hồ treo tường',
    'necklace': 'dây chuyền',
    'bracelet': 'vòng tay',
    'earring': 'bông tai',
    'ring': 'nhẫn',
    'wallet': 'ví',
    'purse': 'ví',
    'handbag': 'túi xách',
    'shoulder bag': 'túi đeo vai',
    'clutch': 'túi cầm tay',
    'backpack': 'túi đeo lưng',
    'laptop': 'laptop',
    'smartphone': 'điện thoại',
    'tablet': 'máy tính bảng',
    'charger': 'sạc',
    'cable': 'cáp',
    'earphone': 'tai nghe',
    'headphone': 'tai nghe',
    'speaker': 'loa',
    'powerbank': 'sạc dự phòng',
    'usb': 'usb',
    'adapter': 'adapter',
    'brush': 'bàn chải',
    'comb': 'lược',
    'mirror': 'gương',
    ' organizer': 'hộp đựng',
    'container': 'hộp',
    'bottle': 'chai',
    'jar': 'lọ',
    'spray': 'xịt',
    'pump': 'bơm',
    'set': 'bộ',
    'kit': 'bộ',
    'pack': 'gói',
    'box': 'hộp',
    'gold': 'vàng',
    'silver': 'bạc',
    'rose gold': 'hồng vàng',
    'black': 'đen',
    'white': 'trắng',
    'red': 'đỏ',
    'blue': 'xanh dương',
    'green': 'xanh lá',
    'pink': 'hồng',
    'yellow': 'vàng',
    'purple': 'tím',
    'orange': 'cam',
    'brown': 'nâu',
    'gray': 'xám',
    'beige': 'be',
    'navy': 'xanh navy',
    'multicolor': 'nhiều màu',
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def remove_seo_spam(text: str) -> str:
    """Loại bỏ SEO spam keywords."""
    text_lower = text.lower()
    for keyword in SEO_SPAM_KEYWORDS:
        # Replace keyword với boundary để tránh partial match
        pattern = r'\b' + re.escape(keyword) + r'\b'
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def remove_consecutive_repetition(text: str, max_repeat: int = 2) -> str:
    """
    Loại bỏ repetition LIÊN TIẾP, ví dụ:
    - "chân / chân / chân" -> "chân"
    - "1 lít 1 lít 1 lít" -> "1 lít"
    
    NHƯNG GIỮ NGUYÊN từ trùng KHÔNG liên tiếp:
    - "áo thun nam nam tính" -> giữ nguyên (nam không liên tiếp)
    """
    words = text.split()
    if not words:
        return text
    
    result = []
    prev_word = None
    count = 1
    
    for word in words:
        if word == prev_word:
            count += 1
            if count <= max_repeat:
                result.append(word)
        else:
            result.append(word)
            prev_word = word
            count = 1
    
    return ' '.join(result)


def remove_english_stopwords(text: str) -> str:
    """Loại bỏ English stopwords nhưng giữ important brands/products."""
    words = text.split()
    kept_words = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        
        # Luôn giữ important brands
        if word_lower in IMPORTANT_BRANDS:
            kept_words.append(word)
            continue
        
        # Giữ từ có số (model numbers, sizes)
        if re.match(r'^[\d.,]+$', word):
            kept_words.append(word)
            continue
        
        # Giữ các đơn vị đo
        if word_lower in ['ml', 'mm', 'cm', 'kg', 'g', 'l', 'pcs', 'pc', 'set', 'inch', 'volt', 'watt', 'mah', 'gb']:
            kept_words.append(word)
            continue
        
        # Loại bỏ stopwords
        if word_lower not in ENGLISH_STOPWORDS:
            kept_words.append(word)
    
    return ' '.join(kept_words)


def normalize_categories(text: str) -> str:
    """Chuẩn hóa category keywords."""
    for pattern, replacement in CATEGORY_NORM.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_product_codes(text: str) -> str:
    """Loại bỏ mã sản phẩm dạng: ABC-123, LT1535, BQ2481."""
    # Pattern cho mã sản phẩm
    patterns = [
        r'[A-Z]{2,4}[-]?\d{3,6}',  # ABC-123, LT1535
        r'\d{3,6}[-]?\d{3,6}',     # 123-456, 123456
        r'[A-Z]{1,2}\d{2,6}',       # A12, ABC1234
        r'\d+[A-Z]{1,3}\d*',        # 100ML, 50G
        r'BPOM',                    # Regulatory code
        r'KOMPAS', r'CNN', r'OK',   # Media tags
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text


def remove_special_chars(text: str) -> str:
    """Loại bỏ ký tự đặc biệt nhưng giữ dấu câu."""
    # Giữ: chữ cái (có dấu), số, space, dấu câu cơ bản
    text = re.sub(r'[^\w\sÀ-ỹ.,!?;:\-\'\"()]', ' ', text)
    
    # Loại bỏ nhiều slash: A / B / C -> space
    text = re.sub(r'\s*/\s*', ' ', text)
    
    # Loại bỏ nhiều dash
    text = re.sub(r'\s*-\s*', ' ', text)
    
    return text


def remove_too_many_numbers(text: str) -> str:
    """Loại bỏ chuỗi số quá dài (có thể là mã)."""
    # Loại bỏ chuỗi số >= 8 chữ số liên tiếp
    text = re.sub(r'\d{8,}', ' ', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa whitespace."""
    # Nhiều space -> 1 space
    text = re.sub(r'\s+', ' ', text)
    # Trim
    text = text.strip()
    return text


def remove_repeated_patterns(text: str) -> str:
    """
    Loại bỏ các pattern lặp dạng 'ABC ABC ABC ABC' CHỈ KHI quá 70% từ là trùng.
    
    Ví dụ:
    - "vui vẻ bjg 3029 vui vẻ bjg 3029 bjg 3029" -> "vui vẻ bjg 3029"
    - "áo thun nam nam tính" -> giữ nguyên (nam không lặp liên tiếp)
    """
    words = text.split()
    if len(words) < 4:
        return text
    
    # Kiểm tra nếu > 70% từ là trùng lặp -> có vấn đề
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        # Giữ 1 instance của mỗi từ duy nhất (theo thứ tự xuất hiện đầu tiên)
        seen = set()
        result = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen:
                result.append(w)
                seen.add(w_lower)
        return ' '.join(result)
    
    return text


def has_too_much_english(text: str, max_ratio: float = 0.35) -> bool:
    """Kiểm tra xem caption có quá nhiều tiếng Anh không."""
    words = text.split()
    if not words:
        return False
    
    english_count = 0
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        # Check if word is mostly ASCII (likely English)
        if word_clean and len(word_clean) > 2:
            ascii_ratio = sum(1 for c in word_clean if ord(c) < 128) / len(word_clean)
            if ascii_ratio > 0.7:
                english_count += 1
    
    english_ratio = english_count / len(words)
    return english_ratio > max_ratio


def is_valid_caption(text: str, min_len: int = 8) -> Tuple[bool, str]:
    """
    Kiểm tra caption có hợp lệ không.
    Returns: (is_valid, reason)
    """
    if not text or len(text.strip()) == 0:
        return False, "empty"
    
    words = text.split()
    
    if len(words) < min_len:
        return False, f"too_short ({len(words)} words)"
    
    # Kiểm tra quá nhiều số liên tiếp
    if re.search(r'\d[\s,]*\d[\s,]*\d[\s,]*\d', text):
        return False, "too_many_numbers"
    
    # Kiểm tra chỉ toàn số
    if re.match(r'^[\d\s,.]+$', text):
        return False, "only_numbers"
    
    # Kiểm tra gibberish (kí tự lạ)
    letters = re.findall(r'[a-zA-ZÀ-ỹ]', text)
    if len(letters) < 5:
        return False, "gibberish"
    
    return True, "ok"


def clean_caption(text: str) -> str:
    """
    Main cleaning function - apply all cleaning steps.
    
    Order of operations:
    1. Basic normalization
    2. Remove SEO spam
    3. Remove consecutive repetition (CRITICAL: only consecutive!)
    4. Remove product codes
    5. Remove special characters
    6. Normalize categories
    7. Remove English stopwords (carefully)
    8. Remove repeated patterns
    9. Final cleanup
    """
    if not text:
        return ""
    
    # 1. Basic normalization
    text = text.strip()
    text = text.lower()
    
    # 2. Remove SEO spam
    text = remove_seo_spam(text)
    
    # 3. Remove consecutive repetition (KEY FIX: only consecutive!)
    text = remove_consecutive_repetition(text, max_repeat=2)
    
    # 4. Remove product codes
    text = remove_product_codes(text)
    
    # 5. Remove special characters
    text = remove_special_chars(text)
    
    # 6. Normalize categories
    text = normalize_categories(text)
    
    # 7. Remove English stopwords (carefully - don't remove too aggressively)
    text = remove_english_stopwords(text)
    
    # 8. Remove repeated patterns
    text = remove_repeated_patterns(text)
    
    # 9. Remove too many numbers
    text = remove_too_many_numbers(text)
    
    # 10. Final cleanup
    text = normalize_whitespace(text)
    
    return text


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_csv(
    input_path: str,
    output_path: str,
    min_len: int = 8,
    max_en_ratio: float = 0.35,
    dry_run: bool = False,
    sample_size: int = 100
) -> dict:
    """
    Process CSV file with cleaning.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        min_len: Minimum word count for caption
        max_en_ratio: Maximum English ratio allowed
        dry_run: If True, don't write output (just stats)
        sample_size: Number of samples to show in dry run
    
    Returns:
        Statistics dict
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    stats = {
        'total': 0,
        'cleaned': 0,
        'removed_too_short': 0,
        'removed_too_english': 0,
        'removed_gibberish': 0,
        'removed_other': 0,
    }
    
    samples_removed = []
    samples_cleaned = []
    
    print(f"📖 Đọc file: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    stats['total'] = len(rows)
    output_rows = []
    
    for row in rows:
        original_caption = row.get('caption_vi_cleaned', row.get('caption_vi', ''))
        
        if not original_caption:
            stats['removed_other'] += 1
            continue
        
        # Apply cleaning
        cleaned_caption = clean_caption(original_caption)
        
        # Validate
        is_valid, reason = is_valid_caption(cleaned_caption, min_len=min_len)
        
        if not is_valid:
            if reason == "too_short":
                stats['removed_too_short'] += 1
            elif reason == "too_many_numbers" or reason == "only_numbers":
                stats['removed_gibberish'] += 1
            else:
                stats['removed_other'] += 1
            
            if len(samples_removed) < sample_size:
                samples_removed.append({
                    'original': original_caption,
                    'cleaned': cleaned_caption,
                    'reason': reason,
                    'image': row.get('image', 'unknown')
                })
            continue
        
        # Check English ratio
        if has_too_much_english(cleaned_caption, max_ratio=max_en_ratio):
            stats['removed_too_english'] += 1
            if len(samples_removed) < sample_size:
                samples_removed.append({
                    'original': original_caption,
                    'cleaned': cleaned_caption,
                    'reason': 'too_english',
                    'image': row.get('image', 'unknown')
                })
            continue
        
        # Caption is valid
        row['caption_vi_cleaned_v2'] = cleaned_caption
        output_rows.append(row)
        stats['cleaned'] += 1
        
        if len(samples_cleaned) < sample_size:
            samples_cleaned.append({
                'original': original_caption,
                'cleaned': cleaned_caption,
                'image': row.get('image', 'unknown')
            })
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 THỐNG KÊ CLEANING")
    print("="*60)
    print(f"Tổng samples:           {stats['total']:,}")
    print(f"✅ Được giữ lại:        {stats['cleaned']:,} ({100*stats['cleaned']/stats['total']:.1f}%)")
    print(f"❌ Bị loại (tổng):      {stats['total'] - stats['cleaned']:,} ({100*(stats['total']-stats['cleaned'])/stats['total']:.1f}%)")
    print(f"   - Quá ngắn:          {stats['removed_too_short']:,}")
    print(f"   - Quá nhiều Eng:     {stats['removed_too_english']:,}")
    print(f"   - Gibberish/số:      {stats['removed_gibberish']:,}")
    print(f"   - Lý do khác:        {stats['removed_other']:,}")
    print("="*60)
    
    # Print sample comparisons
    print("\n📝 MẪU ĐƯỢC CLEAN (đầu tiên):")
    print("-"*60)
    for i, s in enumerate(samples_cleaned[:5], 1):
        print(f"[{i}] {s['image']}")
        print(f"    Trước: {s['original'][:80]}...")
        print(f"    Sau:   {s['cleaned'][:80]}...")
        print()
    
    if samples_removed:
        print("\n📝 MẪU BỊ LOẠI (đầu tiên):")
        print("-"*60)
        for i, s in enumerate(samples_removed[:5], 1):
            print(f"[{i}] {s['image']} - Lý do: {s['reason']}")
            print(f"    Trước: {s['original'][:80]}...")
            print(f"    Sau:   {s['cleaned'][:80] if s['cleaned'] else '(trống)'}...")
            print()
    
    # Write output if not dry run
    if not dry_run:
        print(f"\n💾 Ghi file: {output_path}")
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = list(rows[0].keys()) + ['caption_vi_cleaned_v2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"✅ Đã ghi {len(output_rows):,} rows")
    else:
        print(f"\n⚠️  Dry-run mode: Không ghi file")
    
    return stats


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Data Cleaning Script v2 cho Vietnamese Product Caption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python tools/clean_data_v2.py --input data/train_80_cleaned.csv --output data/train_80_cleaned_v2.csv
  
  # Test nhanh 100 samples
  python tools/clean_data_v2.py --input data/train_80_cleaned.csv --output /tmp/test.csv --dry-run --sample-size 100
  
  # Custom thresholds
  python tools/clean_data_v2.py --input data/train_80_cleaned.csv --output data/v2.csv --min-len 10 --max-en-ratio 0.3
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Đường dẫn file CSV đầu vào')
    parser.add_argument('--output', '-o', required=True, help='Đường dẫn file CSV đầu ra')
    parser.add_argument('--min-len', type=int, default=8, help='Số từ tối thiểu (default: 8)')
    parser.add_argument('--max-en-ratio', type=float, default=0.35, help='Tỷ lệ English tối đa (default: 0.35)')
    parser.add_argument('--dry-run', action='store_true', help='Chỉ hiển thị stats, không ghi file')
    parser.add_argument('--sample-size', type=int, default=100, help='Số mẫu hiển thị (default: 100)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧹 DATA CLEANING SCRIPT v2")
    print("="*60)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Min len: {args.min_len} words")
    print(f"Max EN:  {args.max_en_ratio*100:.0f}%")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    process_csv(
        input_path=args.input,
        output_path=args.output,
        min_len=args.min_len,
        max_en_ratio=args.max_en_ratio,
        dry_run=args.dry_run,
        sample_size=args.sample_size
    )


if __name__ == '__main__':
    main()
