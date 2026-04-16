#!/usr/bin/env python3
"""
Post-processing cho Vietnamese Captions - Nâng cấp

Áp dụng sau khi model generate để cải thiện chất lượng:

1. Remove repeated words liên tiếp: "máy làm sạch máy làm sạch" → "máy làm sạch"
2. Remove leftover English words
3. Capitalize first letter
4. Fix spacing (không có space trước dấu)
5. Remove trailing/leading spaces
"""

import re
import json
from pathlib import Path
from typing import List, Set

BASE = Path(__file__).parent.parent
BATCH_TEST_JSON = BASE / "outputs" / "batch_test" / "batch_test_results_20260415_125950.json"
OUTPUT_JSON = BASE / "outputs" / "batch_test_results_20260415_125950_postprocessed.json"

# ENGLISH WORDS cần loại bỏ nếu xuất hiện một mình
COMMON_ENGLISH_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'so',
    'free', 'sale', 'hot', 'best', 'new', 'original',
    'import', 'authentic', 'premium', 'official', 'store',
    'ready', 'available', 'stock', 'limited', 'promo',
    'discount', 'cheap', 'affordable', 'murah', 'gratis',
    'ongkir', 'freeship', 'cod', 'transfer',
    'terlaris', 'terbaik', 'terpercaya', 'terjamin',
    'mega', 'ultra', 'plus', 'pro', 'max', 'super',
    'flash', 'clearance', 'outlet', 'big', 'mega',
    # Product type words (often in English in Vietnamese listings)
    'set', 'combo', 'pack', 'bundle',
    'shirt', 'dress', 'pants', 'jeans', 'jacket',
    'hoodie', 'sweater', 'blouse', 'skirt',
    'sandal', 'boots', 'sneakers', 'heels',
    'bag', 'backpack', 'wallet', 'purse', 'clutch',
    'watch', 'necklace', 'bracelet', 'earring', 'ring',
    'lipstick', 'mascara', 'foundation', 'powder',
    'serum', 'cream', 'lotion', 'toner', 'cleanser',
    'mask', 'scrub', 'shampoo', 'conditioner',
    'brush', 'comb', 'mirror', 'organizer',
    'bottle', 'jar', 'spray', 'pump',
    'glasses', 'sunglasses', 'lens',
    'strap', 'cover', 'case', 'holder',
    'cable', 'charger', 'adapter', 'usb', 'hdmi',
    'earphone', 'headphone', 'speaker', 'microphone',
    'powerbank', 'battery',
    'blender', 'mixer', 'grinder', 'chopper', 'slicer',
    'pan', 'pot', 'kettle', 'knife', 'scissors',
    'mat', 'blanket', 'pillow', 'cushion',
    'toy', 'game', 'puzzle',
    'book', 'notebook', 'diary', 'planner',
    'tape', 'glue',
    'door', 'window', 'lock', 'hinge',
    'filter', 'cartridge', 'refill',
    'sensor', 'alarm', 'detector',
    'lamp', 'light', 'bulb', 'torch', 'flashlight',
    'gold', 'silver', 'black', 'white', 'red', 'blue',
    'green', 'pink', 'yellow', 'purple', 'orange', 'brown',
    'gray', 'grey', 'beige', 'navy',
    # Units that should be removed if standalone
    'ml', 'mm', 'cm', 'kg', 'g', 'l', 'pcs', 'pc', 'set',
    'inch', 'volt', 'watt', 'amp', 'mah', 'gb', 'tb',
}

# Words to keep (brand names, product-specific, measurement units)
KEEP_WORDS = {
    'joyko', 'casio', 'samsung', 'xiaomi', 'redmi', 'oppo', 'vivo',
    'huawei', 'apple', 'iphone', 'ipad', 'macbook', 'sony', 'lg',
    'philips', 'panasonic', 'sharp', 'toshiba', 'hitachi', 'crown',
    'nivea', 'vaseline', 'ponds', 'maybelline', 'mac', 'clinique',
    'loreal', 'innisfree', 'body', 'shop',
    'set', 'combo', 'pack', 'mini', 'pro', 'max', 'plus', 'slim',
    'wide', 'auto', 'manual', 'electric', 'digital', 'smart',
    'wireless', 'bluetooth', 'led', 'lcd', 'oled', 'touch', 'screen',
    'display', 'waterproof', 'water', 'resistant', 'splash',
    'baby', 'kids', 'children', 'men', 'women', 'unisex',
    'size', 'small', 'medium', 'large', 'xl', 'xxl', 'free',
    'one', 'full', 'half', 'quarter',
    'ml', 'mm', 'cm', 'kg', 'gram', 'g', 'liter', 'l', 'pcs',
    'pc', 'set', 'inch', 'volt', 'watt', 'amp', 'mah', 'gb', 'tb',
    'rpm',
    'wifi', 'usb', 'hdmi', 'bluetooth', 'gps', 'fm', 'am',
    'dc', 'ac', 'led', 'lcd', 'oled', 'amoled',
    '3d', '4d', 'hd', 'full', '4k', '8k',
    'spf', 'uva', 'uvb', 'bpa', 'organic', 'natural', 'herbal',
    'vintage', 'classic', 'modern', 'cute', 'sexy', 'elegant',
    'casual', 'formal', 'sporty', 'cozy', 'warm', 'cool',
    'soft', 'hard', 'thick', 'thin', 'long', 'short', 'round',
    'square', 'new', 'used', 'original', 'import', 'local',
}


def remove_consecutive_repetition(text: str, max_repeat: int = 1) -> str:
    """Loại bỏ từ lặp liên tiếp."""
    words = text.split()
    if not words:
        return text

    result = [words[0]]
    prev_word = words[0].lower()
    count = 1

    for word in words[1:]:
        word_lower = word.lower()
        if word_lower == prev_word:
            count += 1
            if count <= max_repeat:
                result.append(word)
        else:
            result.append(word)
            prev_word = word_lower
            count = 1

    return ' '.join(result)


def remove_leftover_english(text: str) -> str:
    """Loại bỏ từ tiếng Anh còn sót lại."""
    words = text.split()
    result = []

    for word in words:
        word_lower = word.lower().strip('.,!?;:')

        # Giữ nếu là brand/product name
        if word_lower in KEEP_WORDS:
            result.append(word)
            continue

        # Giữ nếu có dấu tiếng Việt
        if bool(re.search(r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị'
                          r'òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', word)):
            result.append(word)
            continue

        # Loại bỏ từ thuần Anh (có ít nhất 3 ký tự)
        if len(word) >= 3 and word_lower not in KEEP_WORDS:
            # Check if word is mostly ASCII (English)
            ascii_count = sum(1 for c in word if ord(c) < 128)
            if ascii_count / len(word) > 0.8:
                # Skip (remove English word)
                continue

        result.append(word)

    return ' '.join(result)


def normalize_caption(text: str) -> str:
    """
    Chuẩn hóa caption:
    - Viết hoa chữ cái đầu
    - Fix spacing: không có space trước dấu câu
    - Loại bỏ space thừa
    """
    if not text:
        return ""

    # 1. Remove consecutive repetition (chỉ 1 lần)
    text = remove_consecutive_repetition(text, max_repeat=1)

    # 2. Remove leftover English words
    text = remove_leftover_english(text)

    # 3. Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # "hello , world" → "hello, world"
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)  # Ensure space after punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces

    # 4. Strip
    text = text.strip()

    # 5. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # 6. Remove trailing punctuation space
    text = re.sub(r'\s+([.,!?;:])$', r'\1', text)

    return text


def process_batch_test():
    """Apply post-processing cho toàn bộ batch test results."""
    print(f"📖 Đang đọc {BATCH_TEST_JSON}...")

    with open(BATCH_TEST_JSON) as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)
    improved = 0

    print(f"📊 Tổng {total} samples, đang post-process...")

    for i, item in enumerate(results):
        if not item.get("success"):
            continue

        original_caption = item["caption_vi"]

        # Apply post-processing
        new_caption = normalize_caption(original_caption)

        # Track improvements
        if new_caption != original_caption:
            improved += 1

        item["caption_vi"] = new_caption
        item["caption_vi_postprocessed"] = new_caption

    # Save
    print(f"\n💾 Đang ghi {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Hoàn thành!")
    print(f"   Tổng samples: {total}")
    print(f"   Đã cải thiện: {improved} ({improved*100/total:.1f}%)")
    print(f"   Không đổi: {total - improved} ({(total-improved)*100/total:.1f}%)")


if __name__ == '__main__':
    process_batch_test()
