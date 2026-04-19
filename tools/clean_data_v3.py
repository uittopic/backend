"""
Data Cleaning Script v3 — CẢI TIẾN
===================================

Vấn đề với v2: CLEANING QUÁ MỨC → mất thông tin có giá trị
  - "áo thun nam cổ tròn màu trắng size M" → "áo thun nam" (mất size, màu)
  - "áo sơ mi nữ họa tiết hoa" → "áo" (mất họa tiết)

Giải pháp v3: CLEANING THÔNG MINH
  1. GIỮ NGUYÊN thông tin sản phẩm có giá trị (size, màu, chất liệu, hoa văn)
  2. Chỉ LOẠI BỎ SEO spam rõ ràng (sale, hot, free, ...)
  3. KHÔNG xóa English product-specific words (shirt, dress, color names)
  4. Giữ format tự nhiên: "áo thun nam cổ tròn màu trắng size M" → giữ nguyên

Nguyên tắc: "Cleaning = Loại bỏ nhiễu, KHÔNG phải loại bỏ thông tin"

Usage:
  python tools/clean_data_v3.py --input data/train_80_cleaned.csv --output data/train_80_cleaned_v3.csv
"""

import re
import csv
import argparse
import unicodedata
from pathlib import Path
from typing import List, Tuple, Set


# ============================================================
# SECTION 1: SEO SPAM KEYWORDS — CẦN LOẠI BỎ
# ============================================================
SEO_SPAM_KEYWORDS: Set[str] = {
    # Tiếng Anh — chỉ loại bỏ marketing language rõ ràng
    'free', 'sale', 'hot ', 'best', 'new ', 'newest', 'latest',
    'original', 'authentic', 'genuine', 'premium quality',
    'best seller', 'bestseller', 'top ', 'popular',
    'ready', 'available', 'in stock', 'limited', 'promo', 'promotion',
    'discount', 'cheap', 'affordable', 'murah', 'gratis',
    'ongkir', 'free ongkir', 'free-ship', 'freeship',
    ' COD ', 'cod', 'bayar', 'transfer',
    'terlaris', 'terbaik', 'terpercaya', 'terjamin',
    'official store', 'official shop', 'boutique',
    # Tiếng Việt — marketing rõ ràng
    'giá rẻ', 'miễn phí', 'khuyến mãi', 'giảm giá', 'hotsale',
    'bán chạy', 'nổi bật', 'yêu thích', 'hàng mới', 'về gấp',
    'đáng mua', 'siêu rẻ', 'cực rẻ', 'mới nhất',
    # Marketing quá mức
    'super', 'mega', 'ultra', 'plus', 'pro', 'max',
    'big sale', 'flash sale', 'clearance', 'outlet',
}


# ============================================================
# SECTION 2: WORDS GIỮ NGUYÊN — KHÔNG XÓA
# ============================================================
# v3: GIỮ LẠI tất cả product-specific words
# Đây là thông tin quan trọng cho BLEU score
KEEP_WORDS: Set[str] = {
    # === TIẾNG VIỆT CƠ BẢN ===
    # Loại quần áo
    'áo thun', 'áo sơ mi', 'áo khoác', 'áo len', 'áo hoodie',
    'áo polo', 'áo phông', 'áo vest', 'áo choàng', 'áo câu lạc bộ',
    'áo dài', 'áo gió', 'áo ba lỗ', 'áo kiểu', 'áo croptop',
    'váy', 'váy đầm', 'váy xòe', 'váy ôm', 'váy suông',
    'quần jeans', 'quần short', 'quần tây', 'quần baggy',
    'quần ống rộng', 'quần ống loe', 'quần legging', 'quần jogger',
    'chân váy', 'đầm', 'set', 'bộ', 'combo',
    # Giày dép
    'giày', 'sandal', 'dép', 'cao gót', 'boots', 'sneakers',
    ' giày ', 'sandals', 'dép',
    # Túi xách
    'túi xách', 'túi đeo chéo', 'túi đeo vai', 'túi đeo lưng',
    'túi tote', 'ví', 'clutch', 'balo', 'túi mini',
    # Phụ kiện
    'đồng hồ', 'dây chuyền', 'vòng tay', 'bông tai', 'nhẫn',
    'kính', 'kính râm', 'mũ', 'khăn', 'nơ',
    # Màu sắc (quan trọng!)
    'màu trắng', 'màu đen', 'màu đỏ', 'màu xanh', 'màu hồng',
    'màu vàng', 'màu tím', 'màu cam', 'màu nâu', 'màu xám',
    'trắng', 'đen', 'đỏ', 'xanh dương', 'xanh lá', 'hồng',
    'vàng', 'tím', 'cam', 'nâu', 'xám', 'be', 'navy',
    'hồng gold', 'hồng nhạt', 'hồng đậm',
    # Kích thước (quan trọng!)
    'size s', 'size m', 'size l', 'size xl', 'size xxl',
    'size ', 'cỡ ', 'cỡ s', 'cỡ m', 'cỡ l',
    # Chất liệu
    'chất liệu', 'vải', 'cotton', 'nỉ', 'len', 'lụa',
    'da', 'nhựa', 'kim loại', 'gỗ',
    # Hoa văn (quan trọng!)
    'họa tiết', 'hoa', 'lá', 'cây', 'animal', 'striped',
    'pattern', 'in hoa', 'in lá', 'trơn', 'đơn sắc',
    ' Caro ', 'khải', 'cam', 'sọc', 'hoa văn',
    # Kiểu dáng
    'cổ tròn', 'cổ vuông', 'cổ tim', 'cổ polo', 'cổ bẻ',
    'tay dài', 'tay ngắn', 'tay lỡ', 'ôm', 'rộng', 'xòe',
    'ngắn', 'dài', 'suông', 'bó',
    # === ENGLISH PRODUCT-SPECIFIC ===
    # Colors
    'gold', 'silver', 'rose gold', 'black', 'white', 'red', 'blue',
    'green', 'pink', 'yellow', 'purple', 'orange', 'brown', 'gray',
    'beige', 'navy', 'multicolor', 'color',
    # Sizes
    'small', 'medium', 'large', 'xl', 'xxl', 'free size', 'one size',
    # Garment types
    'shirt', 'blouse', 'dress', 'skirt', 'pants', 'shorts', 'jeans',
    'jacket', 'hoodie', 'sweater', 'cardigan', 'vest', 'coat',
    'sandal', 'slipper', 'heels', 'boots', 'sneakers', 'flat',
    'bag', 'wallet', 'backpack', 'clutch', 'tote', 'purse',
    'watch', 'necklace', 'bracelet', 'earring', 'ring',
    'lipstick', 'mascara', 'foundation', 'powder', 'blush',
    'serum', 'cream', 'lotion', 'toner', 'mask', 'scrub',
    'shampoo', 'conditioner', 'brush', 'comb', 'mirror',
    'bottle', 'jar', 'spray', 'set', 'combo', 'pack', 'mini',
    # Tech specs
    'led', 'lcd', 'oled', 'touch', 'screen', 'display',
    'waterproof', 'water resistant',
    'wireless', 'bluetooth', 'usb', 'hdmi',
    'smart', 'digital', 'electric',
    'ml', 'mm', 'cm', 'kg', 'gram', 'g', 'l', 'pcs', 'set',
    'inch', 'volt', 'watt', 'mah', 'gb', 'tb',
    'wifi', 'gps', 'fm', '3d', '4d', 'hd', 'full hd', '4k',
    'baby', 'kids', 'children', 'men', 'women', 'unisex',
    'vintage', 'classic', 'modern', 'cute', 'elegant', 'casual',
    'formal', 'sporty', 'sexy',
}


# ============================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================

def remove_seo_spam(text: str) -> str:
    """Loại bỏ SEO spam keywords — CẨN THẬN, không xóa quá nhiều"""
    text_lower = text.lower()
    for keyword in SEO_SPAM_KEYWORDS:
        pattern = r'\b' + re.escape(keyword.strip()) + r'\b'
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def remove_consecutive_repetition(text: str, max_repeat: int = 3) -> str:
    """
    Loại bỏ repetition LIÊN TIẾP (>=3 lần).
    Giữ nguyên từ trùng KHÔNG liên tiếp.
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


def remove_product_codes(text: str) -> str:
    """Loại bỏ mã sản phẩm dạng: ABC-123, LT1535"""
    patterns = [
        r'[A-Z]{2,4}[-]?\d{3,6}',
        r'\d{3,6}[-]?\d{3,6}',
        r'[A-Z]{1,2}\d{2,6}',
        r'\d+[A-Z]{1,3}\d*',
        r'BPOM', r'KOMPAS', r'CNN', r'OK',
    ]
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def has_vietnamese(text: str) -> bool:
    """Kiểm tra text có tiếng Việt không"""
    VI_CHARS = 'àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ'
    return any(c in VI_CHARS for c in text)


def is_valid_caption(text: str, min_len: int = 6) -> Tuple[bool, str]:
    """
    Kiểm tra caption có hợp lệ không.
    v3: Giảm min_len từ 8 → 6 (vì caption clean hơn rồi)
    """
    if not text or len(text.strip()) == 0:
        return False, "empty"

    words = text.split()

    if len(words) < min_len:
        return False, f"too_short ({len(words)} words)"

    # Kiểm tra chuỗi số quá dài
    if re.search(r'\d{8,}', text):
        return False, "too_many_numbers"

    # Kiểm tra chỉ toàn số
    if re.match(r'^[\d\s,.]+$', text):
        return False, "only_numbers"

    # Kiểm tra gibberish (kí tự lạ)
    letters = re.findall(r'[a-zA-ZÀ-ỹ]', text)
    if len(letters) < 3:
        return False, "gibberish"

    return True, "ok"


# ============================================================
# SECTION 4: MAIN CLEANING FUNCTION
# ============================================================

def clean_caption_v3(text: str) -> str:
    """
    v3: CLEANING THÔNG MINH
    - GIỮ NGUYÊN thông tin sản phẩm có giá trị
    - Chỉ loại bỏ SEO spam rõ ràng
    - KHÔNG xóa English product words
    """
    if not text:
        return ""

    # 1. Basic normalization
    text = text.strip()
    text = text.lower()

    # 2. Remove SEO spam (CẨN THẬN)
    text = remove_seo_spam(text)

    # 3. Remove consecutive repetition (>=3 lần)
    text = remove_consecutive_repetition(text, max_repeat=3)

    # 4. Remove product codes
    text = remove_product_codes(text)

    # 5. Remove ký tự đặc biệt — GIỮ dấu câu cơ bản
    text = re.sub(r'[^\w\sÀ-ỹ.,!?;:\-\'\"()]', ' ', text)

    # 6. Remove nhiều slash/dash
    text = re.sub(r'\s*/\s*', ' ', text)
    text = re.sub(r'\s*-\s*', ' ', text)

    # 7. Final cleanup
    text = normalize_whitespace(text)

    return text


# ============================================================
# SECTION 5: MAIN PROCESSING
# ============================================================

def process_csv(
    input_path: str,
    output_path: str,
    min_len: int = 6,
    dry_run: bool = False,
    sample_size: int = 100
) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = {
        'total': 0, 'cleaned': 0,
        'removed_too_short': 0, 'removed_gibberish': 0, 'removed_other': 0,
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

        # Apply v3 cleaning
        cleaned_caption = clean_caption_v3(original_caption)

        # Validate
        is_valid, reason = is_valid_caption(cleaned_caption, min_len=min_len)

        if not is_valid:
            if reason == "too_short":
                stats['removed_too_short'] += 1
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

        # Caption is valid — giữ lại
        row['caption_vi_cleaned_v3'] = cleaned_caption
        output_rows.append(row)
        stats['cleaned'] += 1

        if len(samples_cleaned) < sample_size:
            samples_cleaned.append({
                'original': original_caption,
                'cleaned': cleaned_caption,
                'image': row.get('image', 'unknown')
            })

    # Print statistics
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ CLEANING v3 (THÔNG MINH)")
    print("=" * 60)
    print(f"Tổng samples:           {stats['total']:,}")
    print(f"✅ Được giữ lại:        {stats['cleaned']:,} ({100*stats['cleaned']/stats['total']:.1f}%)")
    print(f"❌ Bị loại (tổng):      {stats['total'] - stats['cleaned']:,}")
    print(f"   - Quá ngắn:          {stats['removed_too_short']:,}")
    print(f"   - Lý do khác:        {stats['removed_other']:,}")
    print("=" * 60)

    # Print sample comparisons
    print("\n📝 MẪU ĐƯỢC CLEAN (đầu tiên):")
    print("-" * 60)
    for i, s in enumerate(samples_cleaned[:5], 1):
        print(f"[{i}] {s['image']}")
        print(f"    Trước: {s['original'][:80]}")
        print(f"    Sau:   {s['cleaned'][:80]}")
        print()

    if samples_removed:
        print("\n📝 MẪU BỊ LOẠI (đầu tiên):")
        print("-" * 60)
        for i, s in enumerate(samples_removed[:5], 1):
            print(f"[{i}] {s['image']} - Lý do: {s['reason']}")
            print(f"    Trước: {s['original'][:80]}")
            print()

    # Write output
    if not dry_run:
        print(f"\n💾 Ghi file: {output_path}")
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = list(rows[0].keys()) + ['caption_vi_cleaned_v3']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"✅ Đã ghi {len(output_rows):,} rows")
    else:
        print(f"\n⚠️  Dry-run: Không ghi file")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Data Cleaning Script v3 — CLEANING THÔNG MINH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python tools/clean_data_v3.py --input data/train_80_cleaned.csv --output data/train_80_cleaned_v3.csv
  python tools/clean_data_v3.py --input data/train_80_cleaned.csv --output /tmp/test.csv --dry-run
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Đường dẫn file CSV đầu vào')
    parser.add_argument('--output', '-o', required=True, help='Đường dẫn file CSV đầu ra')
    parser.add_argument('--min-len', type=int, default=6, help='Số từ tối thiểu (default: 6)')
    parser.add_argument('--dry-run', action='store_true', help='Chỉ hiển thị stats, không ghi file')
    parser.add_argument('--sample-size', type=int, default=100, help='Số mẫu hiển thị')

    args = parser.parse_args()

    print("=" * 60)
    print("🧹 DATA CLEANING v3 — THÔNG MINH")
    print("=" * 60)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Min len: {args.min_len} words")
    print("=" * 60)

    process_csv(
        input_path=args.input,
        output_path=args.output,
        min_len=args.min_len,
        dry_run=args.dry_run,
        sample_size=args.sample_size
    )


if __name__ == '__main__':
    main()
