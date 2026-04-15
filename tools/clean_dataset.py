"""
Clean Dataset - Loại bỏ SEO words và template phrases
Để giảm template bias cho model training
"""
import csv
import re
from pathlib import Path

# SEO words và template phrases cần loại bỏ
SEO_WORDS = [
    # Giá cả
    "giá rẻ", "gía rẻ", "giá bèo", "giá tốt", "giá sinh viên",
    "sale", "sạl", "sl", "off", "ưu đãi", "khuyến mãi", "km",
    "hàng giảm giá", "giảm giá mạnh", "giá sốc", "siêu sale",
    
    # Chất lượng
    "chính hãng", "chính hãng 100%", "hàng chính hãng",
    "authentic", "auth", "genuine", "original",
    "cao cấp", "cấp cao", "sang trọng", "premium",
    "hàng đẹp", "đẹp nhất", "xịn nhất", "hàng xịn",
    
    # Bán chạy
    "bán chạy", "hot", "trend", "best seller", "bs",
    "top", "yêu thích", "tik tok", "tiktok",
    "phổ biến", "phổ biến nhất", "nhiều người mua",
    
    # Đối tượng
    "phụ nữ", "nam giới", "nam nữ", "unisex",
    "trẻ em", "em bé", "baby", "kids",
    "người lớn", "người cao tuổi", "senior",
    
    # Địa điểm/xuất xứ
    "hàng nhập khẩu", "nhập khẩu", "made in", "xuất xứ",
    "hàn quốc", "korea", "trung quốc", "nhật bản",
    "hàng việt nam", "việt nam", "vn",
    
    # Thời trang
    "thời trang", "fashion", "style", "phong cách",
    "hàng mới", "new", "mới nhất", "2024", "2023", "2022",
    
    # Khuyến khích mua
    "mua ngay", "order now", "đặt hàng", "hết hàng",
    "số lượng có hạn", "limited", "quà tặng", "tặng kèm",
    
    # Khác
    "ảnh thật", "hình thật", "review", "đánh giá",
    "ship toàn quốc", "giao hàng", " COD ",
]

# Regex pattern cho các template phổ biến
TEMPLATE_PATTERNS = [
    r'\bban hang\b', r'\bwebgame\b', r'\bgame\b',
    r'\bchoi\b', r'\bdung cu\b', r'\bmeo\b',
    r'\btoan quoc\b', r'\bha noi\b', r'\bsai gon\b',
    r'\bnuoc hoa\b', r'\bmat na\b', r'\bson duong\b',
]

# Regex pattern loại bỏ các cụm như "[ Exp ... ]" hoặc "SKU:xxx"
SPECIAL_PATTERNS = [
    r'\[.*?\]',  # [ Exp tháng 7 năm 2021 ]
    r'SKU:.*?(?=\s|$)',  # SKU:MASKER99
    r'EX.*?(?=\s|$)',  # EX tháng...
]


def clean_caption(text: str) -> str:
    """Loại bỏ SEO words và template phrases từ caption"""
    if not text:
        return text
    
    text_lower = text.lower()
    
    # 1. Loại bỏ các pattern đặc biệt (SKU, Exp...)
    for pattern in SPECIAL_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 2. Loại bỏ SEO words
    for word in SEO_WORDS:
        text = re.sub(re.escape(word), '', text, flags=re.IGNORECASE)
    
    # 3. Loại bỏ template patterns
    for pattern in TEMPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 4. Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Loại bỏ dấu câu thừa
    text = re.sub(r'[-/]+$', '', text)  # Loại bỏ dấu - / cuối câu
    text = re.sub(r'^[-/]+', '', text)  # Loại bỏ dấu - / đầu câu
    
    return text.strip()


def clean_csv(input_path: Path, output_path: Path, caption_col: str = "caption_vi"):
    """Clean một file CSV"""
    count_before = 0
    count_after = 0
    count_removed = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
    
    cleaned_rows = []
    for row in rows:
        count_before += 1
        original_caption = row.get(caption_col, "")
        cleaned_caption = clean_caption(original_caption)
        row["caption_vi_cleaned"] = cleaned_caption
        
        # Đếm số từ bị loại bỏ
        words_removed = len(original_caption.split()) - len(cleaned_caption.split()) if cleaned_caption else 0
        count_removed += words_removed
        
        if cleaned_caption:
            count_after += 1
            cleaned_rows.append(row)
    
    # Ghi file đã clean
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"✅ Đã clean dataset:")
    print(f"   - Trước: {count_before} dòng")
    print(f"   - Sau: {count_after} dòng (loại {count_before - count_after} dòng trống)")
    print(f"   - Từ bị loại: ~{count_removed} từ")
    print(f"   - Output: {output_path}")
    
    return cleaned_rows


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean dataset - loại bỏ SEO words")
    parser.add_argument("--input", type=str, default="data/train_80.csv",
                        help="Input CSV file")
    parser.add_argument("--output", type=str, default="data/train_80_cleaned.csv",
                        help="Output CSV file đã clean")
    parser.add_argument("--caption-col", type=str, default="caption_vi",
                        help="Tên column chứa caption cần clean")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ File không tồn tại: {input_path}")
        exit(1)
    
    print(f"🔄 Đang clean: {input_path}")
    clean_csv(input_path, output_path, args.caption_col)