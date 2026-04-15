"""
20 Ảnh Test - Đánh giá Model BLIP Tiếng Việt
Evaluation Framework
"""
import csv
import time
from pathlib import Path

# Kết quả test sẽ lưu vào đây
RESULTS_FILE = Path("outputs/eval_20_images.csv")

EVALUATION_TEMPLATE = """
╔══════════════════════════════════════════════════════════════════╗
║                    20 ẢNH TEST EVALUATION                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Mỗi ảnh ghi:                                                   ║
║  1. Caption output     2. Ngôn ngữ: VI / EN / Mixed / Error     ║
║  3. Đúng sản phẩm?     4. Có rác/ lặp từ?                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

def analyze_caption(caption: str) -> dict:
    """Phân tích caption output"""
    result = {
        "caption": caption,
        "language": "Unknown",
        "is_vietnamese": False,
        "is_english": False,
        "has_error": False,
        "has_garbage": False,
    }

    # Check for garbled/broken text (Vietnamese accent restoration failure pattern)
    if "chân" in caption or "chan" in caption:
        # Check if it's garbled (repeated characters)
        words = caption.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                result["has_garbage"] = True
                result["has_error"] = True
                result["language"] = "Error"
                return result

    # Check language
    vietnamese_markers = ["áo", "quần", "túi", "giày", "ví", "balo", "đồng hồ", "son", "kem",
                          "nước hoa", "mỹ phẩm", "phụ kiện", "da", "vải", "màu", "trắng", "đen",
                          "xanh", "hồng", "nữ", "nam", "unisex", "cho", "của", "với", "có", "là"]
    english_markers = ["a ", "the", "photo", "holding", "wearing", "with", "and", "on", "is",
                      "shoes", "watch", "bag", "wallet", "product", "style", "new"]

    vi_count = sum(1 for m in vietnamese_markers if m.lower() in caption.lower())
    en_count = sum(1 for m in english_markers if m.lower() in caption.lower())

    # Vietnamese diacritics check
    vietnamese_chars = "ăâáàảạãằắẳẵặẩấầẫậộợôốồờớỡởợơửứừựửỵếềừư"
    has_vietnamese_chars = any(c in caption.lower() for c in vietnamese_chars)

    if has_vietnamese_chars:
        result["is_vietnamese"] = True
        result["language"] = "VI"
    elif en_count > vi_count:
        result["is_english"] = True
        result["language"] = "EN"
    else:
        result["language"] = "Mixed"

    return result

def print_results(results: list):
    """In kết quả đánh giá"""
    total = len(results)
    correct = sum(1 for r in results if r["correct_product"])
    vietnamese = sum(1 for r in results if r["language"] == "VI")
    english = sum(1 for r in results if r["language"] == "EN")
    errors = sum(1 for r in results if r["has_error"])

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                         KẾT QUẢ TỔNG HỢP                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Tổng ảnh test:       {total:>3}                                      ║
║  Đúng sản phẩm:      {correct:>3} ({correct*100/total:>5.1f}%)                         ║
║  Tiếng Việt:         {vietnamese:>3} ({vietnamese*100/total:>5.1f}%)                         ║
║  Tiếng Anh:           {english:>3} ({english*100/total:>5.1f}%)                         ║
║  Lỗi/ rác:            {errors:>3} ({errors*100/total:>5.1f}%)                         ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Phân loại model
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                         KẾT LUẬN                                 ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    if correct/total >= 0.7 and vietnamese/total >= 0.5 and errors/total <= 0.1:
        print("║  ✅ Model TỐT - Có thể train tiếp (3 epoch)                    ║")
    elif correct/total >= 0.5 and english/total > vietnamese/total:
        print("║  ⚠️  Model TRUNG BÌNH - Đúng nghĩa nhưng thiên tiếng Anh        ║")
        print("║     → Cần cải thiện training data hoặc objective              ║")
    else:
        print("║  ❌ Model CHƯA ĐẠT - Chưa nên train full                       ║")
        print("║     → Cần kiểm tra lại data hoặc approach                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

# Mẫu kết quả để điền
SAMPLE_RESULTS = """
# === 20 ẢNH TEST RESULTS ===
# Format: id, image_path, gt_category, caption_output, language, correct, has_garbage

# NHÓM 1: PHỤ KIỆN
1, img1.png, dong_ho, ..., EN/VI, 0/1, 0/1
2, img2.png, dong_ho_nu, ..., EN/VI, 0/1, 0/1
3, img3.png, tui_xach, ..., EN/VI, 0/1, 0/1
4, img4.png, tui_deo, ..., EN/VI, 0/1, 0/1
5, img5.png, vi_da, ..., EN/VI, 0/1, 0/1
6, img6.png, balo_laptop, ..., EN/VI, 0/1, 0/1

# NHÓM 2: GIÀY DÉP
7, img7.png, sneaker, ..., EN/VI, 0/1, 0/1
8, img8.png, cao_got, ..., EN/VI, 0/1, 0/1
9, img9.png, sandal, ..., EN/VI, 0/1, 0/1
10, img10.png, the_thao, ..., EN/VI, 0/1, 0/1

# NHÓM 3: MỸ PHẨM
11, img11.png, son, ..., EN/VI, 0/1, 0/1
12, img12.png, kem_duong, ..., EN/VI, 0/1, 0/1
13, img13.png, nuoc_hoa, ..., EN/VI, 0/1, 0/1
14, img14.png, bo_trang_diem, ..., EN/VI, 0/1, 0/1

# NHÓM 4: ĐỒ GIA DỤNG
15, img15.png, com_dien, ..., EN/VI, 0/1, 0/1
16, img16.png, may_xay, ..., EN/VI, 0/1, 0/1
17, img17.png, quat_dien, ..., EN/VI, 0/1, 0/1
18, img18.png, binh_nhiet, ..., EN/VI, 0/1, 0/1

# NHÓM 5: KHÓ / DỄ FAIL
19, img19.png, nhieu_san_pham, ..., EN/VI, 0/1, 0/1
20, img20.png, co_chu, ..., EN/VI, 0/1, 0/1

# Sau khi điền đủ, chạy:
# python tools/evaluate_20_images.py
"""
