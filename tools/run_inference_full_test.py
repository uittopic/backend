import os
import csv
from pathlib import Path

from PIL import Image
import torch

from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent
from app.core.config import IMAGES_DIR, CSV_PATH


# Đường dẫn dữ liệu để chạy inference
# Mặc định dùng cùng CSV và thư mục ảnh như khi train:
#   - CSV: data/train_bilingual_clean_v2.csv
#   - Ảnh: data/images/
TEST_IMG_DIR = IMAGES_DIR
TEST_CSV = CSV_PATH

# Đường dẫn file output
OUTPUT_CSV = Path("outputs/predictions_test.csv")


def get_device() -> torch.device:
    """
    Lấy device hiện tại.
    Ở đây ưu tiên MPS (macOS), sau đó đến CUDA, cuối cùng là CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


def generate_caption(image_path: Path):
    """
    Sinh caption cho một ảnh:
    - BLIP: caption tiếng Việt không dấu
    - Accent Restoration: thêm dấu tiếng Việt
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=25,
            num_beams=3,
            early_stopping=True,
        )

    caption_no_accent = processor.decode(output[0], skip_special_tokens=True)
    caption_with_accent = restore_accent(caption_no_accent)

    return caption_no_accent, caption_with_accent


def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV tại {TEST_CSV}. "
            "Hãy đảm bảo đã có file CSV (ví dụ train_bilingual_clean_v2.csv) đúng vị trí."
        )

    if not TEST_IMG_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh tại {TEST_IMG_DIR}. "
            "Hãy đảm bảo đã copy ảnh vào đúng thư mục (ví dụ data/images/)."
        )

    rows = []

    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image"]
            img_path = TEST_IMG_DIR / img_name

            if not img_path.exists():
                print(f"⚠️  Không tìm thấy ảnh: {img_path}, bỏ qua.")
                continue

            ca_no_ac, ca_full = generate_caption(img_path)

            rows.append(
                {
                    "image": img_name,
                    "caption_no_accent": ca_no_ac,
                    "caption_full": ca_full,
                }
            )

            print(f"✔ {img_name} → {ca_full}")

    # Đảm bảo thư mục outputs tồn tại
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "caption_no_accent", "caption_full"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n🎉 DONE! File output nằm tại:")
    print("➡", OUTPUT_CSV)


if __name__ == "__main__":
    main()


