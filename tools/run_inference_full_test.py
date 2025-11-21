import os
import csv
from pathlib import Path

from PIL import Image
import torch

from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent
from app.core.config import DATA_DIR, IMAGES_DIR, CSV_PATH


# Đường dẫn dữ liệu để chạy inference
# Ưu tiên dùng split test 20% nếu có, và cho phép override qua ENV
DEFAULT_TEST_CSV = DATA_DIR / "test_20.csv"
TEST_CSV = Path(
    os.environ.get(
        "INFER_CSV_PATH",
        DEFAULT_TEST_CSV if DEFAULT_TEST_CSV.exists() else CSV_PATH,
    )
)
TEST_IMG_DIR = Path(os.environ.get("INFER_IMAGES_DIR", IMAGES_DIR))

# Đường dẫn file output
OUTPUT_CSV = Path(os.environ.get("INFER_OUTPUT_CSV", "outputs/predictions_test.csv"))


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
    
    # Fix cho MPS: Chuyển model về CPU khi generate vì MPS không hỗ trợ tốt attention_mask auto-inference
    # BLIP sẽ tự tạo input_ids cho text decoder, nhưng trên MPS cần attention_mask rõ ràng
    # Cách đơn giản nhất: chuyển về CPU cho text decoder generation
    generate_device = device
    if device.type == "mps":
        # Chuyển model về CPU tạm thời cho generation
        model_cpu = model.cpu()
        inputs_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
    else:
        model_cpu = model
        inputs_cpu = inputs

    with torch.no_grad():
        output = model_cpu.generate(
            **inputs_cpu,
            max_new_tokens=25,
            num_beams=3,
            early_stopping=True,
        )
    
    # Chuyển output về device ban đầu nếu cần
    if device.type == "mps":
        output = output.to(device)
        model.to(device)  # Chuyển model về MPS lại

    caption_no_accent = processor.decode(output[0], skip_special_tokens=True)
    caption_with_accent = restore_accent(caption_no_accent)

    return caption_no_accent, caption_with_accent


def main():
    print("=" * 60)
    print("🚀 BẮT ĐẦU INFERENCE")
    print("=" * 60)
    print(f"📁 CSV input: {TEST_CSV}")
    print(f"📁 Thư mục ảnh: {TEST_IMG_DIR}")
    print(f"📁 CSV output: {OUTPUT_CSV}")
    print()
    
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

    print("📖 Đang đọc file CSV...")
    rows = []
    all_rows = []

    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    total = len(all_rows)
    print(f"✅ Đã đọc {total} ảnh từ CSV")
    print(f"🔄 Bắt đầu inference...")
    print()
    
    for idx, row in enumerate(all_rows, 1):
            img_name = row["image"]
            img_path = TEST_IMG_DIR / img_name

            if not img_path.exists():
                print(f"[{idx}/{total}] ⚠️  Không tìm thấy ảnh: {img_path}, bỏ qua.")
                continue

            print(f"[{idx}/{total}] 🔄 Đang xử lý: {img_name}...", end=" ", flush=True)
            ca_no_ac, ca_full = generate_caption(img_path)
            print(f"✅ {ca_full}")

            rows.append(
                {
                    "image": img_name,
                    "caption_no_accent": ca_no_ac,
                    "caption_full": ca_full,
                }
            )

    print()
    print("💾 Đang ghi file output...")
    # Đảm bảo thư mục outputs tồn tại
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "caption_no_accent", "caption_full"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("🎉 DONE! File output nằm tại:")
    print("➡", OUTPUT_CSV)
    print(f"✅ Đã xử lý {len(rows)}/{total} ảnh")
    print("=" * 60)


if __name__ == "__main__":
    main()


