import os
import sys
from pathlib import Path

# Tự động thêm project root vào sys.path để import module 'app'
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import csv
from typing import Any

from PIL import Image
import torch

from app.core.model_loader import model, processor
from app.core.accent_restoration_loader import restore_accent
from app.core.config import DATA_DIR, IMAGES_DIR, CSV_PATH


DEFAULT_TEST_CSV = DATA_DIR / "test_20.csv"
TEST_CSV = Path(
    os.environ.get(
        "INFER_CSV_PATH",
        str(DEFAULT_TEST_CSV if DEFAULT_TEST_CSV.exists() else CSV_PATH),
    )
)
TEST_IMG_DIR = Path(os.environ.get("INFER_IMAGES_DIR", str(IMAGES_DIR)))
OUTPUT_CSV = Path(os.environ.get("INFER_OUTPUT_CSV", "outputs/predictions_test.csv"))


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
model_any: Any = model  # type: ignore
processor_any: Any = processor  # type: ignore


def generate_caption(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor_any(images=image, return_tensors="pt")  # type: ignore
    inputs_pt = inputs.to(device)  # type: ignore

    if device.type == "mps":
        model_cpu = model_any.cpu()  # type: ignore
        inputs_cpu: Any = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs_pt.items()}
    else:
        model_cpu = model_any
        inputs_cpu = inputs_pt

    with torch.no_grad():
        output = model_cpu.generate(  # type: ignore
            **inputs_cpu,
            max_new_tokens=25,
            num_beams=3,
            early_stopping=True,
        )

    if device.type == "mps":
        model_any.to(device)  # type: ignore

    caption_no_accent = processor_any.decode(output[0], skip_special_tokens=True)  # type: ignore
    caption_with_accent = restore_accent(caption_no_accent)

    return caption_no_accent, caption_with_accent


def main():
    print("=" * 60)
    print("🚀 BAT DAU INFERENCE")
    print("=" * 60)
    print(f"📁 CSV input: {TEST_CSV}")
    print(f"📁 Thu muc anh: {TEST_IMG_DIR}")
    print(f"📁 CSV output: {OUTPUT_CSV}")
    print()

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Khong tim thay file CSV tai {TEST_CSV}. "
            "Hay dam bao da co file CSV (vi du train_bilingual_clean_v2.csv) dung vi tri."
        )

    if not TEST_IMG_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc anh tai {TEST_IMG_DIR}. "
            "Hay dam bao da copy anh vao dung thu muc (vi du data/images/)."
        )

    print("Dang doc file CSV...")
    rows = []
    all_rows = []

    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    total = len(all_rows)
    print(f"Da doc {total} anh tu CSV")
    print(f"Bat dau inference...")
    print()

    for idx, row in enumerate(all_rows, 1):
        img_name = row["image"]
        img_path = TEST_IMG_DIR / img_name

        if not img_path.exists():
            print(f"[{idx}/{total}] Khong tim thay anh: {img_path}, bo qua.")
            continue

        print(f"[{idx}/{total}] Dang xu ly: {img_name}...", end=" ", flush=True)
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
    print("Dang ghi file output...")
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption_no_accent", "caption_full"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("DONE! File output nam tai:")
    print(f"➡ {OUTPUT_CSV}")
    print(f"Da xu ly {len(rows)}/{total} anh")
    print("=" * 60)


if __name__ == "__main__":
    main()
