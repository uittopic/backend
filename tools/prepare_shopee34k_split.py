#!/usr/bin/env python3
"""
Chuẩn bị dataset Shopee 34k để train model mới (tách biệt model cũ):
- Đọc train.csv gốc từ bộ shopee-product-matching
- Làm sạch text title (bao gồm decode một phần chuỗi escape)
- Lọc dòng thiếu ảnh/text
- Split theo label_group để hạn chế data leakage (group-aware 80/20)

Output mặc định:
- data/shopee34250_full_clean.csv
- data/shopee34250_train_80.csv
- data/shopee34250_test_20.csv
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOPEE_DIR = Path("/Users/nguyenhuuviet/Downloads/shopee-product-matching 2")
DEFAULT_INPUT_CSV = DEFAULT_SHOPEE_DIR / "train.csv"
DEFAULT_IMAGE_DIR = DEFAULT_SHOPEE_DIR / "train_images"
DEFAULT_FULL_CSV = REPO_ROOT / "data" / "shopee34250_full_clean.csv"
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "shopee34250_train_80.csv"
DEFAULT_TEST_CSV = REPO_ROOT / "data" / "shopee34250_test_20.csv"

HEX_ESCAPE_RE = re.compile(r"\\x[0-9A-Fa-f]{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Shopee 34k dataset (group split)")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-full-csv", type=Path, default=DEFAULT_FULL_CSV)
    parser.add_argument("--output-train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output-test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--text-source-column", default="title")
    parser.add_argument("--target-caption-column", default="caption_target")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decode_escaped_text(text: str) -> str:
    """
    Làm sạch text title:
    - decode chuỗi kiểu \\xNN nếu có
    - gom khoảng trắng
    """
    s = str(text).strip()
    if not s:
        return ""

    if HEX_ESCAPE_RE.search(s) or "\\u" in s or "\\U" in s:
        try:
            decoded = bytes(s, "utf-8").decode("unicode_escape")
            # Heuristic fix cho trường hợp ra dạng "Ã‰" sau unicode_escape
            if "Ã" in decoded or "Â" in decoded:
                try:
                    decoded = decoded.encode("latin1").decode("utf-8")
                except Exception:
                    pass
            s = decoded
        except Exception:
            pass

    s = re.sub(r"\s+", " ", s).strip()
    return s


def validate_ratio(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"--train-ratio phải trong (0,1), nhận: {ratio}")
    return ratio


def group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split theo group để tránh cùng label_group nằm ở cả train và test.
    """
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột group '{group_col}' trong dataset")

    group_sizes = df[group_col].value_counts().to_dict()
    groups = list(group_sizes.keys())

    rng = random.Random(seed)
    rng.shuffle(groups)

    target_train = int(len(df) * train_ratio)
    train_groups = set()
    train_count = 0

    for group_id in groups:
        if train_count < target_train:
            train_groups.add(group_id)
            train_count += int(group_sizes[group_id])

    train_groups_list: list = list(train_groups)
    train_df: pd.DataFrame = df[df[group_col].isin(train_groups_list)].copy()  # type: ignore
    test_df: pd.DataFrame = df[~df[group_col].isin(train_groups_list)].copy()  # type: ignore
    return train_df, test_df


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_ratio(args.train_ratio)

    input_csv = args.input_csv.resolve()
    image_dir = args.image_dir.resolve()
    output_full_csv = args.output_full_csv.resolve()
    output_train_csv = args.output_train_csv.resolve()
    output_test_csv = args.output_test_csv.resolve()

    print("=" * 68)
    print("📦 PREPARE SHOPEE 34K DATASET")
    print("=" * 68)
    print(f"Input CSV       : {input_csv}")
    print(f"Image directory : {image_dir}")
    print(f"Text source col : {args.text_source_column}")
    print(f"Target col      : {args.target_caption_column}")
    print(f"Train ratio     : {args.train_ratio:.2f}")
    print(f"Seed            : {args.seed}")
    print("-" * 68)

    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy input CSV: {input_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy image dir: {image_dir}")

    df = pd.read_csv(input_csv)
    required_cols = {"posting_id", "image", "label_group", args.text_source_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV thiếu cột bắt buộc: {sorted(missing_cols)}. Có: {list(df.columns)}"
        )

    print(f"✅ Raw rows: {len(df)}")

    # Làm sạch text mục tiêu
    df[args.text_source_column] = df[args.text_source_column].fillna("").astype(str)
    df[args.target_caption_column] = df[args.text_source_column].map(decode_escaped_text)

    # Lọc dữ liệu trống
    before = len(df)
    df["image"] = df["image"].astype(str).str.strip()
    df = df[(df["image"] != "") & (df[args.target_caption_column] != "")].copy()
    print(f"🧹 Drop empty image/text: {before - len(df)}")

    # Lọc theo file ảnh thực tế
    available_images_list: list = list({p.name for p in image_dir.glob("*.jpg")})
    before = len(df)
    df = df[df["image"].isin(available_images_list)].copy()  # type: ignore
    print(f"🧹 Drop missing image file: {before - len(df)}")
    print(f"✅ Clean rows: {len(df)}")

    # Split group-aware
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = group_split(
        df=df,  # type: ignore
        group_col="label_group",
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Verify không leak group
    train_groups = set(train_df["label_group"].unique())
    test_groups = set(test_df["label_group"].unique())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} groups overlap")

    # Shuffle nhẹ trước khi lưu
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    # Lưu file
    for out_path in (output_full_csv, output_train_csv, output_test_csv):
        ensure_parent(out_path)

    full_df: pd.DataFrame = df  # type: ignore
    train_split: pd.DataFrame = train_df  # type: ignore
    test_split: pd.DataFrame = test_df  # type: ignore
    full_df.to_csv(output_full_csv, index=False)
    train_split.to_csv(output_train_csv, index=False)
    test_split.to_csv(output_test_csv, index=False)

    print("-" * 68)
    print(f"💾 Full clean : {output_full_csv} ({len(df)} rows)")
    print(f"💾 Train split: {output_train_csv} ({len(train_df)} rows)")
    print(f"💾 Test split : {output_test_csv} ({len(test_df)} rows)")
    print(f"📊 Actual train ratio: {len(train_df) / len(df):.4f}")
    print(f"📊 Actual test ratio : {len(test_df) / len(df):.4f}")
    print(f"🔒 Group overlap: 0")
    print("=" * 68)


if __name__ == "__main__":
    main()
