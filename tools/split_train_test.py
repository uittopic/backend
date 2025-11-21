"""
Utility để tách dataset thành train/test theo tỷ lệ (mặc định 80/20).
"""
import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "train_bilingual_clean_v2.csv"
DEFAULT_TRAIN_OUTPUT = DATA_DIR / "train_80.csv"
DEFAULT_TEST_OUTPUT = DATA_DIR / "test_20.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset CSV thành train/test.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Đường dẫn tới CSV gốc (mặc định: data/train_bilingual_clean_v2.csv)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="File CSV lưu tập train (mặc định: data/train_80.csv)",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT,
        help="File CSV lưu tập test (mặc định: data/test_20.csv)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Tỷ lệ train (0-1). Mặc định 0.8 = 80%% train, 20%% test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed cho việc shuffle dữ liệu (mặc định: 42)",
    )
    return parser.parse_args()


def validate_ratio(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"Tỷ lệ train phải nằm trong (0,1). Giá trị hiện tại: {ratio}")
    return ratio


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ratio = validate_ratio(args.train_ratio)

    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {args.input}")

    print("=" * 60)
    print("🔀 Đang tách dataset thành train/test")
    print("=" * 60)
    print(f"📥 Input CSV : {args.input}")
    print(f"📤 Train CSV : {args.train_output}")
    print(f"📤 Test CSV  : {args.test_output}")
    print(f"📊 Train ratio: {ratio:.2f}")
    print(f"🎲 Random state: {args.random_state}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"Dataset rỗng: {args.input}")

    df = df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    split_idx = int(len(df) * ratio)

    if split_idx == 0 or split_idx == len(df):
        raise ValueError("Split tạo ra tập train/test rỗng. Kiểm tra lại tỷ lệ.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    ensure_parent_dir(args.train_output)
    ensure_parent_dir(args.test_output)

    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print("\n✅ Hoàn thành!")
    print(f"   • Train samples: {len(train_df)}")
    print(f"   • Test samples : {len(test_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

