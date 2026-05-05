#!/usr/bin/env python3
"""
Blind test tool — chấm điểm thủ công caption sinh ra bởi model.

Chạy: python tools/blind_test.py --count 50
  - Hiển thị ảnh + caption sinh ra (KHÔNG có ground truth)
  - Chấm điểm 1-5 cho từng ảnh
  - Log kết quả vào outputs/blind_test/
  - Resume được nếu bị interrupt

Điểm:
  1 — Rất tệ: caption sai hoàn toàn, không liên quan ảnh
  2 — Tệ: caption có liên quan yếu, sai nhiều
  3 — Trung bình: caption đúng một phần, còn lỗi
  4 — Tốt: caption đúng, có thể dùng được
  5 — Xuất sắc: caption chính xác, mạch lạc
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# ── CLI colors ────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def color(s: str, c: str) -> str:
    return f"{c}{s}{RESET}"


def bold(s: str) -> str:
    return color(s, BOLD)


def green(s: str) -> str:
    return color(s, GREEN)


def red(s: str) -> str:
    return color(s, RED)


def yellow(s: str) -> str:
    return color(s, YELLOW)


def cyan(s: str) -> str:
    return color(s, CYAN)


# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
INPUT_CSV  = BASE_DIR / "outputs" / "full_batch_test" / "detailed_predictions.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "blind_test"
LOG_FILE   = OUTPUT_DIR / "blind_test_results.csv"
SESSION_FILE = OUTPUT_DIR / ".session.json"


# ── Rating description ──────────────────────────────────────────
RATINGS = {
    1: "Rất tệ — sai hoàn toàn, không liên quan",
    2: "Tệ — liên quan yếu, sai nhiều",
    3: "Trung bình — đúng một phần, còn lỗi",
    4: "Tốt — đúng, có thể dùng được",
    5: "Xuất sắc — chính xác, mạch lạc",
}


# ── Session management ──────────────────────────────────────────
def load_session() -> dict:
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "results": {}, "started_at": datetime.now().isoformat()}


def save_session(session: dict) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def load_results() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_result(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    fieldnames = ["idx", "image", "caption", "rating", "notes", "timestamp",
                  "gt_available", "gt_caption"]
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Load dataset ────────────────────────────────────────────────
def load_samples(count: int, seed: int) -> list[dict]:
    """Load `count` random samples from detailed_predictions.csv, excluding done ones."""
    session = load_session()
    done_ids = set(session.get("done", []))

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Filter out already-done images
    remaining = [r for r in all_rows if r["image"] not in done_ids]
    if not remaining:
        print(red("Đã chấm hết tất cả ảnh!"))
        sys.exit(0)

    random.seed(seed)
    samples = random.sample(remaining, min(count, len(remaining)))
    return samples


# ── Display helpers ─────────────────────────────────────────────
def clear_screen() -> None:
    print("\033[2J\033[H", end="")  # VT100 clear


def print_header(idx: int, total: int) -> None:
    clear_screen()
    print("=" * 60)
    print(f"  {bold('BLIND TEST')} — Ảnh {green(str(idx))}/{yellow(str(total))}")
    print("=" * 60)


def print_caption(caption: str) -> None:
    print(f"\n  {bold('Caption sinh ra:')}")
    print(f"  {'─' * 56}")
    # Wrap text ~60 chars
    for i in range(0, len(caption), 60):
        print(f"  {caption[i:i+60]}")
    print(f"  {'─' * 56}\n")


def print_ratings() -> None:
    for score, desc in RATINGS.items():
        bar = "★" * score + "☆" * (5 - score)
        print(f"  [{bold(str(score))}] {bar}  {desc}")


def print_image_info(image_name: str) -> None:
    img_path = IMAGES_DIR / image_name
    status = green("✓ Tìm thấy") if img_path.exists() else red("✗ Không tìm thấy")
    print(f"  {bold('Ảnh:')} {image_name}  [{status}]")
    if not img_path.exists():
        print(f"  {yellow('⚠ Không tìm thấy ảnh — bỏ qua đánh giá')}")


def print_progress(done: int, total: int) -> None:
    pct = done / total * 100
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Tiến độ: [{green(bar)}] {done}/{total} ({pct:.0f}%)")


def print_stats(results: list[dict]) -> None:
    if not results:
        return
    scores = [int(r["rating"]) for r in results]
    n = len(scores)
    avg = sum(scores) / n

    score_counts = {i: scores.count(i) for i in range(1, 6)}

    print(f"\n  {'─' * 56}")
    print(f"  {bold('THỐNG KÊ HIỆN TẠI (chưa lưu)')}")
    print(f"  {'─' * 56}")
    print(f"  Đã chấm: {n}/{len(results) + n}  |  Avg: {avg:.2f}/5")
    for s in range(1, 6):
        bar = "█" * score_counts[s]
        print(f"  [{s}] {RATINGS[s][:30]:<30} {score_counts[s]:>4} ({score_counts[s]/n*100:.0f}%) {bar}")
    print(f"  {'─' * 56}")


# ── Main loop ───────────────────────────────────────────────────
def run(count: int, seed: int, view_gt: bool) -> None:
    samples = load_samples(count, seed)
    session = load_session()

    total = len(samples)
    for i, row in enumerate(samples, 1):
        image = row["image"]
        caption = row["prediction"]
        gt = row.get("ground_truth", "")
        img_path = IMAGES_DIR / image

        print_header(i, total)
        print_image_info(image)

        # Try to display image
        if img_path.exists():
            try:
                # macOS
                import subprocess
                subprocess.run(["open", str(img_path)], capture_output=True)
            except Exception:
                pass

        print_caption(caption)

        if view_gt and gt:
            print(f"  {bold('Ground truth (ẩn trong blind test):')}")
            print(f"  {yellow(gt)}")

        print_ratings()
        print_stats(load_results())

        print(f"\n  {bold('Nhập điểm [1-5], q=thống kê, s=skip, x=thoát:')}")
        choice = input(f"  → {bold('')}").strip().lower()

        if choice == "x":
            print(red("\nĐã dừng. Kết quả đã lưu được giữ nguyên."))
            sys.exit(0)
        elif choice == "q":
            results = load_results()
            print_stats(results)
            input(f"\n  {bold('Enter để tiếp tục...')}")
            continue
        elif choice == "s":
            print(yellow("  → Đã bỏ qua"))
            session["done"].append(image)
            save_session(session)
            continue

        try:
            score = int(choice)
            if score < 1 or score > 5:
                raise ValueError
        except ValueError:
            print(red(f"  → Nhập số 1-5, q, s, hoặc x. Bỏ qua ảnh này."))
            input(f"  {bold('Enter để tiếp tục...')}")
            continue

        # Optional notes
        print(f"  Ghi chú (Enter để bỏ qua):")
        notes = input(f"  → {''}").strip()

        # Save
        log_row = {
            "idx": i,
            "image": image,
            "caption": caption,
            "rating": score,
            "notes": notes,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "gt_available": "true" if gt else "false",
            "gt_caption": gt if gt else "",
        }
        save_result(log_row)
        session["done"].append(image)
        session["results"][image] = score
        save_session(session)

        # Quick feedback
        if score >= 4:
            print(green(f"  ✓ Điểm {score} — Tốt!"))
        elif score == 3:
            print(yellow(f"  · Điểm {score} — Trung bình"))
        else:
            print(red(f"  ✗ Điểm {score} — Cần cải thiện"))

    # ── Final summary ─────────────────────────────────────────
    results = load_results()
    scores = [int(r["rating"]) for r in results]
    n = len(scores)
    avg = sum(scores) / n

    score_counts = {i: scores.count(i) for i in range(1, 6)}

    print(f"\n{'=' * 60}")
    print(f"  {bold('KẾT QUẢ BLIND TEST')}")
    print(f"{'=' * 60}")
    print(f"  Tổng ảnh đã chấm:  {n}")
    print(f"  Điểm trung bình:    {green(f'{avg:.2f}/5')}")
    print(f"  Median:             {sorted(scores)[n//2]}/5")
    for s in range(1, 6):
        pct = score_counts[s] / n * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  [{s}] {RATINGS[s][:28]:<28} {score_counts[s]:>3} ({pct:5.1f}%) {bar}")
    print(f"{'=' * 60}")
    print(f"  Kết quả lưu tại: {green(str(LOG_FILE.absolute()) )}")
    print(f"  Xem lại:         python tools/blind_test.py --report")
    print()


def show_report() -> None:
    """Print summary report from saved results."""
    results = load_results()
    if not results:
        print(red("Chưa có kết quả blind test nào."))
        return

    scores = [int(r["rating"]) for r in results]
    n = len(scores)
    avg = sum(scores) / n
    score_counts = {i: scores.count(i) for i in range(1, 6)}

    print(f"\n{'=' * 60}")
    print(f"  {bold('BLIND TEST REPORT')}")
    print(f"{'=' * 60}")
    print(f"  Ngày:   {results[0]['timestamp'][:10]} → {results[-1]['timestamp'][:10]}")
    print(f"  Tổng:   {n} ảnh")
    print(f"  Avg:    {avg:.2f}/5")
    print(f"  Median: {sorted(scores)[n//2]}/5")
    print(f"  Std:    {(sum((s-avg)**2 for s in scores)/n)**0.5:.2f}")

    print(f"\n  {bold('Phân bố điểm:')}")
    for s in range(1, 6):
        pct = score_counts[s] / n * 100
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        label = RATINGS[s][:30]
        print(f"  [{s}] {label:<30} {score_counts[s]:>3} ({pct:5.1f}%) {bar}")

    # Quality tiers
    good = sum(score_counts[i] for i in [4, 5])
    mid  = score_counts[3]
    bad  = sum(score_counts[i] for i in [1, 2])
    print(f"\n  {bold('Phân loại:')}")
    print(f"  Tốt (4-5):    {green(good)} ({good/n*100:.1f}%)")
    print(f"  Trung bình:   {yellow(mid)} ({mid/n*100:.1f}%)")
    print(f"  Cần cải thiện:{red(bad)} ({bad/n*100:.1f}%)")

    # Sample bad ones
    bad_rows = [r for r in results if int(r["rating"]) <= 2]
    if bad_rows:
        print(f"\n  {bold('Caption tệ (điểm 1-2):')}")
        for r in bad_rows[:5]:
            print(f"  [{r['rating']}] {r['caption'][:60]}")

    print(f"\n  File: {LOG_FILE}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind test — chấm điểm thủ công caption sinh ra"
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Số ảnh cần chấm (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed để tái tạo (default: 42)"
    )
    parser.add_argument(
        "--view-gt", action="store_true",
        help="Hiển thị ground truth sau caption (khuyến nghị: KHÔNG dùng khi đánh giá thật)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Chỉ hiển thị báo cáo từ kết quả đã lưu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.report:
        show_report()
    else:
        run(count=args.count, seed=args.seed, view_gt=args.view_gt)
