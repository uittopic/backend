#!/usr/bin/env python3
"""
Xuất bảng kết quả và báo cáo markdown từ outputs/experiment_results.csv.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).parent.parent
DEFAULT_RESULTS_CSV = BASE_DIR / "outputs" / "experiment_results.csv"
DEFAULT_EXPORT_CSV = BASE_DIR / "outputs" / "experiment_summary_latest.csv"
DEFAULT_EXPORT_MD = BASE_DIR / "outputs" / "experiment_summary_latest.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export experiment ranking report")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_EXPORT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_EXPORT_MD)
    return parser.parse_args()


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_row(row: Dict[str, str]) -> float:
    bleu = safe_float(row.get("bleu", "0"))
    rouge = safe_float(row.get("rouge_l", "0"))
    sbert = safe_float(row.get("sbert_similarity", "0"))
    # Ưu tiên semantic similarity, sau đó ROUGE và BLEU
    return (0.5 * sbert) + (0.3 * rouge) + (0.2 * bleu)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kết quả: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, ranked_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append("# Báo cáo thực nghiệm (tự động)")
    lines.append("")
    lines.append(f"- Thời điểm xuất: `{now}`")
    lines.append(f"- Số thực nghiệm: `{len(ranked_rows)}`")
    lines.append("")

    if ranked_rows:
        best = ranked_rows[0]
        lines.append("## Cấu hình tốt nhất hiện tại")
        lines.append("")
        lines.append(f"- EXP ID: `{best.get('exp_id', '')}`")
        lines.append(f"- Composite Score: `{best.get('composite_score', 0):.4f}`")
        lines.append(f"- BLEU: `{best.get('bleu', 0)}`")
        lines.append(f"- ROUGE-L: `{best.get('rouge_l', 0)}`")
        lines.append(f"- SBERT: `{best.get('sbert_similarity', 0)}`")
        lines.append("")

    lines.append("## Bảng xếp hạng")
    lines.append("")
    lines.append("| Rank | EXP ID | Status | BLEU | ROUGE-L | SBERT | Composite | Notes |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for row in ranked_rows:
        lines.append(
            f"| {row.get('rank','')} | {row.get('exp_id','')} | {row.get('status','')} | "
            f"{row.get('bleu',0)} | {row.get('rouge_l',0)} | {row.get('sbert_similarity',0)} | "
            f"{row.get('composite_score',0)} | {row.get('notes','')} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results_csv)

    done_rows = [r for r in rows if str(r.get("status", "")).upper() == "DONE"]
    if not done_rows:
        print("⚠️  Không có thực nghiệm status=DONE để xuất báo cáo.")
        done_rows = rows

    ranked = []
    for row in done_rows:
        ranked.append(
            {
                **row,
                "composite_score": round(score_row(row), 6),
            }
        )

    ranked.sort(key=lambda r: float(r["composite_score"]), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    fieldnames = [
        "rank",
        "exp_id",
        "status",
        "objective",
        "bleu",
        "rouge_l",
        "sbert_similarity",
        "composite_score",
        "model_path",
        "train_csv",
        "val_csv",
        "test_csv",
        "epochs",
        "train_batch_size",
        "learning_rate",
        "num_beams",
        "no_repeat_ngram_size",
        "repetition_penalty",
        "use_accent",
        "timestamp",
        "owner",
        "notes",
    ]

    # Filter fieldnames to only include existing keys
    for row in ranked:
        for key in list(row.keys()):
            if key not in fieldnames and key != "composite_score":
                del row[key]

    write_csv(args.output_csv, ranked, fieldnames)
    write_markdown(args.output_md, ranked)

    print("✅ Đã xuất báo cáo thực nghiệm")
    print(f"   - CSV: {args.output_csv}")
    print(f"   - MD:  {args.output_md}")
    if ranked:
        print(
            f"   - Best: {ranked[0]['exp_id']} "
            f"(score={ranked[0]['composite_score']})"
        )


if __name__ == "__main__":
    main()
