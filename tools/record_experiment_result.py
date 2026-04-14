#!/usr/bin/env python3
"""
Ghi kết quả thực nghiệm vào bảng tổng hợp thống nhất.

Input chính:
- outputs/evaluation_metrics.csv (metric,score)

Output:
- outputs/experiment_results.csv
- cập nhật configs/experiment_matrix.csv nếu có exp_id tương ứng
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


BASE_DIR = Path(__file__).parent.parent
DEFAULT_METRICS_CSV = BASE_DIR / "outputs" / "evaluation_metrics.csv"
DEFAULT_RESULTS_CSV = BASE_DIR / "outputs" / "experiment_results.csv"
DEFAULT_MATRIX_CSV = BASE_DIR / "configs" / "experiment_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one experiment result")
    parser.add_argument("--exp-id", required=True, help="Mã thực nghiệm, ví dụ: EXP-02-BEAM2")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)

    parser.add_argument("--objective", default="", help="Mục tiêu thực nghiệm")
    parser.add_argument("--model-path", default="models/blip_vietnamese_80_20")
    parser.add_argument("--train-csv", default="data/train_80.csv")
    parser.add_argument("--val-csv", default="data/test_20.csv")
    parser.add_argument("--test-csv", default="data/test_20.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--use-accent", choices=["true", "false"], default="true")
    parser.add_argument("--status", default="DONE", choices=["PLANNED", "RUNNING", "DONE", "FAILED"])
    parser.add_argument("--notes", default="")

    return parser.parse_args()


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_metrics(metrics_csv: Path) -> Dict[str, float]:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy metrics file: {metrics_csv}")

    metrics: Dict[str, float] = {}
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = str(row.get("metric", "")).strip()
            score = safe_float(str(row.get("score", "")).strip(), 0.0)
            if metric_name:
                metrics[metric_name] = score
    return metrics


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) for k, v in row.items()})


def upsert_results(results_csv: Path, new_row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "exp_id",
        "status",
        "objective",
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
        "bleu",
        "rouge_l",
        "sbert_similarity",
        "notes",
        "owner",
    ]

    existing = read_csv_rows(results_csv)
    updated = False
    for row in existing:
        if row.get("exp_id") == new_row["exp_id"]:
            for key, value in new_row.items():
                row[key] = str(value)
            updated = True
            break

    if not updated:
        existing.append({k: str(v) for k, v in new_row.items()})

    existing.sort(key=lambda r: str(r.get("exp_id", "")))
    write_csv_rows(results_csv, fieldnames, existing)


def update_experiment_matrix(matrix_csv: Path, new_row: Dict[str, Any]) -> None:
    if not matrix_csv.exists():
        return

    rows = read_csv_rows(matrix_csv)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    has_match = False
    for row in rows:
        if row.get("exp_id") == new_row["exp_id"]:
            has_match = True
            row["objective"] = str(new_row["objective"])
            row["model_path"] = str(new_row["model_path"])
            row["train_csv"] = str(new_row["train_csv"])
            row["val_csv"] = str(new_row["val_csv"])
            row["test_csv"] = str(new_row["test_csv"])
            row["epochs"] = str(new_row["epochs"])
            row["train_batch_size"] = str(new_row["train_batch_size"])
            row["learning_rate"] = str(new_row["learning_rate"])
            row["num_beams"] = str(new_row["num_beams"])
            row["no_repeat_ngram_size"] = str(new_row["no_repeat_ngram_size"])
            row["repetition_penalty"] = str(new_row["repetition_penalty"])
            row["use_accent"] = str(new_row["use_accent"])
            row["status"] = str(new_row["status"])
            row["bleu"] = str(new_row["bleu"])
            row["rouge_l"] = str(new_row["rouge_l"])
            row["sbert_similarity"] = str(new_row["sbert_similarity"])
            row["last_run_at"] = str(new_row["timestamp"])
            row["notes"] = str(new_row["notes"])
            break

    if not has_match:
        append_row: Dict[str, str] = {k: "" for k in fieldnames}
        append_row["exp_id"] = str(new_row["exp_id"])
        for key in append_row.keys():
            if key in new_row:
                append_row[key] = str(new_row[key])
        append_row["last_run_at"] = str(new_row["timestamp"])
        rows.append(append_row)

    write_csv_rows(matrix_csv, fieldnames, rows)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_csv)

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exp_id": args.exp_id,
        "status": args.status,
        "objective": args.objective,
        "model_path": args.model_path,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "use_accent": args.use_accent,
        "bleu": round(metrics.get("BLEU", 0.0), 6),
        "rouge_l": round(metrics.get("ROUGE-L", 0.0), 6),
        "sbert_similarity": round(metrics.get("SBERT_Similarity", 0.0), 6),
        "notes": args.notes,
        "owner": os.environ.get("USER", "unknown"),
    }

    upsert_results(args.results_csv, row)
    update_experiment_matrix(args.matrix_csv, row)

    print("Đã ghi kết quả thực nghiệm")
    print(f"   - Results: {args.results_csv}")
    print(f"   - Matrix:  {args.matrix_csv}")
    print(
        f"   - Metrics: BLEU={row['bleu']}, "
        f"ROUGE-L={row['rouge_l']}, SBERT={row['sbert_similarity']}"
    )


if __name__ == "__main__":
    main()
