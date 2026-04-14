#!/usr/bin/env python3
"""
Bulk test /api/caption_full hoặc /api/caption_full/batch trên toàn bộ ảnh trong CSV.

Mục tiêu:
- Chạy được lâu (hàng nghìn ảnh) và có thể resume nếu bị ngắt.
- Lưu kết quả từng batch để không mất dữ liệu đã test.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk test caption_full API")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--csv-path", type=Path, default=Path("data/train_80.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/api_caption_full_train80_results.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["batch", "single"],
        default="batch",
        help="batch dùng /api/caption_full/batch, single dùng /api/caption_full",
    )
    return parser.parse_args()


def read_input_rows(csv_path: Path, max_rows: int = 0) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            image = str(row.get("image", "")).strip()
            if image:
                rows.append((idx, image))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def read_done_indices(output_csv: Path) -> Set[int]:
    done: Set[int] = set()
    if not output_csv.exists():
        return done
    with output_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["row_index"]))
            except Exception:
                continue
    return done


def ensure_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "image",
                "success",
                "http_status",
                "caption_vi",
                "caption_vi_no_accent",
                "accent_restored",
                "cached",
                "processing_time",
                "error",
                "batch_elapsed",
                "ts",
            ],
        )
        writer.writeheader()


def append_rows(output_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "image",
                "success",
                "http_status",
                "caption_vi",
                "caption_vi_no_accent",
                "accent_restored",
                "cached",
                "processing_time",
                "error",
                "batch_elapsed",
                "ts",
            ],
        )
        writer.writerows(rows)


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_batch_request(
    session: requests.Session,
    api_base: str,
    image_dir: Path,
    batch: List[Tuple[int, str]],
    timeout: int,
) -> List[Dict[str, object]]:
    ts = int(time.time())
    files: List[Tuple[str, Tuple[str, Any, str]]] = []
    opened = []
    missing_rows: List[Tuple[int, str]] = []
    valid_rows: List[Tuple[int, str]] = []

    for row_idx, image_name in batch:
        image_path = image_dir / image_name
        if not image_path.exists():
            missing_rows.append((row_idx, image_name))
            continue
        fh = image_path.open("rb")
        opened.append(fh)
        files.append(("files", (image_name, fh, "image/jpeg")))
        valid_rows.append((row_idx, image_name))

    results: List[Dict[str, object]] = []

    for row_idx, image_name in missing_rows:
        results.append(
            {
                "row_index": row_idx,
                "image": image_name,
                "success": False,
                "http_status": 0,
                "caption_vi": "",
                "caption_vi_no_accent": "",
                "accent_restored": False,
                "cached": False,
                "processing_time": 0,
                "error": "image_not_found",
                "batch_elapsed": 0,
                "ts": ts,
            }
        )

    if not valid_rows:
        return results

    url = f"{api_base.rstrip('/')}/api/caption_full/batch"
    start = time.time()
    http_status = 0
    try:
        resp = session.post(url, files=files, timeout=timeout)  # type: ignore
        http_status = resp.status_code
        elapsed = time.time() - start
        if resp.status_code == 200:
            payload = resp.json()
            payload_results = payload.get("results", [])
            for i, (row_idx, image_name) in enumerate(valid_rows):
                item = payload_results[i] if i < len(payload_results) else {}
                results.append(
                    {
                        "row_index": row_idx,
                        "image": image_name,
                        "success": bool(item.get("success", False)),
                        "http_status": http_status,
                        "caption_vi": item.get("caption_vi", ""),
                        "caption_vi_no_accent": item.get("caption_vi_no_accent", ""),
                        "accent_restored": bool(item.get("accent_restored", False)),
                        "cached": bool(item.get("cached", False)),
                        "processing_time": payload.get("processing_time", 0),
                        "error": item.get("error", ""),
                        "batch_elapsed": round(elapsed, 3),
                        "ts": ts,
                    }
                )
        else:
            err = (resp.text or "")[:500]
            for row_idx, image_name in valid_rows:
                results.append(
                    {
                        "row_index": row_idx,
                        "image": image_name,
                        "success": False,
                        "http_status": http_status,
                        "caption_vi": "",
                        "caption_vi_no_accent": "",
                        "accent_restored": False,
                        "cached": False,
                        "processing_time": 0,
                        "error": f"http_{http_status}:{err}",
                        "batch_elapsed": round(elapsed, 3),
                        "ts": ts,
                    }
                )
    except Exception as e:
        elapsed = time.time() - start
        for row_idx, image_name in valid_rows:
            results.append(
                {
                    "row_index": row_idx,
                    "image": image_name,
                    "success": False,
                    "http_status": http_status,
                    "caption_vi": "",
                    "caption_vi_no_accent": "",
                    "accent_restored": False,
                    "cached": False,
                    "processing_time": 0,
                    "error": f"exception:{e}",
                    "batch_elapsed": round(elapsed, 3),
                    "ts": ts,
                }
            )
    finally:
        for fh in opened:
            try:
                fh.close()
            except Exception:
                pass

    return results


def run_single_request(
    session: requests.Session,
    api_base: str,
    image_dir: Path,
    row: Tuple[int, str],
    timeout: int,
) -> Dict[str, object]:
    row_idx, image_name = row
    ts = int(time.time())
    image_path = image_dir / image_name
    if not image_path.exists():
        return {
            "row_index": row_idx,
            "image": image_name,
            "success": False,
            "http_status": 0,
            "caption_vi": "",
            "caption_vi_no_accent": "",
            "accent_restored": False,
            "cached": False,
            "processing_time": 0,
            "error": "image_not_found",
            "batch_elapsed": 0,
            "ts": ts,
        }

    url = f"{api_base.rstrip('/')}/api/caption_full"
    start = time.time()
    http_status = 0
    try:
        with image_path.open("rb") as fh:
            resp = session.post(
                url, files={"file": (image_name, fh, "image/jpeg")}, timeout=timeout
            )
        http_status = resp.status_code
        elapsed = time.time() - start
        if resp.status_code == 200:
            payload = resp.json()
            return {
                "row_index": row_idx,
                "image": image_name,
                "success": bool(payload.get("success", False)),
                "http_status": http_status,
                "caption_vi": payload.get("caption_vi", ""),
                "caption_vi_no_accent": payload.get("caption_vi_no_accent", ""),
                "accent_restored": bool(payload.get("accent_restored", False)),
                "cached": bool(payload.get("cached", False)),
                "processing_time": payload.get("processing_time", 0),
                "error": "",
                "batch_elapsed": round(elapsed, 3),
                "ts": ts,
            }
        return {
            "row_index": row_idx,
            "image": image_name,
            "success": False,
            "http_status": http_status,
            "caption_vi": "",
            "caption_vi_no_accent": "",
            "accent_restored": False,
            "cached": False,
            "processing_time": 0,
            "error": f"http_{http_status}:{(resp.text or '')[:500]}",
            "batch_elapsed": round(elapsed, 3),
            "ts": ts,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "row_index": row_idx,
            "image": image_name,
            "success": False,
            "http_status": http_status,
            "caption_vi": "",
            "caption_vi_no_accent": "",
            "accent_restored": False,
            "cached": False,
            "processing_time": 0,
            "error": f"exception:{e}",
            "batch_elapsed": round(elapsed, 3),
            "ts": ts,
        }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size phải > 0")
    if args.mode == "batch" and args.batch_size > 10:
        raise ValueError("--batch-size cho mode=batch nên <= 10 (theo MAX_BATCH_SIZE API)")

    csv_path = args.csv_path.resolve()
    image_dir = args.image_dir.resolve()
    output_csv = args.output_csv.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy csv: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy image_dir: {image_dir}")

    rows = read_input_rows(csv_path=csv_path, max_rows=args.max_rows)
    ensure_output_header(output_csv)

    done_indices: Set[int] = set()
    if not args.no_resume:
        done_indices = read_done_indices(output_csv)

    todo = [r for r in rows if r[0] not in done_indices]

    print("=" * 80)
    print("BULK TEST CAPTION FULL API")
    print(f"api_base      : {args.api_base}")
    print(f"mode          : {args.mode}")
    print(f"csv_path      : {csv_path}")
    print(f"image_dir     : {image_dir}")
    print(f"output_csv    : {output_csv}")
    print(f"batch_size    : {args.batch_size}")
    print(f"timeout       : {args.timeout}s")
    print(f"total_rows    : {len(rows)}")
    print(f"done_from_csv : {len(done_indices)}")
    print(f"todo_rows     : {len(todo)}")
    print("=" * 80)

    if not todo:
        print("Nothing to do. All rows are already processed.")
        return

    session = requests.Session()
    started = time.time()
    processed = 0
    success = 0
    failed = 0
    batches = 0

    if args.mode == "single":
        for row in todo:
            out = run_single_request(
                session=session,
                api_base=args.api_base,
                image_dir=image_dir,
                row=row,
                timeout=args.timeout,
            )
            append_rows(output_csv, [out])
            processed += 1
            if out["success"]:
                success += 1
            else:
                failed += 1

            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(todo) - processed) / rate if rate > 0 else float("inf")
            if processed % 20 == 0 or processed == len(todo):
                print(
                    f"[{processed}/{len(todo)}] success={success} failed={failed} "
                    f"rate={rate:.3f} img/s eta={format_eta(eta)}"
                )
        return

    # mode=batch
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        rows_out = run_batch_request(
            session=session,
            api_base=args.api_base,
            image_dir=image_dir,
            batch=batch,
            timeout=args.timeout,
        )
        append_rows(output_csv, rows_out)

        batches += 1
        processed += len(batch)
        for r in rows_out:
            if r["success"]:
                success += 1
            else:
                failed += 1

        elapsed = time.time() - started
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(todo) - processed) / rate if rate > 0 else float("inf")
        if batches % 5 == 0 or processed >= len(todo):
            print(
                f"[batch {batches}] [{processed}/{len(todo)}] success={success} failed={failed} "
                f"rate={rate:.3f} img/s eta={format_eta(eta)}"
            )

    total_elapsed = time.time() - started
    print("=" * 80)
    print("DONE")
    print(f"processed : {processed}")
    print(f"success   : {success}")
    print(f"failed    : {failed}")
    print(f"elapsed   : {format_eta(total_elapsed)}")
    print(f"avg_rate  : {processed / total_elapsed:.3f} img/s")
    print(f"output    : {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

