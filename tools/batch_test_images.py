"""
Batch Test Script - Test toàn bộ ảnh với API Captioning

Usage:
    python tools/batch_test_images.py                    # Test full folder
    python tools/batch_test_images.py --limit 100        # Test 100 ảnh
    python tools/batch_test_images.py --random --limit 50  # Random 50 ảnh
    python tools/batch_test_images.py --sample            # Random 20 ảnh
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

# Config
API_URL = "http://localhost:8000/api/caption_full"
IMAGES_DIR = Path("data/images")
OUTPUT_DIR = Path("outputs/batch_test")

# Mặc định API timeout (giây)
API_TIMEOUT = 30


def load_image_as_base64(image_path: Path) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        import base64
        return base64.b64encode(f.read()).decode("utf-8")


def test_image_via_api(image_path: Path, api_url: str = API_URL) -> dict:
    """
    Gửi ảnh lên API và nhận kết quả caption.
    
    Hỗ trợ 2 cách gửi:
    1. file:// (local file) - dùng multipart form
    2. URL - dùng JSON body
    """
    result = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "success": False,
        "error": None,
        "caption_vi": None,
        "caption_vi_no_accent": None,
        "accent_restored": None,
        "device": None,
        "processing_time": None,
    }
    
    start_time = time.time()
    
    try:
        # Đọc ảnh và gửi qua multipart form
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(
                api_url,
                files=files,
                timeout=API_TIMEOUT
            )
        
        elapsed = time.time() - start_time
        result["processing_time"] = round(elapsed, 2)
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = data.get("success", False)
            result["caption_vi"] = data.get("caption_vi")
            result["caption_vi_no_accent"] = data.get("caption_vi_no_accent")
            result["accent_restored"] = data.get("accent_restored")
            result["device"] = data.get("device")
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        result["error"] = f"API timeout after {API_TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        result["error"] = "Cannot connect to API. Is server running?"
    except Exception as e:
        result["error"] = str(e)[:200]
    
    return result


def get_all_images(images_dir: Path) -> list:
    """Lấy danh sách tất cả ảnh trong thư mục."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    images = []
    
    if not images_dir.exists():
        print(f"❌ Directory not found: {images_dir}")
        return images
    
    for ext in image_extensions:
        images.extend(images_dir.glob(f"*{ext}"))
        images.extend(images_dir.glob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def run_batch_test(
    images_dir: Path = IMAGES_DIR,
    output_dir: Path = OUTPUT_DIR,
    limit: Optional[int] = None,
    random_sample: bool = False,
    api_url: str = API_URL,
    save_json: bool = True,
    save_csv: bool = True,
) -> dict:
    """
    Chạy batch test toàn bộ ảnh.
    
    Args:
        images_dir: Thư mục chứa ảnh
        output_dir: Thư mục lưu kết quả
        limit: Giới hạn số ảnh test (None = full)
        random_sample: Chọn ngẫu nhiên thay vì theo thứ tự
        api_url: URL của API
        save_json: Lưu kết quả ra JSON
        save_csv: Lưu kết quả ra CSV
    
    Returns:
        Dictionary chứa kết quả và thống kê
    """
    import random
    
    print("=" * 60)
    print("BATCH TEST - Image Captioning API")
    print("=" * 60)
    print(f"📁 Images directory: {images_dir}")
    print(f"🔗 API URL: {api_url}")
    print(f"📊 Mode: {'Random sample' if random_sample else 'Full batch'}")
    if limit:
        print(f"📊 Limit: {limit} images")
    print()
    
    # Tạo output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh
    all_images = get_all_images(images_dir)
    
    if not all_images:
        print("❌ No images found!")
        return {"results": [], "stats": {}}
    
    print(f"✅ Found {len(all_images)} images")
    
    # Chọn ảnh để test
    images_to_test = all_images.copy()
    
    if random_sample:
        random.seed(42)  # Reproducible
        random.shuffle(images_to_test)
    
    if limit:
        images_to_test = images_to_test[:limit]
    
    print(f"📝 Will test {len(images_to_test)} images")
    print()
    
    # Chạy test
    results = []
    errors = []
    
    print("🚀 Starting batch test...")
    print()
    
    for img_path in tqdm(images_to_test, desc="Testing images", unit="img"):
        result = test_image_via_api(img_path, api_url)
        results.append(result)
        
        if result["error"]:
            errors.append({
                "image": result["image_name"],
                "error": result["error"]
            })
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Thống kê
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    error_count = len(errors)
    
    # Phân loại kết quả
    good_captions = []      # Caption tốt, có nghĩa
    medium_captions = []    # Caption gần đúng nhưng có lỗi
    bad_captions = []       # Caption sai / vô nghĩa
    no_caption = []        # Không có caption
    
    for r in results:
        if not r["success"] or not r["caption_vi"]:
            no_caption.append(r)
        else:
            caption = r["caption_vi"].lower()
            # Heuristics đơn giản để phân loại
            bad_keywords = ["unused", "[unk]", "error", "null", "none"]
            has_bad = any(kw in caption for kw in bad_keywords)
            has_repeat = len(set(caption.split())) < len(caption.split()) * 0.3  # Quá nhiều từ lặp
            is_too_short = len(caption.split()) < 3
            
            if has_bad or has_repeat or is_too_short:
                bad_captions.append(r)
            elif len(caption.split()) < 5:
                medium_captions.append(r)
            else:
                good_captions.append(r)
    
    stats = {
        "total_images": total,
        "success": success,
        "failed": failed,
        "error_count": error_count,
        "good_captions": len(good_captions),
        "medium_captions": len(medium_captions),
        "bad_captions": len(bad_captions),
        "no_caption": len(no_caption),
        "success_rate": f"{success/total*100:.1f}%" if total > 0 else "0%",
    }
    
    # In thống kê
    print(f"\n📊 Overall Statistics:")
    print(f"   Total:     {total}")
    print(f"   Success:   {success} ({stats['success_rate']})")
    print(f"   Failed:    {failed}")
    print(f"   Errors:    {error_count}")
    
    print(f"\n📊 Caption Quality (heuristic-based):")
    print(f"   ✅ Good:    {len(good_captions)} ({len(good_captions)/total*100:.1f}%)" if total > 0 else "   ✅ Good: 0 (0%)")
    print(f"   ⚠️ Medium:  {len(medium_captions)} ({len(medium_captions)/total*100:.1f}%)" if total > 0 else "   ⚠️ Medium: 0 (0%)")
    print(f"   ❌ Bad:     {len(bad_captions)} ({len(bad_captions)/total*100:.1f}%)" if total > 0 else "   ❌ Bad: 0 (0%)")
    
    # Thời gian xử lý
    processing_times = [r["processing_time"] for r in results if r["processing_time"]]
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        print(f"\n⏱️  Processing Time:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Min:     {min_time:.2f}s")
        print(f"   Max:     {max_time:.2f}s")
    
    # In sample kết quả tốt/xấu
    print(f"\n📝 Sample GOOD captions:")
    for r in good_captions[:3]:
        print(f"   [{r['image_name']}] {r['caption_vi'][:60]}...")
    
    print(f"\n📝 Sample BAD captions:")
    for r in bad_captions[:3]:
        print(f"   [{r['image_name']}] {r['caption_vi'] or r.get('error', 'N/A')}")
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_json:
        json_path = output_dir / f"batch_test_results_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "images_dir": str(images_dir),
                "api_url": api_url,
                "total_images": total,
                "stats": stats,
            },
            "results": results,
            "errors": errors,
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 JSON saved: {json_path}")
    
    if save_csv:
        csv_path = output_dir / f"batch_test_summary_{timestamp}.csv"
        
        import csv
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "caption_vi",
                "caption_vi_no_accent",
                "success",
                "accent_restored",
                "device",
                "processing_time",
                "error"
            ])
            
            for r in results:
                writer.writerow([
                    r["image_name"],
                    r["caption_vi"] or "",
                    r["caption_vi_no_accent"] or "",
                    r["success"],
                    r["accent_restored"] or "",
                    r["device"] or "",
                    r["processing_time"] or "",
                    r["error"] or "",
                ])
        
        print(f"💾 CSV saved: {csv_path}")
    
    # In đường dẫn output
    print(f"\n📂 Output directory: {output_dir}")
    
    return {
        "results": results,
        "stats": stats,
        "json_path": json_path if save_json else None,
        "csv_path": csv_path if save_csv else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch test image captioning API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/batch_test_images.py                    # Test full folder
  python tools/batch_test_images.py --limit 50         # Test 50 images
  python tools/batch_test_images.py --random --limit 20  # Random 20 images
  python tools/batch_test_images.py --no-csv            # Skip CSV output
        """
    )
    
    parser.add_argument(
        "--images-dir", "-i",
        type=Path,
        default=IMAGES_DIR,
        help=f"Directory containing images (default: {IMAGES_DIR})"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--api-url", "-a",
        type=str,
        default=API_URL,
        help=f"API URL (default: {API_URL})"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of images to test"
    )
    
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Random sample instead of sequential"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output"
    )
    
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra API có đang chạy không
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code != 200:
            print("⚠️  API health check failed, but will continue...")
    except:
        print("⚠️  Warning: Cannot connect to API at http://localhost:8000")
        print("   Make sure API server is running!")
        print("   Start with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print()
    
    run_batch_test(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        random_sample=args.random,
        api_url=args.api_url,
        save_json=not args.no_json,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
