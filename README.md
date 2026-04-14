# CHUYEN_DE_Backend

FastAPI backend cho bài toán sinh caption sản phẩm tiếng Việt:
- BLIP sinh caption không dấu
- Accent restoration thêm dấu tiếng Việt
- Hỗ trợ single/batch inference, health check, cache

## 1. Cài đặt

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Chạy server

```bash
source .venv/bin/activate
export DEVICE=mps
export MODEL_PATH=/Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend/models/blip_vietnamese_80_20
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3. Endpoints chính

- `POST /api/caption`: caption không dấu (1 ảnh)
- `POST /api/caption/batch`: caption không dấu (n ảnh)
- `POST /api/caption_full`: caption có dấu (1 ảnh)
- `POST /api/caption_full/batch`: caption có dấu (n ảnh)
- `POST /api/accent/restore`: test restore dấu cho text
- `GET /api/health`: trạng thái model/cache
- `POST /api/cache/clear`: xóa cache

## 4. Cấu trúc chính

- `app/main.py`: khởi tạo FastAPI, middleware, router
- `app/api/routes_caption.py`: định nghĩa endpoints
- `app/services/caption_service.py`: nghiệp vụ caption
- `app/core/model_loader.py`: load BLIP model
- `app/core/accent_restoration_loader.py`: load accent model
- `app/core/config.py`: config qua biến môi trường
- `tools/`: train/eval/inference scripts

## 5. Ghi chú

- File cấu hình mẫu: `.env.example`
- Để demo Postman: dùng `form-data`, key `file`, type `File` cho endpoints caption

## 6. Bộ Thực Nghiệm Chuẩn (cho báo cáo/khóa luận)

### 6.1 Ma trận thực nghiệm

- File kế hoạch: `configs/experiment_matrix.csv`
- Mỗi dòng là 1 EXP, có cột params + metrics + trạng thái (`PLANNED/RUNNING/DONE/FAILED`)

### 6.2 Chạy đánh giá metrics

```bash
python tools/evaluate_test_20.py \
  --predictions outputs/predictions_test.csv \
  --ground-truth data/test_20.csv \
  --create-comparison
```

Kết quả:
- `outputs/evaluation_metrics.csv`
- `outputs/comparison_test_20.csv`

### 6.3 Ghi kết quả của một EXP vào bảng tổng hợp

```bash
python tools/record_experiment_result.py \
  --exp-id EXP-01-BASELINE \
  --metrics-csv outputs/evaluation_metrics.csv \
  --objective "Baseline decode default" \
  --model-path models/blip_vietnamese_80_20 \
  --num-beams 3 \
  --no-repeat-ngram-size 3 \
  --repetition-penalty 1.2 \
  --use-accent true \
  --notes "First official baseline"
```

Kết quả:
- `outputs/experiment_results.csv` (upsert theo `exp_id`)
- cập nhật lại `configs/experiment_matrix.csv`

### 6.4 Xuất bảng xếp hạng và báo cáo markdown

```bash
python tools/export_experiment_report.py \
  --results-csv outputs/experiment_results.csv
```

Kết quả:
- `outputs/experiment_summary_latest.csv`
- `outputs/experiment_summary_latest.md`

### 6.5 Template phân tích lỗi

- `docs/ERROR_ANALYSIS_TEMPLATE.md`
- `outputs/error_analysis_sheet_template.csv`

Quy trình khuyến nghị mỗi vòng:
1. Chạy inference + evaluate.
2. Ghi kết quả bằng `record_experiment_result.py`.
3. Xuất báo cáo bằng `export_experiment_report.py`.
4. Điền phân tích lỗi theo template.

## 7. Train Model Mới Từ Shopee 34k (Không Ảnh Hưởng Model Cũ)

Mục tiêu: tạo một model mới hoàn toàn tách biệt với `models/blip_vietnamese_80_20`.

### 7.0 Kết quả thực nghiệm (Model baseline)

**Bảng metrics trên tập kiểm thử (1514 samples):**

| Metric | Giá trị | Mô tả |
|--------|---------|--------|
| BLEU-1 | 0.0896 | Tỷ lệ unigram trùng khớp |
| BLEU-2 | 0.0581 | Tỷ lệ bigram trùng khớp |
| BLEU-3 | 0.0403 | Tỷ lệ trigram trùng khớp |
| BLEU-4 | 0.0315 | Tỷ lệ 4-gram trùng khớp |
| ROUGE-L | 0.1486 | Độ phủ longest common subsequence |
| SBERT | 0.6330 | Độ tương đồng ngữ nghĩa |

**Phân tích:**

Kết quả BLEU-1 đến BLEU-4 đều ở mức thấp và giảm dần theo bậc n-gram. Điều này là phù hợp với đặc thù bài toán vì caption tham chiếu từ Shopee thường dài, mang tính SEO và chứa nhiều từ khóa marketing, trong khi caption do mô hình sinh ra ngắn gọn, tập trung vào nội dung chính của sản phẩm. Do đó, các metric ngữ nghĩa như SBERT phản ánh chất lượng mô hình phù hợp hơn. Với SBERT similarity đạt 0.6330, mô hình cho thấy khả năng nắm bắt đúng ý nghĩa sản phẩm ở mức khá tốt.

**Thí nghiệm bổ sung:**

Nhóm có thử nghiệm thêm với cấu hình generation khác và một phiên bản model khác (blip_vietnamese_vi_v2), tuy nhiên kết quả BLEU trên tập kiểm thử không cải thiện so với baseline. Vì vậy nhóm chọn baseline làm mô hình chính.

### 7.1 Chuẩn bị split mới từ bộ Shopee gốc

```bash
python tools/prepare_shopee34k_split.py \
  --input-csv "/Users/nguyenhuuviet/Downloads/shopee-product-matching 2/train.csv" \
  --image-dir "/Users/nguyenhuuviet/Downloads/shopee-product-matching 2/train_images" \
  --output-full-csv data/shopee34250_full_clean.csv \
  --output-train-csv data/shopee34250_train_80.csv \
  --output-test-csv data/shopee34250_test_20.csv \
  --text-source-column title \
  --target-caption-column caption_target \
  --train-ratio 0.8 \
  --seed 42
```

### 7.2 Train model mới (isolated output path)

```bash
TRAIN_CSV_PATH=data/shopee34250_train_80.csv \
VAL_CSV_PATH=data/shopee34250_test_20.csv \
BASE_DATASET_CSV=data/shopee34250_full_clean.csv \
IMAGE_DIR="/Users/nguyenhuuviet/Downloads/shopee-product-matching 2/train_images" \
CAPTION_COLUMN=caption_target \
MODEL_OUTPUT_DIR=models/blip_vietnamese_shopee34250_v1 \
python train/train_blip_vietnamese.py
```

Kết quả:
- Model mới nằm ở `models/blip_vietnamese_shopee34250_v1`
- Model cũ `models/blip_vietnamese_80_20` giữ nguyên để làm baseline.

### 7.3 Chạy sanity check trước khi train full (khuyến nghị)

```bash
TRAIN_CSV_PATH=data/shopee34250_train_80.csv \
VAL_CSV_PATH=data/shopee34250_test_20.csv \
BASE_DATASET_CSV=data/shopee34250_full_clean.csv \
IMAGE_DIR="/Users/nguyenhuuviet/Downloads/shopee-product-matching 2/train_images" \
CAPTION_COLUMN=caption_target \
MODEL_OUTPUT_DIR=models/blip_vietnamese_shopee34250_sanity \
NUM_EPOCHS=1 \
MAX_TRAIN_SAMPLES=2000 \
MAX_VAL_SAMPLES=400 \
python train/train_blip_vietnamese.py
```

### 7.4 Preset M1 Pro/Max 64GB (ưu tiên tốc độ + an toàn checkpoint)

```bash
TRAIN_CSV_PATH=data/shopee34250_train_80.csv \
VAL_CSV_PATH=data/shopee34250_test_20.csv \
BASE_DATASET_CSV=data/shopee34250_full_clean.csv \
IMAGE_DIR="/Users/nguyenhuuviet/Downloads/shopee-product-matching 2/train_images" \
CAPTION_COLUMN=caption_target \
MODEL_OUTPUT_DIR=models/blip_vietnamese_shopee34250_v1 \
NUM_EPOCHS=5 \
TRAIN_BATCH_SIZE=8 \
EVAL_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
MAP_BATCH_SIZE=32 \
SAVE_STRATEGY=steps \
SAVE_STEPS=200 \
EVAL_STRATEGY=steps \
EVAL_STEPS=200 \
SAVE_TOTAL_LIMIT=6 \
AUTO_RESUME=true \
OVERWRITE_OUTPUT_DIR=false \
python train/train_blip_vietnamese.py
```

Ghi chú:
- Nếu máy báo hết bộ nhớ MPS: giảm `TRAIN_BATCH_SIZE` từ 8 xuống 6 rồi 4.
- Nếu bị ngắt điện/crash: chạy lại đúng lệnh trên, script sẽ tự tìm checkpoint mới nhất và resume.
- Script có "language gate" mặc định (`REQUIRE_VI_TARGET=true`): nếu cột caption không đủ tiếng Việt, script sẽ dừng để tránh train nhầm model không phải tiếng Việt.
