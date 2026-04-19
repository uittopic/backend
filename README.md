# 🖼️ BLIP Vietnamese Product Captioning — Chuyên Đề Backend

> **Mô tả**: Fine-tune BLIP (Bootstrapped Language-Image Pre-training) cho bài toán Image Captioning tiếng Việt trên ảnh sản phẩm e-commerce. Pipeline gồm 2 stage: sinh caption không dấu bằng BLIP → khôi phục dấu tiếng Việt bằng XLM-RoBERTa.

---

## 📁 Cấu Trúc Project

```
CHUYEN_DE_Backend/
├── app/                          # FastAPI backend
│   ├── api/
│   │   └── routes_caption.py     # API endpoints /api/caption, /api/caption_full
│   ├── core/
│   │   ├── config.py            # Cấu hình (device, model paths, generation params)
│   │   ├── model_loader.py      # Load BLIP model (caption không dấu)
│   │   └── accent_restoration_loader.py  # Load XLM-R (khôi phục dấu)
│   ├── services/
│   │   └── caption_service.py   # Logic sinh caption, cache, inference
│   ├── middleware/
│   │   ├── auth.py              # Authentication (API key)
│   │   └── rate_limit.py        # Rate limiting
│   ├── utils/
│   │   └── cache.py             # In-memory cache
│   └── main.py                  # FastAPI app entry point
│
├── train/                        # Training scripts
│   ├── train_blip_vietnamese.py  # Fine-tune BLIP với HuggingFace Trainer
│   ├── train_cleaned_v1.py       # Fine-tune với cleaned data + lazy loading
│   └── train_optimized.py        # ⚡ Phiên bản TỐI ƯU (8 epoch, cosine LR, augmentation)
│
├── tools/                        # Utility scripts
│   ├── clean_data_v2.py         # Data cleaning v2 (loại bỏ SEO spam)
│   ├── clean_data_v3.py         # Data cleaning v3 (thông minh, giữ thông tin sản phẩm)
│   ├── eval_model.py            # Đánh giá model trên test set
│   ├── eval_detailed.py         # Đánh giá chi tiết (BLEU, ROUGE-L)
│   ├── eval_compare_params.py    # So sánh inference params
│   ├── plot_diagrams.py         # Sinh 5 figures cho báo cáo
│   ├── plot_tables.py           # Sinh bảng so sánh pipeline
│   ├── plot_metrics.py          # Vẽ biểu đồ metrics
│   ├── export_experiment_report.py # Export báo cáo experiment
│   └── split_train_test.py      # Chia train/test split
│
├── configs/
│   └── infer.yaml               # Cấu hình inference
│
├── models/                       # Trained models (gitignore)
│   ├── blip_vietnamese_80_20/  # Model train 80/20 split
│   ├── blip_vietnamese_cleaned_v1/  # Model train với cleaned data v1
│   └── accent_restoration/      # Accent restoration model
│
├── data/                        # Datasets
│   ├── images/                  # Ảnh sản phẩm
│   ├── train_80.csv             # Train set (80%)
│   ├── test_20.csv              # Test set (20%)
│   ├── train_80_cleaned.csv     # Train set đã clean v1
│   └── train_80_cleaned_v2.csv # Train set đã clean v2
│
├── outputs/                      # Kết quả đánh giá, hình ảnh
│   ├── fig1_problem.png         # Hình 1: Định nghĩa bài toán
│   ├── fig2_pipeline_simple.png  # Hình 2: Pipeline tổng quan
│   ├── fig3_pipeline_detail.png # Hình 3: Pipeline chi tiết 2 stages
│   ├── fig4_blip_architecture.png # Hình 4: Kiến trúc BLIP (ViT + Transformer)
│   ├── fig5_accent_restoration.png # Hình 5: Accent Restoration
│   └── table1_pipeline_summary.png # Bảng 1: Tổng kết pipeline
│
├── logs/                        # Training logs (gitignore)
├── .gitignore
├── .env.example
├── demo_streamlit.py            # Giao diện Streamlit demo
└── run_full_pipeline.py        # Chạy full pipeline (train → eval → export)
```

---

## 🏗️ Pipeline Tổng Quan

```
INPUT IMAGE
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: BLIP — Image Captioning               │
│  Model: Salesforce/blip-image-captioning-base    │
│  Fine-tune trên ảnh sản phẩm tiếng Việt        │
│  Output: Caption KHÔNG DẤU                      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: Accent Restoration                   │
│  Model: peterhung/vietnamese-accent-marker-    │
│         xlm-roberta                            │
│  Token Classification: thêm dấu tiếng Việt     │
│  Output: Caption CÓ DẤU ✓                      │
└─────────────────────────────────────────────────┘
```

---

## 🧠 Mô Hình Chi Tiết

### Stage 1: BLIP (Bootstrapped Language-Image Pre-training)

**Base model**: `Salesforce/blip-image-captioning-base`

**Kiến trúc bên trong**:

```
INPUT IMAGE
    │
    ▼
Vision Encoder (ViT — Vision Transformer)
    │  Patch embedding + Transformer encoder layers
    ▼
Image Features (197 tokens × 768 dims)
    │
    ▼
Text Decoder (Transformer Decoder)
    │  Cross-attention với image features
    ▼
Generated Caption (tiếng Việt KHÔNG DẤU)
```

**Pretrained weights**: ImageNet + COCO Captioning (English) → **Fine-tune** trên data tiếng Việt để học ngôn ngữ mới.

**Fine-tuning params** (xem `train/train_optimized.py`):
- Epochs: 8
- Learning rate: 3e-5 (cosine scheduler)
- Batch size: 4 (gradient accumulation ×4 = 16)
- Max length: 128 tokens
- Augmentation: RandomHorizontalFlip + ColorJitter

### Stage 2: Accent Restoration (XLM-RoBERTa)

**Model**: `peterhung/vietnamese-accent-marker-xlm-roberta`

**Phương pháp**: Token Classification
- Mỗi word được gán label (ví dụ: `ao → áo`, `ma → mà`)
- Áp dụng rule-based: thay thế raw vowel → accented vowel

**Ưu điểm**:
- Nhẹ, nhanh (chỉ classify tokens)
- Không cần train lại — dùng pretrained model

---

## 📊 Dữ Liệu

### Nguồn dữ liệu
- Dataset e-commerce tiếng Việt (Shopee/Tiki/Lazada)
- ~7,400 ảnh sản phẩm

### Data Cleaning (v2)
1. Loại bỏ SEO spam keywords: `sale`, `hot`, `free`, `giá rẻ`, `bán chạy`...
2. Loại bỏ product codes: `ABC-123`, `LT1535`, `BPOM`...
3. Chuẩn hóa category keywords: `sendal → sandal`, `celana → quần`...
4. Loại bỏ English stopwords nhưng giữ product-specific English: `shirt`, `dress`, `gold`...
5. Loại bỏ repetition liên tiếp: `áo áo áo → áo`
6. Filter captions quá ngắn (< 8 words) hoặc quá nhiều English (> 35%)

### Data Cleaning (v3 — THÔNG MINH)
- Giữ nguyên thông tin sản phẩm có giá trị: **size, màu sắc, chất liệu, hoa văn**
- Chỉ loại bỏ SEO spam rõ ràng
- Không xóa English product words
- min_len: 6 words

---

## 🚀 Cách Chạy

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models (tự động khi chạy app)

```bash
# Models được download từ HuggingFace khi chạy lần đầu
# BLIP: Salesforce/blip-image-captioning-base
# Accent: peterhung/vietnamese-accent-marker-xlm-roberta
```

### 3. Training Model Mới (Optional)

```bash
# Phiên bản tối ưu (8 epoch, cosine LR, augmentation)
python train/train_optimized.py

# Hoặc version đơn giản
python train/train_blip_vietnamese.py
```

**Training parameters** (env variables):
```bash
NUM_EPOCHS=8 \
LEARNING_RATE=3e-5 \
TRAIN_BATCH_SIZE=4 \
MAX_LENGTH=128 \
python train/train_optimized.py
```

### 4. Cleaning Data

```bash
# v2 (aggressive cleaning)
python tools/clean_data_v2.py \
  --input data/train_80.csv \
  --output data/train_80_cleaned_v2.csv

# v3 (thông minh, giữ thông tin sản phẩm)
python tools/clean_data_v3.py \
  --input data/train_80_cleaned.csv \
  --output data/train_80_cleaned_v3.csv
```

### 5. Đánh Giá Model

```bash
# Đánh giá nhanh
python tools/eval_model.py --max-samples 100

# So sánh 2 inference configs
python tools/eval_compare_params.py --max-samples 50

# Đánh giá chi tiết (BLEU-1/2/3/4, ROUGE-L, human eval samples)
python tools/eval_detailed.py
```

### 6. Sinh Figures cho Báo Cáo

```bash
python tools/plot_diagrams.py
# Output: outputs/fig1-5.png
```

### 7. Chạy API Server

```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Hoặc dùng Streamlit demo
streamlit run demo_streamlit.py
```

### 8. Inference qua API

```bash
# Caption không dấu
curl -X POST "http://localhost:8000/api/caption" \
  -F "image=@test.jpg"

# Caption có dấu (full pipeline)
curl -X POST "http://localhost:8000/api/caption_full" \
  -F "image=@test.jpg"
```

---

## ⚙️ Cấu Hình Inference

Chỉnh sửa `configs/infer.yaml` hoặc `app/core/config.py`:

```yaml
generation:
  max_new_tokens: 128      # Độ dài caption tối đa (60→128: caption dài hơn)
  num_beams: 5            # Beam search width (3→5: chất lượng cao hơn)
  repetition_penalty: 1.1  # Giảm từ lặp (1.05→1.1)
  length_penalty: 1.2     # Ưu tiên câu dài (1.1→1.2: BLEU cao hơn)
  no_repeat_ngram_size: 3 # Không lặp 3-gram
  top_k: 50               # Top-k sampling
  top_p: 0.95             # Nucleus sampling
  do_sample: false        # false = greedy (ổn định nhất)
```

---

## 📈 Inference Params Tối Ưu

| Tham số | Baseline | Tối ưu | Ảnh hưởng |
|---------|----------|---------|------------|
| `max_new_tokens` | 60 | 128 | Caption dài hơn, đầy đủ hơn |
| `num_beams` | 3 | 5 | Beam search rộng hơn, chất lượng cao hơn |
| `repetition_penalty` | 1.05 | 1.1 | Giảm từ lặp |
| `length_penalty` | 1.1 | 1.2 | Ưu tiên câu dài → BLEU cao hơn |
| `top_k` | 0 | 50 | Sampling đa dạng |
| `top_p` | — | 0.95 | Nucleus sampling |

---

## 📊 Kết Quả Đánh Giá

### Trên Test Set (20% — 1,530 mẫu)

| Metric | Giá trị | Ghi chú |
|--------|---------|---------|
| BLEU-1 | ~0.04 | Cần cải thiện |
| BLEU-4 | ~0.00 | Do model generate English + data noisy |
| ROUGE-L | ~0.05 | Cần cải thiện |

### Vấn đề hiện tại

1. **73% English output** — model pretrained English chiếm ưu thế
   - Nguyên nhân: Fine-tuning chưa đủ epoch + data noisy
   - Giải pháp: Train lại với `train_optimized.py` (8 epoch)

2. **BLEU thấp** — ground truth không đồng nhất
   - Nguyên nhân: Dataset e-commerce có nhiều nhiễu
   - Giải pháp: Dùng `clean_data_v3.py` để clean thông minh hơn

### Hướng cải thiện

1. **Train lại với params tối ưu**:
   ```bash
   python train/train_optimized.py
   ```
2. **Clean data với v3**:
   ```bash
   python tools/clean_data_v3.py --input data/train_80.csv --output data/train_80_cleaned_v3.csv
   ```
3. **Tăng training data** — thu thập thêm ảnh sản phẩm tiếng Việt

---

## 🛠️ Development

### Device Support

| Device | Support | Ghi chú |
|--------|---------|---------|
| Apple Silicon (MPS) | ✅ | Tự động detect, Mac M1/M2/M3 |
| NVIDIA GPU (CUDA) | ✅ | Cần `torch` với CUDA |
| CPU | ✅ | Chậm, chỉ dùng để dev/test |

### Environment Variables

```bash
# Model paths
MODEL_PATH=models/blip_vietnamese_cleaned_v1
ACCENT_MODEL_NAME=peterhung/vietnamese-accent-marker-xlm-roberta

# Training
NUM_EPOCHS=8
LEARNING_RATE=3e-5
TRAIN_BATCH_SIZE=4
MAX_LENGTH=128

# Inference
MAX_NEW_TOKENS=128
NUM_BEAMS=5
REPETITION_PENALTY=1.1
LENGTH_PENALTY=1.2

# Device override (cpu / cuda / mps)
DEVICE=mps
```

### API Authentication

```bash
# Bật auth bằng cách set API_KEYS trong .env
ENABLE_AUTH=true
API_KEYS=your-secret-key-1,your-secret-key-2
```

```bash
# Gọi API với API key
curl -H "X-API-Key: your-secret-key-1" \
  -X POST "http://localhost:8000/api/caption_full" \
  -F "image=@test.jpg"
```

---

## 📝 Các File Quan Trọng cho Đồng Đội

| File | Mô tả |
|------|-------|
| `train/train_optimized.py` | Training script TỐI ƯU NHẤT — dùng cái này để train |
| `tools/clean_data_v3.py` | Data cleaning thông minh — giữ thông tin sản phẩm |
| `tools/eval_model.py` | Đánh giá nhanh — xem metrics + sample predictions |
| `tools/plot_diagrams.py` | Sinh 5 figures cho báo cáo |
| `app/core/config.py` | Tất cả cấu hình inference (generation params) |
| `app/core/accent_restoration_loader.py` | Accent restoration — xử lý subword token |

---

## 🔧 Troubleshooting

### Lỗi `MPS not available`
```bash
# Trên Mac Intel hoặc Linux không có GPU
DEVICE=cpu python train/train_optimized.py
```

### Lỗi `CUDA out of memory`
```bash
# Giảm batch size
TRAIN_BATCH_SIZE=2 python train/train_optimized.py
# Hoặc bật gradient checkpointing (đã bật mặc định trong train_optimized.py)
```

### Model generate toàn tiếng Anh
1. Kiểm tra train data: `caption_vi_cleaned` column có dấu tiếng Việt không
2. Chạy lại training: `python train/train_optimized.py`
3. Tăng epoch: `NUM_EPOCHS=10 python train/train_optimized.py`

### Accent restoration không hoạt động
```bash
# Kiểm tra model đã load chưa
python -c "from app.core.accent_restoration_loader import accent_model; print(accent_model)"
```

---

## 📚 References

1. **BLIP**: Li et al., "BLIP: Bootstrapped Language-Image Pre-training for Unified Vision-Language Understanding and Generation", 2022
2. **XLM-RoBERTa**: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", 2020
3. **Vietnamese Accent Marker**: peterhung/vietnamese-accent-marker-xlm-roberta (HuggingFace)

---

## 👥 Team

- **Backend/ML**: Nguyễn Huy Việt
- **Framework**: FastAPI + Streamlit
- **Models**: BLIP + XLM-RoBERTa
- **Device**: Apple Silicon (MPS) / NVIDIA GPU (CUDA) / CPU

---

*Lần cuối cập nhật: 2026-04-16*
