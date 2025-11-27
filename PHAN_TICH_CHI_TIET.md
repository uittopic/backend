# 📊 PHÂN TÍCH CHI TIẾT DỰ ÁN - BÁO CÁO CHUYÊN ĐỀ

## 🎯 TỔNG QUAN DỰ ÁN

### 1.1. Mục tiêu dự án
- **Bài toán**: Tạo caption tiếng Việt có dấu, tự nhiên cho ảnh sản phẩm từ Shopee
- **Giải pháp**: Fine-tune BLIP model + Accent Restoration pipeline
- **Ứng dụng**: Hỗ trợ tra cứu và tìm kiếm sản phẩm

### 1.2. Kiến trúc tổng thể
```
┌─────────────┐
│   Ảnh Input │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  BLIP Model         │ ← Fine-tuned trên 80% dataset
│  (Không dấu)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Accent Restoration │ ← peterhung/vietnamese-accent-marker
│  (Có dấu)           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Caption Tiếng Việt │
│  Có dấu             │
└─────────────────────┘
```

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### 2.1. Cấu trúc thư mục

```
backend/
├── app/                          # Application chính
│   ├── main.py                   # FastAPI entry point
│   ├── api/                      # API Routes
│   │   └── routes_caption.py     # Endpoints cho caption generation
│   ├── core/                     # Core modules
│   │   ├── config.py             # Configuration management
│   │   ├── model_loader.py       # BLIP model loader
│   │   └── accent_restoration_loader.py  # Accent model loader
│   ├── services/                 # Business logic
│   │   └── caption_service.py    # Caption generation service
│   ├── middleware/               # Middleware
│   │   └── auth.py               # Authentication (optional)
│   └── utils/                    # Utilities
│       ├── cache.py              # Caching system
│       └── rate_limit.py         # Rate limiting
│
├── train/                        # Training scripts
│   └── train_blip_vietnamese.py # Fine-tune BLIP model
│
├── tools/                        # Utility tools
│   ├── evaluate_metrics.py       # Đánh giá metrics (BLEU, ROUGE, SBERT)
│   ├── run_inference_full_test.py # Chạy inference trên test set
│   └── split_train_test.py       # Chia dataset 80/20
│
├── data/                         # Data directory
│   ├── train_bilingual_clean_v2.csv  # Dataset đã xử lý (7,638 samples)
│   └── images/                   # 7,443 ảnh sản phẩm
│
├── configs/                      # Configuration files
│   └── infer.yaml                # Inference configuration
│
└── requirements.txt              # Dependencies
```

### 2.2. Technology Stack

| Component | Technology | Version | Mục đích |
|-----------|-----------|---------|----------|
| **Framework** | FastAPI | 0.104.1 | REST API framework |
| **ML Framework** | PyTorch | 2.1.0 | Deep learning |
| **Model** | Transformers | 4.35.0 | BLIP, Accent Restoration |
| **Image Processing** | Pillow | 10.1.0 | Xử lý ảnh |
| **Data Processing** | Pandas | 2.1.3 | Xử lý CSV |
| **Caching** | In-memory | - | Cache kết quả |
| **Rate Limiting** | slowapi | 0.1.9 | Giới hạn request |

---

## 🔧 CÁC THÀNH PHẦN KỸ THUẬT CHI TIẾT

### 3.1. Model Loader (`app/core/model_loader.py`)

**Chức năng**: Load và quản lý BLIP model

**Đặc điểm**:
- Tự động detect device (MPS/CUDA/CPU)
- Load fine-tuned model từ local hoặc pretrained từ HuggingFace
- Tối ưu cho macOS M1/M2/M3 với MPS backend
- Model compilation với `torch.compile()` (nếu hỗ trợ)

**Code Flow**:
```python
1. Kiểm tra MODEL_PATH có tồn tại không
2. Nếu có → Load fine-tuned model
3. Nếu không → Load pretrained model (Salesforce/blip-image-captioning-base)
4. Chuyển model sang device (MPS/CUDA/CPU)
5. Set model.eval() cho inference
6. Compile model nếu PyTorch 2.0+ (tăng tốc)
```

**Tối ưu hóa**:
- Giữ float32 cho MPS (không dùng fp16)
- Synchronize device sau khi load
- Model compilation cho CUDA/CPU (không hỗ trợ MPS)

### 3.2. Accent Restoration Loader (`app/core/accent_restoration_loader.py`)

**Chức năng**: Load và sử dụng model phục hồi dấu tiếng Việt

**Model**: `peterhung/vietnamese-accent-marker-xlm-roberta`

**Pipeline**:
```
Text không dấu → Tokenize → Token Classification → Merge tokens → Text có dấu
```

**Đặc điểm**:
- Token Classification model (nhẹ, nhanh)
- Accuracy 97%+
- Xử lý subword tokens từ XLM-RoBERTa
- Merge tokens với prefix "▁" để tạo words

**Code Flow**:
```python
1. Load tokenizer và model từ HuggingFace
2. Load label list (selected_tags_names.txt)
3. Tokenize text với is_split_into_words=True
4. Predict labels cho mỗi token
5. Merge subword tokens thành words
6. Áp dụng accent labels để tạo text có dấu
```

### 3.3. Caption Service (`app/services/caption_service.py`)

**Chức năng**: Business logic cho caption generation

**Các hàm chính**:
- `generate_caption_for_image()`: Sinh caption không dấu
- `generate_caption_with_accent()`: Sinh caption có dấu (pipeline đầy đủ)
- `_run_blip()`: Chạy BLIP model
- `_prepare_image()`: Resize ảnh nếu quá lớn

**Tối ưu hóa Memory**:
- Resize ảnh nếu > 512px (giảm VRAM)
- Chuyển model về CPU khi generate trên MPS (tránh lỗi attention_mask)
- Cleanup tensors sau mỗi inference
- Move tensors về CPU trước khi delete

### 3.4. API Routes (`app/api/routes_caption.py`)

**Endpoints**:

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/caption` | POST | Sinh caption không dấu (1 ảnh) |
| `/api/caption/batch` | POST | Sinh caption không dấu (nhiều ảnh) |
| `/api/caption_full` | POST | Sinh caption có dấu (1 ảnh) |
| `/api/caption_full/batch` | POST | Sinh caption có dấu (nhiều ảnh) |
| `/api/accent/restore` | POST | Restore accent cho text |
| `/api/health` | GET | Health check |
| `/api/cache/clear` | POST | Xóa cache |

**Request/Response Format**:

**Single Caption (không dấu)**:
```json
Request: multipart/form-data
  - file: image.jpg

Response:
{
  "success": true,
  "caption_vi": "ao khoac the thao nu mau den",
  "device": "mps",
  "cached": false,
  "processing_time": 0.45
}
```

**Single Caption (có dấu)**:
```json
Response:
{
  "success": true,
  "caption_vi": "áo khoác thể thao nữ màu đen",
  "caption_vi_no_accent": "ao khoac the thao nu mau den",
  "accent_restored": true,
  "device": "mps",
  "cached": false,
  "processing_time": 0.78
}
```

### 3.5. Configuration (`app/core/config.py`)

**Các cấu hình chính**:

**Model Configuration**:
- `MODEL_PATH`: Đường dẫn model fine-tuned
- `PRETRAINED_MODEL`: Pretrained model từ HuggingFace
- `ACCENT_MODEL_NAME`: Accent restoration model

**Generation Configuration**:
- `MAX_NEW_TOKENS`: 50 (độ dài caption tối đa)
- `NUM_BEAMS`: 3 (beam search)
- `REPETITION_PENALTY`: 1.2 (tránh lặp từ)
- `NO_REPEAT_NGRAM_SIZE`: 3 (tránh lặp n-gram)

**API Configuration**:
- `MAX_BATCH_SIZE`: 10 (số ảnh tối đa mỗi batch)
- `ENABLE_CACHE`: True (bật cache)
- `CACHE_TTL`: 86400 (24 giờ)
- `ENABLE_RATE_LIMIT`: True
- `RATE_LIMIT_PER_MINUTE`: 60

**Device Configuration**:
- Tự động detect: MPS > CUDA > CPU
- Có thể override bằng biến môi trường `DEVICE`

### 3.6. Caching System (`app/utils/cache.py`)

**Chức năng**: Cache kết quả caption để tăng tốc độ

**Cơ chế**:
- Hash ảnh bằng MD5 để tạo cache key
- In-memory cache (có thể thay bằng Redis)
- TTL (Time To Live): 24 giờ mặc định
- Tự động xóa cache hết hạn

**Cache Key**: MD5 hash của ảnh (PNG format)

**Cache Entry**:
```python
{
  'caption': 'áo khoác thể thao nữ',
  'expires_at': 1234567890.0,
  'created_at': 1234560000.0
}
```

### 3.7. Training Script (`train/train_blip_vietnamese.py`)

**Chức năng**: Fine-tune BLIP model cho tiếng Việt

**Dataset Split**:
- Train: 80% (6,110 samples)
- Test: 20% (1,528 samples)
- Tự động tạo split nếu chưa có

**Training Parameters**:
- Epochs: 5
- Batch size: 2 (tối ưu cho M1)
- Learning rate: 5e-5
- Warmup steps: 500
- Max length: 77 tokens
- Evaluation strategy: epoch
- Save strategy: epoch

**Tối ưu cho macOS**:
- Không dùng fp16 (MPS chưa hỗ trợ tốt)
- `dataloader_num_workers=0` (tránh lỗi multiprocessing)
- Batch size nhỏ (2-4)

**Output**: Model được lưu tại `models/blip_vietnamese_80_20/`

### 3.8. Evaluation Script (`tools/evaluate_metrics.py`)

**Chức năng**: Đánh giá kết quả caption bằng các metrics

**Metrics**:
1. **BLEU Score**: N-gram overlap (0-1)
   - Dùng NLTK với smoothing
   - Tokenize đơn giản (split by space)

2. **ROUGE-L**: Longest Common Subsequence F1 (0-1)
   - Dùng thư viện `rouge-score`
   - Đánh giá độ tương đồng cấu trúc

3. **SBERT Similarity**: Semantic similarity (0-1)
   - Dùng `keepitreal/vietnamese-sbert`
   - Cosine similarity giữa embeddings
   - Normalize về [0, 1]

**Kết quả thực tế** (từ báo cáo):
- BLEU: 0.0141 (thấp - do caption gốc dài, SEO)
- ROUGE-L: 0.1486 (trung bình)
- SBERT: 0.6330 (khá tốt - hiểu nghĩa tốt)

---

## 🔄 DATA FLOW

### 4.1. Training Flow

```
1. Load dataset từ CSV (train_bilingual_clean_v2.csv)
   ↓
2. Chia train/test 80/20 (nếu chưa có)
   ↓
3. Preprocess: Load ảnh + caption
   ↓
4. Fine-tune BLIP với HuggingFace Trainer
   ↓
5. Save model vào models/blip_vietnamese_80_20/
```

### 4.2. Inference Flow (Single Image)

```
1. Client gửi ảnh → API endpoint
   ↓
2. Kiểm tra cache (hash ảnh)
   ├─ Có cache → Trả về ngay
   └─ Không có → Tiếp tục
   ↓
3. Preprocess ảnh:
   - Resize nếu > 512px
   - Convert RGB
   ↓
4. BLIP Generation:
   - Load ảnh vào processor
   - Generate caption không dấu
   - (MPS: chuyển model về CPU khi generate)
   ↓
5. Accent Restoration:
   - Tokenize text không dấu
   - Predict accent labels
   - Merge tokens → text có dấu
   ↓
6. Cache kết quả
   ↓
7. Trả về response
```

### 4.3. Batch Processing Flow

```
1. Client gửi nhiều ảnh (max 10)
   ↓
2. Xử lý từng ảnh:
   - Generate caption
   - Cleanup memory mỗi 5 ảnh
   ↓
3. Trả về danh sách kết quả
```

---

## 🎯 CÁC TÍNH NĂNG CHÍNH

### 5.1. Caption Generation
- ✅ Sinh caption tiếng Việt không dấu (BLIP)
- ✅ Phục hồi dấu tiếng Việt (Accent Restoration)
- ✅ Batch processing (nhiều ảnh cùng lúc)
- ✅ Cache kết quả để tăng tốc

### 5.2. API Features
- ✅ REST API với FastAPI
- ✅ Swagger UI documentation (`/docs`)
- ✅ Health check endpoint
- ✅ Rate limiting
- ✅ CORS support
- ✅ Error handling

### 5.3. Optimization
- ✅ Tự động detect device (MPS/CUDA/CPU)
- ✅ Memory optimization cho macOS
- ✅ Image resizing để giảm VRAM
- ✅ Model compilation (PyTorch 2.0+)
- ✅ Batch cleanup để tránh memory leak

### 5.4. Evaluation
- ✅ Metrics: BLEU, ROUGE-L, SBERT
- ✅ Script đánh giá tự động
- ✅ Inference trên test set 20%

---

## 🐛 THÁCH THỨC KỸ THUẬT & GIẢI PHÁP

### 6.1. MPS (macOS) Compatibility

**Vấn đề**: MPS không hỗ trợ tốt attention_mask auto-inference cho BLIP

**Giải pháp**: Chuyển model về CPU khi generate, sau đó chuyển lại MPS
```python
if device == "mps":
    model_cpu = model.cpu()
    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
    output = model_cpu.generate(**inputs_cpu)
    model.to(device)  # Chuyển lại MPS
```

### 6.2. Memory Management

**Vấn đề**: Memory leak trên MPS khi xử lý nhiều ảnh

**Giải pháp**:
- Cleanup tensors sau mỗi inference
- Clear cache mỗi 5 ảnh trong batch
- Move tensors về CPU trước khi delete
- Synchronize device sau mỗi operation

### 6.3. Tokenizer Compatibility

**Vấn đề**: BLIP dùng BPE tokenizer tiếng Anh, không tương thích với tokenizer tiếng Việt

**Giải pháp**: Không thay tokenizer, dùng pipeline 2 giai đoạn:
1. BLIP sinh caption không dấu
2. Accent Restoration phục hồi dấu

### 6.4. BLEU Score Thấp

**Vấn đề**: BLEU score thấp (0.0141) do caption gốc dài, nhiều SEO keywords

**Giải pháp**: Dùng SBERT để đánh giá semantic similarity (0.6330) thay vì chỉ dựa vào BLEU

---

## 📈 ĐÁNH GIÁ KẾT QUẢ

### 7.1. Dataset
- **Tổng số mẫu**: 7,638
- **Train (80%)**: 6,110
- **Test (20%)**: 1,528

### 7.2. Model Performance

| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **BLEU** | 0.0141 | Thấp - do caption gốc dài, SEO |
| **ROUGE-L** | 0.1486 | Trung bình |
| **SBERT** | 0.6330 | **Khá tốt** - hiểu nghĩa tốt |

### 7.3. Inference Speed
- **Single image**: ~0.8-1.2s/ảnh
- **Batch (10 ảnh)**: ~5-8s

### 7.4. Accent Restoration
- **Accuracy**: 97%+
- **Model**: peterhung/vietnamese-accent-marker-xlm-roberta
- **Speed**: ~0.1-0.2s/text

---

## 💡 ĐIỂM MẠNH & ĐIỂM YẾU

### 8.1. Điểm mạnh
✅ Pipeline 2 giai đoạn hiệu quả (BLIP + Accent Restoration)
✅ Tối ưu cho macOS M1/M2/M3
✅ API hoàn chỉnh với caching, rate limiting
✅ Code structure rõ ràng, dễ maintain
✅ Evaluation metrics đầy đủ
✅ Batch processing support

### 8.2. Điểm yếu
⚠️ BLEU score thấp (do caption gốc dài, SEO)
⚠️ Inference time ~1s/ảnh (có thể tối ưu thêm)
⚠️ Tokenizer BLIP không hỗ trợ trực tiếp tiếng Việt
⚠️ Memory usage cao trên MPS (cần cleanup thường xuyên)

---

## 📝 GỢI Ý CHO BÁO CÁO CHUYÊN ĐỀ

### 9.1. Cấu trúc báo cáo đề xuất

1. **Chương 1: Giới thiệu**
   - Bài toán và mục tiêu
   - Phạm vi nghiên cứu
   - Cấu trúc báo cáo

2. **Chương 2: Cơ sở lý thuyết**
   - BLIP model architecture
   - Image Captioning
   - Accent Restoration
   - Evaluation Metrics (BLEU, ROUGE, SBERT)

3. **Chương 3: Phân tích và thiết kế hệ thống**
   - Kiến trúc tổng thể
   - Pipeline xử lý
   - API design
   - Database/Cache design

4. **Chương 4: Cài đặt và triển khai**
   - Dataset preparation
   - Model training
   - API implementation
   - Optimization techniques

5. **Chương 5: Đánh giá và kết quả**
   - Evaluation metrics
   - Kết quả thực nghiệm
   - So sánh với baseline
   - Phân tích kết quả

6. **Chương 6: Kết luận và hướng phát triển**
   - Tổng kết
   - Hạn chế
   - Hướng phát triển

### 9.2. Các diagram nên có

1. **System Architecture Diagram**
   - Tổng quan hệ thống
   - Components và interactions

2. **Data Flow Diagram**
   - Training flow
   - Inference flow

3. **API Sequence Diagram**
   - Request/Response flow
   - Cache mechanism

4. **Model Architecture Diagram**
   - BLIP architecture
   - Accent Restoration pipeline

### 9.3. Code snippets quan trọng

1. **Model Loading** (`model_loader.py`)
2. **Caption Generation** (`caption_service.py`)
3. **Accent Restoration** (`accent_restoration_loader.py`)
4. **API Endpoint** (`routes_caption.py`)
5. **Training Script** (`train_blip_vietnamese.py`)

### 9.4. Metrics và số liệu

- Dataset statistics (7,638 samples)
- Training time (~3-4 giờ trên M1 Pro Max)
- Inference speed (0.8-1.2s/ảnh)
- Evaluation metrics (BLEU, ROUGE-L, SBERT)
- Accent restoration accuracy (97%+)

### 9.5. Screenshots/Demo

- Swagger UI (`/docs`)
- API response examples
- Sample captions (before/after accent restoration)
- Evaluation results

---

## 🔍 CHI TIẾT KỸ THUẬT BỔ SUNG

### 10.1. BLIP Model Details

**Architecture**:
- Vision Encoder: ViT (Vision Transformer)
- Text Decoder: BERT-based
- Cross-modal attention

**Fine-tuning**:
- Base model: `Salesforce/blip-image-captioning-base`
- Training: 5 epochs, batch size 2
- Learning rate: 5e-5
- Output: Caption tiếng Việt không dấu

### 10.2. Accent Restoration Details

**Model**: `peterhung/vietnamese-accent-marker-xlm-roberta`

**Architecture**:
- Base: XLM-RoBERTa
- Task: Token Classification
- Labels: Format "raw-vowel" (ví dụ: "ao-áo")

**Process**:
1. Tokenize text không dấu
2. Predict label cho mỗi token
3. Merge subword tokens (prefix "▁")
4. Áp dụng accent labels
5. Join thành text có dấu

### 10.3. Caching Strategy

**Cache Key**: MD5 hash của ảnh (PNG format)

**Cache Storage**: In-memory dictionary
- Key: image_hash (MD5)
- Value: {caption, expires_at, created_at}

**TTL**: 24 giờ (86400 giây)

**Benefits**:
- Tăng tốc độ response (0.01s vs 0.8s)
- Giảm tải cho model
- Giảm memory usage (không cần regenerate)

### 10.4. Rate Limiting

**Implementation**: `slowapi` library

**Limits**:
- Per minute: 60 requests
- Per hour: 1000 requests

**Key Function**: `get_remote_address()` (IP-based)

**Error Response**: 429 Too Many Requests

---

## 📚 TÀI LIỆU THAM KHẢO KỸ THUẬT

1. **BLIP**: "BLIP: Bootstrapping Language-Image Pre-training" (Salesforce)
2. **Accent Restoration**: "Vietnamese Accent Marker" (peterhung)
3. **FastAPI**: FastAPI documentation
4. **PyTorch MPS**: PyTorch MPS backend documentation
5. **Evaluation Metrics**: 
   - BLEU: "BLEU: a Method for Automatic Evaluation of Machine Translation"
   - ROUGE: "ROUGE: A Package for Automatic Evaluation of Summaries"
   - SBERT: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

## ✅ CHECKLIST CHO BÁO CÁO

### Nội dung bắt buộc:
- [ ] Giới thiệu bài toán và mục tiêu
- [ ] Cơ sở lý thuyết (BLIP, Accent Restoration)
- [ ] Kiến trúc hệ thống (diagram)
- [ ] Pipeline xử lý (data flow)
- [ ] Dataset và preprocessing
- [ ] Training process và parameters
- [ ] API design và implementation
- [ ] Evaluation metrics và kết quả
- [ ] So sánh và phân tích
- [ ] Kết luận và hướng phát triển

### Code và demo:
- [ ] Code snippets quan trọng
- [ ] API documentation (Swagger)
- [ ] Sample results (captions)
- [ ] Evaluation results (metrics table)

### Diagrams:
- [ ] System architecture
- [ ] Data flow
- [ ] API sequence
- [ ] Model architecture

---

**Tài liệu này cung cấp phân tích chi tiết về dự án để hỗ trợ viết báo cáo chuyên đề. Các thông tin có thể được sử dụng trực tiếp hoặc tham khảo để tạo nội dung báo cáo.**

