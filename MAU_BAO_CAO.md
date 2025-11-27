# 📄 MẪU CẤU TRÚC BÁO CÁO CHUYÊN ĐỀ

> Template này cung cấp cấu trúc chi tiết và nội dung gợi ý cho từng phần của báo cáo chuyên đề.

---

## 📑 TRANG BÌA

**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN**

**BÁO CÁO CHUYÊN ĐỀ**

**ỨNG DỤNG AI NHẬN DIỆN VÀ TRA CỨU SẢN PHẨM**

**Giảng viên hướng dẫn**: Thầy Cáp Phạm Đình Thăng

**Nhóm thực hiện**:
- Nguyễn Hữu Việt — 24410375
- Nguyễn Ngọc Tuyên — 24410371

**Năm học**: 2024-2025

---

## 📋 MỤC LỤC

1. Giới thiệu
2. Cơ sở lý thuyết
3. Phân tích và thiết kế hệ thống
4. Cài đặt và triển khai
5. Đánh giá và kết quả
6. Kết luận và hướng phát triển
7. Tài liệu tham khảo
8. Phụ lục

---

## 1. GIỚI THIỆU

### 1.1. Đặt vấn đề

**Nội dung gợi ý**:
- Vấn đề tra cứu sản phẩm trên các sàn thương mại điện tử
- Tầm quan trọng của caption tiếng Việt cho ảnh sản phẩm
- Khó khăn trong việc tạo caption tự động, chính xác, có dấu

**Số liệu tham khảo**:
- Dataset: 7,638 ảnh sản phẩm từ Shopee
- Nhu cầu: Caption tiếng Việt có dấu, tự nhiên, ngắn gọn

### 1.2. Mục tiêu nghiên cứu

**Mục tiêu chính**:
1. Fine-tune mô hình BLIP để sinh caption tiếng Việt không dấu
2. Sử dụng Accent Restoration để chuyển caption không dấu → có dấu
3. Xây dựng API prototype để demo
4. Đánh giá mô hình theo yêu cầu: train 80% – test 20%

**Mục tiêu phụ**:
- Tối ưu cho macOS M1/M2/M3
- Xây dựng hệ thống caching và rate limiting
- Đánh giá bằng nhiều metrics (BLEU, ROUGE-L, SBERT)

### 1.3. Phạm vi nghiên cứu

**Phạm vi**:
- Dataset: 7,638 ảnh sản phẩm Shopee
- Model: BLIP (Salesforce) + Accent Restoration (peterhung)
- Platform: macOS với MPS backend
- Evaluation: Test set 20% (1,528 samples)

**Giới hạn**:
- Chỉ xử lý ảnh sản phẩm (không phải ảnh tổng quát)
- Caption ngắn gọn (max 50 tokens)
- Tiếng Việt có dấu

### 1.4. Cấu trúc báo cáo

**Mô tả ngắn gọn các chương**:
- Chương 2: Cơ sở lý thuyết về BLIP, Image Captioning, Accent Restoration
- Chương 3: Phân tích và thiết kế hệ thống
- Chương 4: Cài đặt và triển khai
- Chương 5: Đánh giá và kết quả
- Chương 6: Kết luận và hướng phát triển

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Image Captioning

**Định nghĩa**:
- Nhiệm vụ tạo mô tả văn bản cho ảnh
- Kết hợp Computer Vision và Natural Language Processing

**Các phương pháp**:
- Encoder-Decoder architecture
- Attention mechanism
- Transformer-based models

**Ứng dụng**:
- Accessibility (mô tả ảnh cho người khiếm thị)
- E-commerce (mô tả sản phẩm)
- Content generation

### 2.2. BLIP Model

**Giới thiệu**:
- **BLIP**: Bootstrapping Language-Image Pre-training
- Phát triển bởi Salesforce Research
- Pre-trained trên 129M ảnh-caption pairs

**Architecture**:
```
┌─────────────┐
│ Vision      │ ← ViT (Vision Transformer)
│ Encoder     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Cross-modal │ ← Attention mechanism
│ Attention   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Text        │ ← BERT-based decoder
│ Decoder     │
└─────────────┘
```

**Đặc điểm**:
- Vision Encoder: ViT (Vision Transformer)
- Text Decoder: BERT-based
- Cross-modal attention để kết hợp thông tin ảnh và text

**Fine-tuning**:
- Fine-tune trên dataset tiếng Việt
- Giữ nguyên architecture, chỉ cập nhật weights
- Training với caption tiếng Việt không dấu

### 2.3. Accent Restoration

**Vấn đề**:
- BLIP sinh caption tiếng Việt không dấu
- Cần phục hồi dấu để có caption tự nhiên

**Giải pháp**:
- Model: `peterhung/vietnamese-accent-marker-xlm-roberta`
- Architecture: XLM-RoBERTa + Token Classification
- Task: Predict accent label cho mỗi token

**Pipeline**:
```
Text không dấu
  ↓ Tokenize
Tokens
  ↓ Predict labels
Accent labels
  ↓ Merge tokens
Words có dấu
  ↓ Join
Text có dấu
```

**Accuracy**: 97%+

### 2.4. Evaluation Metrics

#### 2.4.1. BLEU Score

**Định nghĩa**:
- Đo độ tương đồng n-gram giữa prediction và reference
- Range: [0, 1]
- Càng cao càng tốt

**Công thức**:
```
BLEU = BP × exp(Σ log(p_n))
```
- BP: Brevity Penalty
- p_n: Precision của n-gram

**Hạn chế**:
- Chỉ đánh giá n-gram overlap
- Không đánh giá semantic similarity
- Thấp khi prediction ngắn hơn reference

#### 2.4.2. ROUGE-L

**Định nghĩa**:
- Đo độ tương đồng Longest Common Subsequence (LCS)
- Range: [0, 1]
- F1 score của LCS

**Công thức**:
```
ROUGE-L = F1(LCS)
```

**Ưu điểm**:
- Đánh giá cấu trúc câu
- Không phụ thuộc vào thứ tự từ

#### 2.4.3. SBERT Similarity

**Định nghĩa**:
- Đo semantic similarity bằng Sentence-BERT embeddings
- Range: [0, 1] (normalized cosine similarity)
- Càng cao càng tốt

**Công thức**:
```
Similarity = (cosine_sim(emb_pred, emb_ref) + 1) / 2
```

**Ưu điểm**:
- Đánh giá semantic similarity
- Phù hợp khi prediction và reference khác từ nhưng cùng nghĩa

**Model sử dụng**: `keepitreal/vietnamese-sbert`

---

## 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Kiến trúc tổng thể

**Diagram**: System Architecture Diagram

```
┌─────────────┐
│   Client    │
│  (Mobile/   │
│    Web)     │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────┐
│   FastAPI Server   │
│  - Routes           │
│  - Middleware       │
│  - Rate Limiting    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Caption Service    │
│  - Image Preprocess │
│  - Cache Check      │
└──────┬──────────────┘
       │
       ├──────────────┐
       ▼              ▼
┌──────────┐   ┌──────────────┐
│  BLIP    │   │   Accent    │
│  Model   │──▶│ Restoration │
└──────────┘   └──────────────┘
```

**Mô tả**:
- Client gửi ảnh qua REST API
- FastAPI server xử lý request
- Caption Service kiểm tra cache
- BLIP model sinh caption không dấu
- Accent Restoration phục hồi dấu
- Trả về caption có dấu

### 3.2. Pipeline xử lý

**Diagram**: Data Flow Diagram

**Training Pipeline**:
```
Dataset CSV (7,638 samples)
  ↓ Split 80/20
Train (6,110) + Test (1,528)
  ↓ Preprocess
Images + Captions
  ↓ Fine-tune BLIP
Model weights
  ↓ Save
models/blip_vietnamese_80_20/
```

**Inference Pipeline**:
```
Image Input
  ↓ Check Cache
  ├─ Cache hit → Return
  └─ Cache miss → Continue
  ↓ Preprocess Image
  - Resize if > 512px
  - Convert RGB
  ↓ BLIP Generation
  - Load image
  - Generate caption (no accent)
  ↓ Accent Restoration
  - Tokenize
  - Predict accents
  - Merge tokens
  ↓ Cache Result
  ↓ Return Response
```

### 3.3. API Design

**Endpoints**:

| Endpoint | Method | Mô tả | Input | Output |
|----------|--------|-------|-------|--------|
| `/api/caption` | POST | Caption không dấu (single) | Image file | Caption + metadata |
| `/api/caption/batch` | POST | Caption không dấu (batch) | Multiple images | List of captions |
| `/api/caption_full` | POST | Caption có dấu (single) | Image file | Caption + accent info |
| `/api/caption_full/batch` | POST | Caption có dấu (batch) | Multiple images | List of captions |
| `/api/accent/restore` | POST | Restore accent cho text | JSON {text} | Text with accent |
| `/api/health` | GET | Health check | - | Status info |
| `/api/cache/clear` | POST | Clear cache | - | Success message |

**Request/Response Examples**:

**Single Caption Request**:
```http
POST /api/caption_full
Content-Type: multipart/form-data

file: [image.jpg]
```

**Single Caption Response**:
```json
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

### 3.4. Database/Cache Design

**Cache Strategy**:
- **Storage**: In-memory dictionary
- **Key**: MD5 hash của ảnh (PNG format)
- **Value**: {caption, expires_at, created_at}
- **TTL**: 24 giờ (86400 giây)

**Cache Flow**:
```
Request Image
  ↓ Hash Image (MD5)
  ↓ Check Cache
  ├─ Found & Valid → Return cached caption
  └─ Not Found → Generate → Cache → Return
```

### 3.5. Error Handling

**Các lỗi xử lý**:
- Invalid image format → 400 Bad Request
- Image too large → 400 Bad Request
- Model not loaded → 500 Internal Server Error
- Rate limit exceeded → 429 Too Many Requests
- Cache error → Continue without cache

---

## 4. CÀI ĐẶT VÀ TRIỂN KHAI

### 4.1. Dataset Preparation

**Dataset gốc**:
- File: `train_bilingual_clean_v2.csv`
- Số lượng: 7,638 samples
- Format: `image, caption_vi`

**Preprocessing**:
1. Load CSV
2. Shuffle với random_state=42
3. Split 80/20
4. Save: `train_80.csv`, `test_20.csv`

**Kết quả**:
- Train: 6,110 samples
- Test: 1,528 samples

### 4.2. Model Training

**Script**: `train/train_blip_vietnamese.py`

**Parameters**:
```python
epochs = 5
batch_size = 2
learning_rate = 5e-5
warmup_steps = 500
max_length = 77
```

**Training Process**:
1. Load pretrained BLIP model
2. Load train dataset
3. Preprocess images + captions
4. Fine-tune với HuggingFace Trainer
5. Evaluate mỗi epoch
6. Save best model

**Thời gian**: ~3-4 giờ trên M1 Pro Max

**Output**: `models/blip_vietnamese_80_20/`

### 4.3. API Implementation

**Framework**: FastAPI

**Cấu trúc code**:
```
app/
├── main.py              # FastAPI app
├── api/
│   └── routes_caption.py  # API endpoints
├── core/
│   ├── config.py        # Configuration
│   ├── model_loader.py  # BLIP loader
│   └── accent_restoration_loader.py
├── services/
│   └── caption_service.py
└── utils/
    ├── cache.py
    └── rate_limit.py
```

**Key Features**:
- CORS support
- Rate limiting (60 req/min)
- Caching system
- Error handling
- Health check

### 4.4. Optimization Techniques

#### 4.4.1. MPS Compatibility

**Vấn đề**: MPS không hỗ trợ tốt attention_mask auto-inference

**Giải pháp**:
```python
if device == "mps":
    model_cpu = model.cpu()
    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
    output = model_cpu.generate(**inputs_cpu)
    model.to(device)  # Chuyển lại MPS
```

#### 4.4.2. Memory Management

**Vấn đề**: Memory leak trên MPS

**Giải pháp**:
- Cleanup tensors sau mỗi inference
- Clear cache mỗi 5 ảnh trong batch
- Move tensors về CPU trước khi delete
- Synchronize device sau mỗi operation

#### 4.4.3. Image Preprocessing

**Tối ưu**:
- Resize ảnh nếu > 512px (giảm VRAM)
- Convert RGB (chuẩn hóa format)
- Cache kết quả để tránh regenerate

---

## 5. ĐÁNH GIÁ VÀ KẾT QUẢ

### 5.1. Evaluation Setup

**Test Set**: 1,528 samples (20%)

**Metrics**:
- BLEU Score
- ROUGE-L F1
- SBERT Similarity

**Script**: `tools/evaluate_metrics.py`

### 5.2. Kết quả thực nghiệm

**Metrics Table**:

| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **BLEU** | 0.0141 | Thấp - do caption gốc dài, SEO |
| **ROUGE-L** | 0.1486 | Trung bình |
| **SBERT** | 0.6330 | **Khá tốt** - hiểu nghĩa tốt |

**Inference Performance**:
- Single image: ~0.8-1.2s/ảnh
- Batch (10 ảnh): ~5-8s
- Accent restoration: ~0.1-0.2s/text

**Accent Restoration Accuracy**: 97%+

### 5.3. Phân tích kết quả

#### 5.3.1. BLEU Score Thấp

**Nguyên nhân**:
- Caption gốc từ Shopee dài, nhiều SEO keywords
- Caption sinh ra ngắn gọn, không bắt chước format gốc
- Mục tiêu khác: caption gốc là SEO, caption sinh là mô tả

**Kết luận**: BLEU thấp là bình thường, không phải lỗi mô hình

#### 5.3.2. SBERT Score Tốt

**Nguyên nhân**:
- Mô hình hiểu nghĩa tốt
- Caption sinh ra đúng nghĩa sản phẩm
- Semantic similarity cao (0.6330)

**Kết luận**: Mô hình đạt mục tiêu - tạo caption đúng nghĩa

#### 5.3.3. Accent Restoration

**Kết quả**:
- Accuracy: 97%+
- Nhanh: ~0.1-0.2s/text
- Không lỗi ký tự

**Kết luận**: Pipeline 2 giai đoạn hiệu quả

### 5.4. Ví dụ minh họa

**Ví dụ 1**:
- **Ảnh**: Nước hoa nữ
- **Không dấu**: `nuoc hoa nu`
- **Có dấu**: `nước hoa nữ`

**Ví dụ 2**:
- **Ảnh**: Giày sneaker đỏ
- **Không dấu**: `giay sneaker mau do`
- **Có dấu**: `giày sneaker màu đỏ`

**Ví dụ 3**:
- **Ảnh**: Áo khoác thể thao
- **Không dấu**: `ao khoac the thao nu mau den`
- **Có dấu**: `áo khoác thể thao nữ màu đen`

### 5.5. So sánh với Baseline

**Baseline**: Pretrained BLIP (chưa fine-tune)

**So sánh**:
- Baseline: Caption tiếng Anh
- Fine-tuned: Caption tiếng Việt không dấu
- + Accent Restoration: Caption tiếng Việt có dấu

**Kết luận**: Fine-tuning thành công, mô hình sinh caption tiếng Việt

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Tổng kết

**Những phần đã hoàn thành**:
- ✅ Tiền xử lý dataset tiếng Việt (7,638 samples)
- ✅ Fine-tune BLIP trên 80% dataset
- ✅ Accent Restoration với accuracy 97%+
- ✅ Xây dựng API hoàn chỉnh với caching, rate limiting
- ✅ Test trên 20% dataset + tính metrics
- ✅ So sánh và phân tích kết quả

**Kết quả đạt được**:
- Mô hình sinh caption tiếng Việt có dấu, tự nhiên
- SBERT score 0.6330 - hiểu nghĩa tốt
- Inference time ~1s/ảnh
- API hoàn chỉnh, sẵn sàng demo

### 6.2. Hạn chế

1. **BLEU Score Thấp**
   - Do caption gốc dài, SEO
   - Không phải lỗi mô hình

2. **Inference Time**
   - ~1s/ảnh (có thể tối ưu thêm)
   - Batch processing giúp cải thiện

3. **Tokenizer BLIP**
   - Không hỗ trợ trực tiếp tiếng Việt
   - Phải dùng pipeline 2 giai đoạn

4. **Memory Usage**
   - Cao trên MPS
   - Cần cleanup thường xuyên

### 6.3. Hướng phát triển

**Ngắn hạn**:
1. Tăng số epoch để cải thiện SBERT score
2. Tối ưu inference time (batch processing, model quantization)
3. Cải thiện memory management

**Dài hạn**:
1. Fine-tune tokenizer tiếng Việt cho BLIP
2. Tích hợp vào ứng dụng thực tế
3. Deploy lên server production
4. Xây dựng UI web để demo
5. Mở rộng dataset với nhiều loại sản phẩm

### 6.4. Đóng góp

**Đóng góp của dự án**:
- Fine-tune BLIP cho tiếng Việt
- Pipeline 2 giai đoạn hiệu quả (BLIP + Accent Restoration)
- API prototype hoàn chỉnh
- Evaluation với nhiều metrics

**Ứng dụng thực tế**:
- E-commerce: Tự động tạo caption cho sản phẩm
- Accessibility: Mô tả ảnh cho người khiếm thị
- Content generation: Tạo mô tả tự động

---

## 7. TÀI LIỆU THAM KHẢO

1. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. *ICML 2022*.

2. Salesforce Research. (2022). BLIP: Bootstrapping Language-Image Pre-training. https://github.com/salesforce/BLIP

3. peterhung. (2023). Vietnamese Accent Marker. https://huggingface.co/peterhung/vietnamese-accent-marker-xlm-roberta

4. FastAPI Documentation. https://fastapi.tiangolo.com/

5. PyTorch MPS Backend. https://pytorch.org/docs/stable/notes/mps.html

6. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL 2002*.

7. Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL 2004*.

8. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

---

## 8. PHỤ LỤC

### Phụ lục A: Code Snippets

**A.1. Model Loading**:
```python
# app/core/model_loader.py
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
```

**A.2. Caption Generation**:
```python
# app/services/caption_service.py
inputs = processor(images=image, return_tensors="pt").to(device)
output = model.generate(**inputs, **generation_kwargs)
caption = processor.decode(output[0], skip_special_tokens=True)
```

**A.3. Accent Restoration**:
```python
# app/core/accent_restoration_loader.py
inputs = accent_tokenizer(tokens, is_split_into_words=True, ...)
outputs = accent_model(**inputs)
predictions = outputs["logits"].argmax(axis=2)
# Apply accents...
```

### Phụ lục B: API Documentation

**Swagger UI**: `http://127.0.0.1:8000/docs`

**Endpoints**:
- `/api/caption` - POST
- `/api/caption/batch` - POST
- `/api/caption_full` - POST
- `/api/caption_full/batch` - POST
- `/api/accent/restore` - POST
- `/api/health` - GET

### Phụ lục C: Evaluation Results

**Full metrics table**:
- BLEU: 0.0141
- ROUGE-L: 0.1486
- SBERT: 0.6330

**Sample predictions**:
- [Danh sách một số caption mẫu]

### Phụ lục D: Dataset Statistics

**Dataset**:
- Total: 7,638 samples
- Train: 6,110 (80%)
- Test: 1,528 (20%)
- Images: 7,443 files

---

**Kết thúc báo cáo**

---

## 📝 GHI CHÚ

- Template này cung cấp cấu trúc và nội dung gợi ý
- Có thể điều chỉnh theo yêu cầu của giảng viên
- Tham khảo `PHAN_TICH_CHI_TIET.md` để có thông tin chi tiết hơn
- Tham khảo `TOM_TAT_BAO_CAO.md` để có số liệu nhanh

