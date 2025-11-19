# 🖼️ BLIP Vietnamese Captioning API

> Hệ thống tạo caption tiếng Việt cho ảnh sản phẩm sử dụng BLIP model được fine-tune với dữ liệu tiếng Việt.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Cấu trúc Project](#-cấu-trúc-project)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [API Endpoints](#-api-endpoints)
- [Training Model](#-training-model)
- [Tối ưu cho macOS](#-tối-ưu-cho-macos)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Tính năng

- ✅ **Caption Generation**: Tạo caption tiếng Việt cho ảnh sản phẩm
- ✅ **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- ✅ **Accent Restoration**: Tự động thêm dấu tiếng Việt cho caption
- ✅ **Caching**: Cache kết quả để tăng tốc độ xử lý
- ✅ **Rate Limiting**: Giới hạn số lượng request
- ✅ **Authentication**: Hỗ trợ API key authentication (tùy chọn)
- ✅ **Health Check**: Endpoint kiểm tra trạng thái API
- ✅ **Auto Device Detection**: Tự động sử dụng MPS (macOS), CUDA, hoặc CPU
- ✅ **Memory Optimization**: Tối ưu memory cho macOS M1/M2/M3

## 📁 Cấu trúc Project

```
CHUYEN_DE_Backend/
├── app/                          # Application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI entrypoint
│   ├── api/                      # API routes
│   │   ├── __init__.py
│   │   └── routes_caption.py     # Caption endpoints
│   ├── core/                     # Core configuration & loaders
│   │   ├── __init__.py
│   │   ├── config.py             # Application configuration
│   │   ├── model_loader.py        # BLIP model loader
│   │   └── accent_restoration_loader.py  # Accent restoration model
│   ├── services/                 # Business logic
│   │   └── caption_service.py    # Caption generation service
│   ├── middleware/               # Middleware
│   │   ├── __init__.py
│   │   └── auth.py               # Authentication middleware
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── cache.py              # Caching utilities
│       └── rate_limit.py         # Rate limiting utilities
│
├── train/                        # Training scripts
│   └── train_blip_vietnamese.py  # Fine-tune BLIP model
│
├── tools/                        # Utility tools
│   ├── copy_images_by_csv.py
│   ├── evaluate_metrics.py       # Evaluate model metrics
│   ├── find_full_test_csv.py
│   ├── run_inference_full_test.py
│   └── run_inference_shopee_test.py
│
├── data/                         # Data directory
│   ├── train_bilingual_clean_v2.csv
│   └── images/                   # Product images (7443 images)
│
├── models/                       # Model weights
│   ├── blip_vietnamese/          # Fine-tuned BLIP model
│   └── accent_restoration/       # Accent restoration model
│
├── configs/                      # Configuration files
│   └── infer.yaml
│
├── logs/                         # Log files
├── outputs/                      # Training outputs & predictions
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- pip
- (Tùy chọn) CUDA cho GPU hoặc macOS với Apple Silicon (M1/M2/M3)

### Bước 1: Clone repository

```bash
git clone https://github.com/nguyenhuuviet/CHUYEN_DE_Backend.git
cd CHUYEN_DE_Backend
```

### Bước 2: Tạo virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc
venv\Scripts\activate     # Windows
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý cho macOS M1/M2/M3:**
- PyTorch sẽ tự động sử dụng MPS (Metal Performance Shaders) backend
- Không cần cài đặt CUDA

### Bước 4: Tải model weights

Model sẽ tự động được tải từ HuggingFace khi chạy lần đầu:
- BLIP model: Tự động load từ `models/blip_vietnamese/` hoặc HuggingFace
- Accent restoration model: Tự động tải từ `peterhung/vietnamese-accent-marker-xlm-roberta`

## 🎯 Sử dụng

### Khởi động API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://127.0.0.1:8000`

### Truy cập API Documentation

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## 📡 API Endpoints

### 1. Root Endpoint

```http
GET /
```

**Response:**
```json
{
  "message": "BLIP Vietnamese Captioning API is running 🚀",
  "docs": "/docs",
  "health": "/api/health"
}
```

### 2. API Information

```http
GET /info
```

Trả về thông tin về API, các tính năng và endpoints có sẵn.

### 3. Health Check

```http
GET /api/health
```

Kiểm tra trạng thái API và model.

### 4. Generate Caption (Không dấu)

```http
POST /api/caption
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "caption_vi": "ao khoac the thao nu mau den",
  "device": "mps",
  "cached": false,
  "processing_time": 0.45
}
```

### 5. Generate Caption Batch (Không dấu)

```http
POST /api/caption/batch
Content-Type: multipart/form-data
```

**Request:**
- `files`: Multiple image files (max 10 images per batch)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "caption_vi": "ao khoac the thao nu mau den",
      "device": "mps",
      "cached": false,
      "processing_time": 0.45
    },
    ...
  ],
  "total_time": 2.34
}
```

### 6. Generate Caption Full (Có dấu)

```http
POST /api/caption_full
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file

**Response:**
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

### 7. Generate Caption Full Batch (Có dấu)

```http
POST /api/caption_full/batch
Content-Type: multipart/form-data
```

**Request:**
- `files`: Multiple image files (max 10 images per batch)

### 8. Restore Accent

```http
POST /api/accent/restore
Content-Type: application/json
```

**Request:**
```json
{
  "text": "ao khoac the thao mau den"
}
```

**Response:**
```json
{
  "success": true,
  "text_no_accent": "ao khoac the thao mau den",
  "text_with_accent": "áo khoác thể thao màu đen",
  "device": "mps",
  "accent_model_loaded": true,
  "processing_time": 0.12
}
```

## 🧪 Testing

### Test với cURL

```bash
# Test single caption
curl -X POST "http://127.0.0.1:8000/api/caption" \
  -F "file=@/path/to/image.jpg"

# Test caption with accent
curl -X POST "http://127.0.0.1:8000/api/caption_full" \
  -F "file=@/path/to/image.jpg"

# Test accent restoration
curl -X POST "http://127.0.0.1:8000/api/accent/restore" \
  -H "Content-Type: application/json" \
  -d '{"text": "ao khoac the thao mau den"}'
```

### Test với Postman

1. Mở Postman
2. Tạo request mới: `POST http://127.0.0.1:8000/api/caption`
3. Chọn tab **Body** → **form-data**
4. Thêm key `file` (type: File) và chọn ảnh
5. Gửi request

## 🎓 Training Model

### Chuẩn bị dữ liệu

Đảm bảo bạn có:
- File CSV: `data/train_bilingual_clean_v2.csv` (7638 samples)
- Thư mục ảnh: `data/images/` với tất cả ảnh sản phẩm
- Tên file ảnh phải khớp với cột `image` trong CSV

### Chạy training

```bash
cd train
python train_blip_vietnamese.py
```

**Thông tin training:**
- **Epochs**: 5
- **Batch size**: 2 (tối ưu cho M1)
- **Learning rate**: 5e-5
- **Model output**: `models/blip_vietnamese/`

**Lưu ý:**
- Training trên M1 Pro Max có thể mất vài giờ
- Model sẽ tự động sử dụng MPS backend nếu có GPU

## 🍎 Tối ưu cho macOS

Project đã được tối ưu đầy đủ cho macOS với Apple Silicon (M1/M2/M3):

### Device & Memory Optimization

- ✅ **Auto MPS Detection**: Tự động detect và sử dụng MPS backend
- ✅ **Memory Management**: Tự động cleanup cache sau mỗi inference
- ✅ **Device Synchronization**: Đảm bảo operations hoàn thành trước khi tiếp tục
- ✅ **Batch Processing**: Cleanup memory mỗi 5 ảnh trong batch

### Model Optimization

- ✅ **Image Preprocessing**: Tự động resize ảnh lớn (>512px) để giảm memory usage
- ✅ **Float32 Precision**: Giữ nguyên float32 cho MPS (đảm bảo stability)
- ✅ **Model Compilation**: Tự động compile model với `torch.compile()` (PyTorch 2.0+)

### Training Optimization

- ✅ **Batch Size**: Phù hợp với M1 (2-4)
- ✅ **No FP16**: Không sử dụng fp16 (MPS chưa hỗ trợ tốt)
- ✅ **No Multiprocessing**: Tắt multiprocessing workers (tránh lỗi trên macOS)

## ⚙️ Cấu hình

Các tham số có thể tùy chỉnh thông qua biến môi trường hoặc sửa trực tiếp trong `app/core/config.py`:

### Model Configuration
- `MODEL_PATH`: Đường dẫn lưu model (mặc định: `models/blip_vietnamese`)
- `ACCENT_MODEL_NAME`: Accent restoration model (mặc định: `peterhung/vietnamese-accent-marker-xlm-roberta`)

### Generation Configuration
- `MAX_NEW_TOKENS`: Số token tối đa khi generate caption (mặc định: 50)
- `NUM_BEAMS`: Số beams cho beam search (mặc định: 3)
- `REPETITION_PENALTY`: Penalty cho repetition (mặc định: 1.2)

### API Configuration
- `CORS_ORIGINS`: Các domain được phép gọi API (mặc định: `*`)
- `ENABLE_AUTH`: Bật/tắt authentication (mặc định: `False`)
- `ENABLE_CACHE`: Bật/tắt caching (mặc định: `True`)
- `ENABLE_RATE_LIMIT`: Bật/tắt rate limiting (mặc định: `True`)
- `MAX_BATCH_SIZE`: Số ảnh tối đa trong batch (mặc định: 10)
- `RATE_LIMIT_PER_MINUTE`: Số request tối đa mỗi phút (mặc định: 60)

Xem chi tiết trong `app/core/config.py`.

## 🔧 Troubleshooting

### Lỗi: Model chưa được train

Nếu chưa train model, API sẽ tự động load pretrained model (chưa fine-tune tiếng Việt).

**Giải pháp:** Chạy training trước:
```bash
cd train
python train_blip_vietnamese.py
```

### Lỗi: Không tìm thấy ảnh

Đảm bảo tất cả ảnh trong CSV đều có trong thư mục `data/images/`

### Lỗi: Out of memory

**Giải pháp:**
1. Giảm batch size trong training: `per_device_train_batch_size=1`
2. Giảm `MAX_BATCH_SIZE` trong config
3. Resize ảnh trước khi gửi lên API

### Lỗi: Too many open files (macOS)

```bash
ulimit -n 2048
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Dự án đồ án chuyên đề - UIT

---

**Made with ❤️ by UIT Students**
