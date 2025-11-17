# BLIP Vietnamese Captioning API

Hệ thống tạo caption tiếng Việt cho ảnh sản phẩm sử dụng BLIP model được fine-tune.

## 📁 Cấu trúc Project

```
CHUYEN_DE_Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes_caption.py        # API routes cho caption
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Cấu hình ứng dụng
│   │   ├── model_loader.py          # Load BLIP model
│   │   └── accent_restoration_loader.py  # Load accent restoration model
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py                  # Authentication middleware
│   └── utils/
│       ├── __init__.py
│       ├── cache.py                 # Caching utilities
│       └── rate_limit.py            # Rate limiting utilities
│
├── train/
│   └── train_blip_vietnamese.py     # Fine-tune BLIP
│
├── tools/
│   └── copy_images_by_csv.py        # Utility tool
│
├── data/
│   ├── train_bilingual_clean_v2.csv
│   └── images/                      # Ảnh sản phẩm
│
├── models/
│   └── blip_vietnamese/             # BLIP model sau fine-tune
│
├── configs/
│   └── infer.yaml                   # Config mẫu
│
├── logs/                            # Logs directory
├── outputs/                         # Training outputs
│
├── .gitignore
├── requirements.txt
├── README.md
├── HUONG_DAN_CHINH_THUC.md          # Hướng dẫn chính thức
├── HUONG_DAN_TEST_POSTMAN.md        # Hướng dẫn test API
└── Vietnamese_Caption_API.postman_collection.json
```

## 🚀 Cài đặt

### 1. Tạo virtual environment (khuyến nghị)

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý cho macOS M1:**
- PyTorch sẽ tự động sử dụng MPS (Metal Performance Shaders) backend
- Không cần cài đặt CUDA

### 3. Cấu hình (Tùy chọn)

Các tham số có thể tùy chỉnh thông qua biến môi trường hoặc sửa trực tiếp trong `app/core/config.py`:

- `MODEL_PATH`: Đường dẫn lưu model (mặc định: `models/blip_vietnamese`)
- Accent restoration model được tự động tải từ HuggingFace (`peterhung/vietnamese-accent-marker-xlm-roberta`)
- `TRAIN_BATCH_SIZE`: Batch size cho training (mặc định: 2)
- `MAX_NEW_TOKENS`: Số token tối đa khi generate caption (mặc định: 50)
- `CORS_ORIGINS`: Các domain được phép gọi API (mặc định: `*`)
- `ENABLE_AUTH`: Bật/tắt authentication (mặc định: `False`)
- `ENABLE_CACHE`: Bật/tắt caching (mặc định: `True`)
- `ENABLE_RATE_LIMIT`: Bật/tắt rate limiting (mặc định: `True`)

Xem chi tiết trong `app/core/config.py`.

## 📊 Training Model

### Bước 1: Chuẩn bị dữ liệu

**QUAN TRỌNG:** Đảm bảo file CSV và thư mục ảnh đã sẵn sàng:
- `data/train_bilingual_clean_v2.csv` ✅ (đã có - 7638 samples)
- `data/images/` - **Cần thêm tất cả ảnh sản phẩm vào đây**

Tên file ảnh phải khớp với cột `image` trong CSV.

### Bước 2: Chạy training

```bash
cd train
python train_blip_vietnamese.py
```

**Thông tin training:**
- Epochs: 5
- Batch size: 2 (tối ưu cho M1)
- Learning rate: 5e-5
- Model sẽ được lưu tại: `models/blip_vietnamese/`

**Lưu ý:**
- Training trên M1 Pro Max có thể mất vài giờ tùy vào số lượng ảnh
- Model sẽ tự động sử dụng MPS backend nếu có GPU

## 🎯 Chạy API

### Khởi động server

```bash
uvicorn app.main:app --reload
```

Server sẽ chạy tại: `http://127.0.0.1:8000`

### API Endpoints

#### 1. Root
```
GET http://127.0.0.1:8000/
```

#### 2. Info
```
GET http://127.0.0.1:8000/info
```
Trả về thông tin về API và các endpoints có sẵn.

#### 3. Health Check
```
GET http://127.0.0.1:8000/api/health
```
Kiểm tra trạng thái API và model.

#### 4. Generate Caption (Không dấu)
```
POST http://127.0.0.1:8000/api/caption
Content-Type: multipart/form-data

Body:
  file: <ảnh sản phẩm>
```

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

#### 5. Generate Caption Batch (Không dấu)
```
POST http://127.0.0.1:8000/api/caption/batch
Content-Type: multipart/form-data

Body:
  files: [<ảnh 1>, <ảnh 2>, ...]
```

#### 6. Generate Caption Full (Có dấu)
```
POST http://127.0.0.1:8000/api/caption_full
Content-Type: multipart/form-data

Body:
  file: <ảnh sản phẩm>
```

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

#### 7. Generate Caption Full Batch (Có dấu)
```
POST http://127.0.0.1:8000/api/caption_full/batch
Content-Type: multipart/form-data

Body:
  files: [<ảnh 1>, <ảnh 2>, ...]
```

#### 8. Restore Accent (Test accent restoration)
```
POST http://127.0.0.1:8000/api/accent/restore
Content-Type: application/json

Body:
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

### Swagger Documentation

Truy cập: `http://127.0.0.1:8000/docs`

## 🧪 Test với Postman

1. Mở Postman
2. Tạo request mới: `POST http://127.0.0.1:8000/api/caption`
3. Chọn tab **Body** → **form-data**
4. Thêm key `file` (type: File) và chọn ảnh
5. Gửi request

## 📱 Test với cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/caption" \
  -F "file=@/path/to/image.jpg"
```

## ⚙️ Tối ưu cho macOS M1/M2/M3

Project đã được tối ưu đầy đủ cho macOS với Apple Silicon (M1/M2/M3):

### Tối ưu Device & Memory:
- ✅ Tự động detect và sử dụng MPS backend (Metal Performance Shaders)
- ✅ Memory management: Tự động cleanup cache sau mỗi inference
- ✅ Device synchronization: Đảm bảo operations hoàn thành trước khi tiếp tục
- ✅ Batch processing: Cleanup memory mỗi 5 ảnh trong batch

### Tối ưu Model:
- ✅ Model compilation: Tự động compile model với `torch.compile()` (PyTorch 2.0+) cho CUDA/CPU
- ✅ Float32 precision: Giữ nguyên float32 cho MPS (đảm bảo stability)
- ✅ Image preprocessing: Tự động resize ảnh lớn (>512px) để giảm memory usage

### Tối ưu Training:
- ✅ Batch size phù hợp với M1 (2-4)
- ✅ Không sử dụng fp16 (MPS chưa hỗ trợ tốt)
- ✅ Tắt multiprocessing workers (tránh lỗi trên macOS)

## 🧱 Cấu hình bằng file YAML

- File huấn luyện: `configs/train_blip.yaml` (hyperparams, đường dẫn dữ liệu, model gốc)
- File suy luận: `configs/infer.yaml` (model_id phiên bản đã phát hành, tham số generate)

Bạn có thể chỉnh sửa hai file trên thay vì sửa trực tiếp trong code.

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

Giảm batch size trong `train/train_blip_vietnamese.py`:
```python
per_device_train_batch_size=1  # Thay vì 2
```

## 📝 Quy trình làm việc

1. **Training** → Fine-tune model với dữ liệu tiếng Việt
2. **Deploy API** → Chạy FastAPI server
3. **Test** → Sử dụng Postman hoặc Swagger docs
4. **Mobile App** → Gọi API từ ứng dụng mobile

## ✨ Tính năng đã có

- ✅ Batch processing (xử lý nhiều ảnh cùng lúc)
- ✅ Caching để tăng tốc độ
- ✅ Authentication/Authorization (có thể bật/tắt)
- ✅ Rate limiting (có thể bật/tắt)
- ✅ Accent restoration (tạo caption có dấu)
- ✅ Health check endpoint
- ✅ API documentation (Swagger)

## 📦 Version hóa và lưu trữ mô hình

- Không commit checkpoint nặng vào Git. Dùng 1 trong các cách:
  - Hugging Face Hub (khuyến nghị): upload model và dùng `model_id` trong `configs/infer.yaml`
  - Hoặc S3/MinIO/Google Drive và trỏ `MODEL_PATH` local khi deploy
- Gợi ý quy trình trên Git:
  1) Tạo nhánh: `git checkout -b feature/train-blip-vn`
  2) Chạy train theo `configs/train_blip.yaml`, upload model lên Hub/S3
  3) Cập nhật `configs/infer.yaml` -> `model.model_id` bằng version mới
  4) Mở PR → review → merge → tạo tag `model-vX.Y`

## 🎯 Next Steps

- [ ] Deploy lên cloud (AWS, GCP, Azure)
- [ ] Thêm monitoring và logging
- [ ] Tối ưu model inference
- [ ] Thêm metrics và analytics

## 📄 License

Dự án đồ án chuyên đề - UIT


