# BLIP Vietnamese Captioning API

Hệ thống tạo caption tiếng Việt cho ảnh sản phẩm sử dụng BLIP model được fine-tune.

## 📁 Cấu trúc Project

```
CHUYEN_DE_Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes_caption.py   # API upload ảnh, trả caption
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Cấu hình ứng dụng
│   │   └── model_loader.py     # Load BLIP model (tối ưu M1)
│   └── utils/
│       └── __init__.py
│
├── train/
│   └── train_blip_vietnamese.py # Fine-tune BLIP
│
├── data/
│   ├── train_bilingual_clean_v2.csv
│   └── images/                 # Ảnh sản phẩm
│
├── models/
│   └── blip_vietnamese/        # Model sau fine-tune (đã tạo sẵn)
│
├── logs/                       # Training logs
│
├── .env.example                # Template file cấu hình
├── .gitignore
├── requirements.txt
└── README.md
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

Nếu muốn tùy chỉnh cấu hình, copy file `.env.example` thành `.env` và chỉnh sửa:

```bash
cp .env.example .env
```

Các tham số có thể tùy chỉnh:
- `MODEL_PATH`: Đường dẫn lưu model (mặc định: `models/blip_vietnamese`)
- `TRAIN_BATCH_SIZE`: Batch size cho training (mặc định: 2)
- `MAX_NEW_TOKENS`: Số token tối đa khi generate caption (mặc định: 50)
- `CORS_ORIGINS`: Các domain được phép gọi API (mặc định: `*`)

Nếu không tạo file `.env`, hệ thống sẽ sử dụng giá trị mặc định từ `app/core/config.py`.

## 📊 Training Model

### Bước 0: Kiểm tra Setup (Khuyến nghị)

Chạy script kiểm tra trước khi training:

```bash
source venv/bin/activate
python check_setup.py
```

Script sẽ kiểm tra:
- ✅ Cấu trúc thư mục
- ✅ Dataset CSV và thư mục images
- ✅ Python packages
- ✅ Device (MPS/CUDA/CPU)
- ✅ Training script và API files

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

#### 2. Health Check
```
GET http://127.0.0.1:8000/api/health
```

#### 3. Generate Caption
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
  "caption_vi": "áo khoác thể thao nữ màu đen",
  "device": "mps"
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

## ⚙️ Tối ưu cho macOS M1

Project đã được tối ưu cho macOS M1 Pro Max:

- ✅ Tự động sử dụng MPS backend (Metal Performance Shaders)
- ✅ Batch size phù hợp với M1 (2-4)
- ✅ Không sử dụng fp16 (MPS chưa hỗ trợ tốt)
- ✅ Tắt multiprocessing workers (tránh lỗi trên macOS)

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

## 🎯 Next Steps

- [ ] Thêm endpoint `/api/batch` để xử lý nhiều ảnh cùng lúc
- [ ] Thêm caching để tăng tốc độ
- [ ] Deploy lên cloud (AWS, GCP, Azure)
- [ ] Thêm authentication/authorization
- [ ] Thêm rate limiting

## 📄 License

Dự án đồ án chuyên đề - UIT


