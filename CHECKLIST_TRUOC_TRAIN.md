# ✅ CHECKLIST TRƯỚC KHI TRAIN

## 🔍 Kiểm tra nhanh

Chạy lệnh sau để kiểm tra tự động:

```bash
source venv/bin/activate
python check_setup.py
```

## 📋 Checklist thủ công

### 1. ✅ Cấu trúc Project
- [x] Thư mục `app/`, `train/`, `data/`, `models/`, `logs/` đã có
- [x] File `train/train_blip_vietnamese.py` tồn tại
- [x] File `app/main.py` và các API routes tồn tại

### 2. ✅ Dataset
- [x] File `data/train_bilingual_clean_v2.csv` tồn tại (7638 samples)
- [x] Cột `image` và `caption_vi` có trong CSV
- [ ] **⚠️ THIẾU:** Thư mục `data/images/` cần có ảnh sản phẩm

**QUAN TRỌNG:** 
- Tất cả ảnh trong CSV phải có trong `data/images/`
- Tên file ảnh phải khớp chính xác với cột `image` trong CSV

### 3. ✅ Model Directory
- [x] Thư mục `models/blip_vietnamese/` đã được tạo sẵn
- [x] Thư mục trống, sẵn sàng để lưu model sau training

### 4. ✅ Python Environment
- [x] Virtual environment `venv/` đã được tạo
- [x] Dependencies đã được cài đặt từ `requirements.txt`
- [x] Transformers, PyTorch, FastAPI đã có

### 5. ✅ Device (MPS/CUDA/CPU)
- [x] MPS (Metal) đã được detect - macOS M1 GPU sẵn sàng
- [x] Training sẽ tự động sử dụng MPS backend

### 6. ✅ Configuration
- [x] File `app/core/config.py` có MODEL_PATH đúng: `models/blip_vietnamese`
- [x] File `.env.example` có sẵn (tùy chọn tạo `.env` để tùy chỉnh)

## 🚀 Sẵn sàng Training

### Trước khi train:

1. **Đảm bảo có ảnh trong `data/images/`**
   ```bash
   # Kiểm tra số lượng ảnh
   ls data/images/ | wc -l
   ```

2. **Kích hoạt virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Chạy training**
   ```bash
   cd train
   python train_blip_vietnamese.py
   ```

### Kết quả mong đợi:

Sau khi training xong, bạn sẽ thấy:

```
models/blip_vietnamese/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt
```

### Test API sau training:

```bash
# Terminal 1: Chạy API
uvicorn app.main:app --reload

# Terminal 2: Test với Postman hoặc cURL
curl -X POST "http://127.0.0.1:8000/api/caption" \
  -F "file=@/path/to/image.jpg"
```

## ⚠️ Lưu ý

1. **Thư mục images trống:** Cần thêm ảnh vào `data/images/` trước khi train
2. **Training time:** Trên M1 Pro Max, training có thể mất 2-6 giờ tùy số lượng ảnh
3. **Memory:** Nếu gặp lỗi out of memory, giảm `TRAIN_BATCH_SIZE` trong config

## 📊 Trạng thái hiện tại

| Mục | Trạng thái |
|-----|-----------|
| Cấu trúc project | ✅ OK |
| Dataset CSV | ✅ OK (7638 samples) |
| Thư mục images | ⚠️ **TRỐNG - Cần thêm ảnh** |
| Model directory | ✅ OK |
| Python packages | ✅ OK |
| Device (MPS) | ✅ OK |
| Config | ✅ OK |

**Tổng kết:** Sẵn sàng training sau khi thêm ảnh vào `data/images/`!

