# 🔄 HƯỚNG DẪN TRAIN LẠI MODEL ĐỂ CÓ DẤU TIẾNG VIỆT

## 🎯 Vấn đề
Model hiện tại đang output tiếng Việt **không có dấu** mặc dù dataset có dấu đầy đủ.

## ✅ Đã cải thiện
1. ✅ **Generation parameters** đã được tối ưu:
   - Tăng `num_beams` từ 3 → 5
   - Thêm `repetition_penalty` = 1.2
   - Thêm `length_penalty` = 1.0

2. ✅ **Config training** đã được cập nhật:
   - Tăng `num_train_epochs` từ 5 → **10 epochs**
   - Giảm `learning_rate` từ 5e-5 → **3e-5** (học kỹ hơn)
   - Set `resume_from_checkpoint: false` để train lại từ đầu

## 🚀 CÁCH TRAIN LẠI

### Bước 1: Kiểm tra dataset có dấu
```bash
# Xem một vài samples từ dataset
head -5 data/train_bilingual_clean_v2.csv | cut -d',' -f8
```

Bạn sẽ thấy các caption có dấu như:
- "Sách trẻ em - Học tiếng Anh cho TK"
- "Mặt nạ 50g của Mahira Beauty"

### Bước 2: Backup model cũ (tùy chọn)
```bash
# Backup model hiện tại nếu muốn
mv models/blip_vietnamese models/blip_vietnamese_backup_$(date +%Y%m%d)
```

### Bước 3: Train lại với config mới
```bash
# Kích hoạt virtual environment (nếu có)
source venv/bin/activate  # hoặc source .venv/bin/activate

# Chạy training
python train/train_blip_vietnamese.py
```

### Bước 4: Kiểm tra kết quả
Sau khi training xong, test API:
```bash
# Khởi động API
python -m uvicorn app.main:app --reload

# Test với Postman hoặc curl
curl -X POST "http://localhost:8000/api/caption" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/images/[một_ảnh_bất_kỳ].jpg"
```

## 📊 Thời gian training dự kiến
- **10 epochs** với dataset ~7,600 samples
- Batch size: 8, gradient accumulation: 2
- **Ước tính:** 4-6 giờ trên M1 Pro Max (tùy vào tốc độ)

## ⚙️ Cấu hình hiện tại (configs/train_blip.yaml)
```yaml
trainer:
  num_train_epochs: 10        # Tăng từ 5
  learning_rate: 3.0e-5       # Giảm từ 5e-5
  resume_from_checkpoint: false  # Train từ đầu
```

## 🔍 Lưu ý quan trọng

1. **Model cần train lại từ đầu** vì model hiện tại chưa học đủ tốt tiếng Việt có dấu
2. **Generation parameters** đã được cải thiện nhưng chỉ giúp một phần, vấn đề chính là model cần học lại
3. **Nếu vẫn không có dấu sau khi train**, có thể cần:
   - Tăng thêm epochs (15-20)
   - Kiểm tra lại dataset có đủ samples có dấu không
   - Thử learning rate khác (2e-5 hoặc 4e-5)

## 📝 Checklist trước khi train
- [ ] Đã backup model cũ (nếu cần)
- [ ] Đã kiểm tra dataset có dấu đầy đủ
- [ ] Đã cập nhật config (num_train_epochs: 10, learning_rate: 3e-5)
- [ ] Đã có đủ thời gian (4-6 giờ)
- [ ] Đã kiểm tra disk space đủ (cần ~2-3GB cho checkpoints)

## 🎯 Kết quả mong đợi
Sau khi train lại, model sẽ output:
- ✅ "Sách trẻ em - Học tiếng Anh cho TK" (có dấu)
- ❌ Không còn: "sach tre em - hoc tieng anh cho tk" (không dấu)

---

**Lưu ý:** Generation parameters mới đã được áp dụng vào API, nhưng để có kết quả tốt nhất, **bắt buộc phải train lại model** với config mới.

