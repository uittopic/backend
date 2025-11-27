# 📋 TÓM TẮT BÁO CÁO CHUYÊN ĐỀ

## 🎯 THÔNG TIN DỰ ÁN

**Tên dự án**: Hệ thống tạo caption tiếng Việt cho ảnh sản phẩm sử dụng BLIP

**Giảng viên hướng dẫn**: Thầy Cáp Phạm Đình Thăng

**Nhóm thực hiện**:
- Nguyễn Hữu Việt — 24410375
- Nguyễn Ngọc Tuyên — 24410371

---

## 📊 SỐ LIỆU QUAN TRỌNG

### Dataset
- **Tổng số mẫu**: 7,638
- **Train (80%)**: 6,110 samples
- **Test (20%)**: 1,528 samples
- **Số ảnh**: 7,443 ảnh sản phẩm Shopee

### Model Performance
| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **BLEU** | 0.0141 | Thấp - do caption gốc dài, SEO |
| **ROUGE-L** | 0.1486 | Trung bình |
| **SBERT** | 0.6330 | **Khá tốt** - hiểu nghĩa tốt |

### Training
- **Epochs**: 5
- **Batch size**: 2
- **Learning rate**: 5e-5
- **Thời gian train**: ~3-4 giờ (M1 Pro Max)

### Inference
- **Single image**: ~0.8-1.2s/ảnh
- **Batch (10 ảnh)**: ~5-8s
- **Accent restoration**: ~0.1-0.2s/text
- **Accuracy accent**: 97%+

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### Pipeline
```
Ảnh Input 
  ↓
BLIP Model (Fine-tuned 80%)
  ↓ Caption không dấu
Accent Restoration (peterhung/vietnamese-accent-marker)
  ↓ Caption có dấu
Output: Caption tiếng Việt có dấu
```

### Technology Stack
- **Framework**: FastAPI 0.104.1
- **ML**: PyTorch 2.1.0, Transformers 4.35.0
- **Model**: BLIP (Salesforce) + Accent Restoration (peterhung)
- **Image**: Pillow 10.1.0
- **Data**: Pandas 2.1.3

---

## 🔑 CÁC THÀNH PHẦN CHÍNH

### 1. Model Loader (`app/core/model_loader.py`)
- Load BLIP fine-tuned model
- Auto-detect device (MPS/CUDA/CPU)
- Tối ưu cho macOS M1/M2/M3

### 2. Accent Restoration (`app/core/accent_restoration_loader.py`)
- Model: `peterhung/vietnamese-accent-marker-xlm-roberta`
- Token Classification
- Accuracy: 97%+

### 3. Caption Service (`app/services/caption_service.py`)
- Business logic cho caption generation
- Memory optimization
- Image preprocessing

### 4. API Routes (`app/api/routes_caption.py`)
- `/api/caption` - Caption không dấu (single)
- `/api/caption/batch` - Caption không dấu (batch)
- `/api/caption_full` - Caption có dấu (single)
- `/api/caption_full/batch` - Caption có dấu (batch)
- `/api/accent/restore` - Restore accent cho text
- `/api/health` - Health check

### 5. Training Script (`train/train_blip_vietnamese.py`)
- Fine-tune BLIP cho tiếng Việt
- Auto split 80/20
- Tối ưu cho macOS

### 6. Evaluation (`tools/evaluate_metrics.py`)
- BLEU, ROUGE-L, SBERT metrics
- Đánh giá tự động trên test set

---

## 💡 ĐIỂM NỔI BẬT

### ✅ Điểm mạnh
1. Pipeline 2 giai đoạn hiệu quả
2. Tối ưu cho macOS M1/M2/M3
3. API hoàn chỉnh với caching, rate limiting
4. Code structure rõ ràng
5. Evaluation metrics đầy đủ

### ⚠️ Hạn chế
1. BLEU score thấp (do caption gốc dài, SEO)
2. Inference time ~1s/ảnh
3. Tokenizer BLIP không hỗ trợ trực tiếp tiếng Việt
4. Memory usage cao trên MPS

---

## 🔧 GIẢI PHÁP KỸ THUẬT

### 1. MPS Compatibility
**Vấn đề**: MPS không hỗ trợ tốt attention_mask auto-inference

**Giải pháp**: Chuyển model về CPU khi generate, sau đó chuyển lại MPS

### 2. Memory Management
**Vấn đề**: Memory leak trên MPS

**Giải pháp**: 
- Cleanup tensors sau mỗi inference
- Clear cache mỗi 5 ảnh trong batch
- Synchronize device sau mỗi operation

### 3. Tokenizer Compatibility
**Vấn đề**: BLIP dùng BPE tokenizer tiếng Anh

**Giải pháp**: Pipeline 2 giai đoạn (BLIP không dấu + Accent Restoration)

---

## 📈 KẾT QUẢ ĐÁNH GIÁ

### Test Set (20% - 1,528 samples)
- **100% hoàn tất** inference
- **SBERT 0.6330** - Mô hình hiểu nghĩa tốt
- **BLEU thấp** là bình thường (caption gốc dài, SEO)

### Nhận xét
- Caption sinh ra **ngắn gọn, rõ nghĩa**
- **Không nhiễu ký tự**
- **Phục hồi dấu chính xác** (97%+)

---

## 📝 CẤU TRÚC BÁO CÁO ĐỀ XUẤT

1. **Chương 1: Giới thiệu**
   - Bài toán và mục tiêu
   - Phạm vi nghiên cứu

2. **Chương 2: Cơ sở lý thuyết**
   - BLIP model
   - Image Captioning
   - Accent Restoration
   - Evaluation Metrics

3. **Chương 3: Phân tích và thiết kế**
   - Kiến trúc hệ thống
   - Pipeline xử lý
   - API design

4. **Chương 4: Cài đặt và triển khai**
   - Dataset preparation
   - Model training
   - API implementation

5. **Chương 5: Đánh giá và kết quả**
   - Evaluation metrics
   - Kết quả thực nghiệm
   - Phân tích

6. **Chương 6: Kết luận**
   - Tổng kết
   - Hạn chế
   - Hướng phát triển

---

## 🎨 DIAGRAMS CẦN CÓ

1. **System Architecture Diagram**
2. **Data Flow Diagram**
3. **API Sequence Diagram**
4. **Model Architecture Diagram**

---

## 📚 TÀI LIỆU THAM KHẢO

1. BLIP: "Bootstrapping Language-Image Pre-training" (Salesforce)
2. Vietnamese Accent Marker (peterhung)
3. FastAPI Documentation
4. PyTorch MPS Backend
5. Evaluation Metrics (BLEU, ROUGE, SBERT)

---

## ✅ CHECKLIST

### Nội dung:
- [x] Dataset: 7,638 samples (80/20 split)
- [x] Training: 5 epochs, batch size 2
- [x] Evaluation: BLEU, ROUGE-L, SBERT
- [x] API: 7 endpoints hoàn chỉnh
- [x] Accent Restoration: 97%+ accuracy

### Code:
- [x] Model loader
- [x] Caption service
- [x] API routes
- [x] Training script
- [x] Evaluation script

### Documentation:
- [x] README.md
- [x] BAO_CAO_CHUYEN_DE.md
- [x] PHAN_TICH_CHI_TIET.md
- [x] TOM_TAT_BAO_CAO.md (file này)

---

**Tài liệu này tóm tắt các thông tin quan trọng cho báo cáo chuyên đề. Tham khảo `PHAN_TICH_CHI_TIET.md` để có phân tích chi tiết hơn.**

