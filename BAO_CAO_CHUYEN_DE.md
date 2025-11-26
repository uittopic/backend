# 📄 BÁO CÁO CHUYÊN ĐỀ — ỨNG DỤNG AI NHẬN DIỆN VÀ TRA CỨU SẢN PHẨM

**Giảng viên hướng dẫn:** Thầy Cáp Phạm Đình Thăng

**Nhóm thực hiện:**
- Nguyễn Hữu Việt — 24410375
- Nguyễn Ngọc Tuyên — 24410371

---

## 1. Giới thiệu bài toán

### 1.1. Đầu vào bài toán là gì?

Ảnh sản phẩm từ Shopee (có trong dataset thầy cung cấp).

Mỗi ảnh đi kèm caption mô tả gốc (tiếng Anh hoặc mô tả dạng SEO).

### 1.2. Đầu ra bài toán là gì?

Sinh mô tả tiếng Việt có dấu, rõ ràng, tự nhiên, ngắn gọn.

Phục vụ cho việc tra cứu – tìm kiếm sản phẩm.

### 1.3. Mục tiêu

- Fine-tune mô hình BLIP để sinh caption tiếng Việt không dấu.
- Chuyển caption không dấu → có dấu bằng Accent Restoration.
- Xây dựng API prototype để demo.
- Đánh giá mô hình theo yêu cầu thầy: train 80% – test 20%.

---

## 2. Chuẩn bị dữ liệu

Dataset thầy cung cấp gồm:

| File | Ý nghĩa |
|------|---------|
| `train_images/` | Ảnh dùng để train |
| `train.csv` | Caption gốc |
| `test_images/` | Ảnh dùng để test |
| `test.csv` | Caption mẫu |
| `sample_submission.csv` | File submit mẫu |

**Tổng số mẫu:** 7,638.

---

## 3. Xử lý dữ liệu & Tạo dataset tiếng Việt

### 3.1. Sinh caption tiếng Anh (tự động)

Dùng BLIP Pretrained → tạo caption tiếng Anh cho từng ảnh.

### 3.2. Dịch sang tiếng Việt

Dùng mô hình ViT5 dịch EN → VI.

### 3.3. Làm sạch dữ liệu

Loại: lỗi, lặp, câu sai ngữ nghĩa.

**Dataset cuối cùng:**

📌 `train_bilingual_clean_v2.csv` — 7,638 hàng.

---

## 4. Fine-tune BLIP để sinh tiếng Việt không dấu (80%)

### 4.1. Thiết lập train

**Script:** `train/train_blip_vietnamese.py`

**Chia dataset:**
- **Train:** 6,110 (80%)
- **Test:** 1,528 (20%)

**Tham số:**
- **Epoch:** 5
- **LR:** 5e-5
- **Batch size:** 2

**Thời gian train:** ~3–4 giờ (Macbook M1 Pro Max)

### 4.2. Kết quả

BLIP sau fine-tune cho ra caption:

- Ngắn gọn
- Không dấu
- Rõ nghĩa
- Không nhiễu ký tự

**Ví dụ:**
- `ao khoac the thao nu`
- `giay the thao mau trang`

---

## 5. Thử nghiệm dùng tokenizer tiếng Việt (BARTpho, PhoBERT, ViT5)

Làm theo yêu cầu thầy — thử các hướng nâng cấp mô hình.

### 5.1. Kết quả

Không khả thi, vì:

- Mô hình BLIP dùng tokenizer BPE tiếng Anh → Không tương thích.

**Caption sinh bị:**
- Lặp từ
- Lỗi ký tự
- Pha tiếng Anh
- Token hóa sai

→ Nhóm ngưng hướng này theo đúng đánh giá chuyên môn.

---

## 6. Giải pháp cuối cùng (Tối ưu & hiệu quả nhất)

### 6.1. Pipeline 2 giai đoạn

**Giai đoạn 1:**
- BLIP sinh caption tiếng Việt không dấu.

**Giai đoạn 2:**
- Accent Restoration phục hồi dấu tiếng Việt.

**Mô hình dùng:**
- `peterhung/vietnamese-accent-marker-xlm-roberta`

**Ưu điểm:**
- Accuracy 97%+
- Nhanh
- Nhẹ
- Không lỗi ký tự
- Tương thích hoàn hảo với output BLIP

### 6.2. Pipeline tổng quát

```
Ảnh → BLIP (caption không dấu) → Accent Restoration → Caption Tiếng Việt có dấu
```

**Ví dụ:**

- **Input ảnh:** túi xách
- **BLIP:** `tui xach phu nu thoi trang cao cap`
- **Accent:** `túi xách phụ nữ thời trang cao cấp`

---

## 7. Xây dựng API cho mô hình

### 7.1. REST API

- **POST `/api/caption_no_accent`** → sinh caption không dấu
- **POST `/api/accent/restore`** → chuyển sang có dấu
- **POST `/api/caption_full`** → pipeline đầy đủ (ảnh → caption VI có dấu)

### 7.2. Tối ưu

- Caching
- Không load model lại
- **Inference time:** 0.8–1.2s/ảnh

---

## 8. Train 80% – Test 20% theo yêu cầu thầy

### 8.1. Tình trạng thực hiện

| Nhiệm vụ | Trạng thái |
|----------|------------|
| Train 80% | ✔ Xong |
| Build API prototype | ✔ Xong |
| Chạy test 20% | ✔ Xong |
| Tính metrics | ✔ Xong |
| So sánh – nhận xét kết quả | ✔ Xong |

### 8.2. Kết quả test 20%

**Số lượng:**
- **Test input:** 1,528 ảnh
- **Prediction output:** 1,528 ảnh
- → **100% hoàn tất.**

**Metrics:**

| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **BLEU** | 0.0141 | Thấp — do caption gốc Shopee dài dạng SEO |
| **ROUGE-L** | 0.1486 | Trung bình |
| **SBERT** | 0.6330 | **Khá** — mô hình hiểu nghĩa tốt |

📝 **SBERT mới là metric quan trọng**, vì caption sinh không trùng từ, mà đánh giá mức độ hiểu nghĩa → đạt 0.63 là ổn.

### 8.3. Nhận xét

- BLEU thấp là bình thường, không phải lỗi mô hình.
- Caption của Shopee dài, nhiều SEO keyword → mô hình không cần bắt chước.
- Mục tiêu: mô tả đúng nghĩa sản phẩm, và điều này đã đạt.

---

## 9. Một số ví dụ minh họa

**Ảnh:** nước hoa

- **Không dấu:** `nuoc hoa nu`
- **Có dấu:** `nước hoa nữ`

**Ảnh:** giày sneaker đỏ

- **Không dấu:** `giay sneaker mau do`
- **Có dấu:** `giày sneaker màu đỏ`

---

## 10. Kết luận dự án

### 10.1. Những phần đã hoàn thành

- ✔ Tiền xử lý dataset tiếng Việt
- ✔ Train BLIP 80%
- ✔ Accent Restoration 97%+
- ✔ Xây dựng API hoàn chỉnh
- ✔ Test 20% + metrics
- ✔ So sánh – đánh giá – phân tích
- ✔ Chuẩn bị slide báo cáo

### 10.2. Hướng phát triển thêm (nếu thầy yêu cầu)

- Tăng thêm epoch để tăng SBERT
- Làm demo UI web
- Deploy server thật

### 10.3. Hạn chế

- **BLEU thấp** (do mục tiêu khác caption gốc)
- **Tokenizer BLIP** không hỗ trợ trực tiếp tiếng Việt
- **Inference** ~1s/ảnh (có thể tối ưu bằng batch)

---

## 11. Tài liệu tham khảo

1. Salesforce BLIP
2. HuggingFace Transformers
3. Vietnamese Accent Marker
4. Shopee Product Matching Dataset
5. Tài liệu giai đoạn thầy cung cấp

---

**Hết báo cáo**
