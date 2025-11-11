# 📖 Hướng dẫn Training - Giải thích chi tiết

## 🔍 Bước 1: Kiểm tra số lượng ảnh

```bash
ls data/images/ | wc -l
```

**Mục đích:**
- Đếm số file ảnh trong thư mục `data/images/`
- So sánh với số dòng trong CSV (7638 samples)

**Tại sao quan trọng?**
- CSV có 7638 dòng → nên có ~7638 ảnh
- Nếu thiếu ảnh → training sẽ bỏ qua những dòng không có ảnh
- Nếu quá nhiều ảnh → không sao, nhưng cần đảm bảo tên file khớp với CSV

**Kết quả mong đợi:**
```
7638
```
(Hoặc gần bằng, có thể ít hơn một chút nếu có ảnh lỗi)

---

## 🚀 Bước 2: Kích hoạt Virtual Environment

```bash
source venv/bin/activate
```

**Mục đích:**
- Kích hoạt môi trường Python ảo (virtual environment)
- Đảm bảo sử dụng đúng Python và packages đã cài

**Tại sao cần?**
- Tránh xung đột với Python system
- Đảm bảo có đủ packages (transformers, torch, etc.)
- Tách biệt dependencies của project này với project khác

**Kết quả:**
- Terminal sẽ hiển thị `(venv)` ở đầu dòng:
```
(venv) user@macbook train %
```

---

## 📂 Bước 3: Di chuyển vào thư mục train

```bash
cd train
```

**Mục đích:**
- Di chuyển vào thư mục chứa script training
- Để chạy file `train_blip_vietnamese.py`

---

## 🏋️ Bước 4: Chạy Training Script

```bash
python train_blip_vietnamese.py
```

**Mục đích:**
- Bắt đầu quá trình fine-tune BLIP model cho tiếng Việt
- Model sẽ học từ dataset ảnh + caption tiếng Việt

**Quá trình sẽ diễn ra:**

1. **Load dataset** (vài phút)
   ```
   📂 Đang load dataset...
   ✅ Đã load 7638 samples
   ```

2. **Load pretrained model** (vài phút)
   ```
   🤖 Đang load BLIP model...
   📱 Device: mps
   ✅ Model đã được load
   ```

3. **Preprocess data** (5-15 phút)
   ```
   🔄 Đang preprocess train dataset...
   ✅ Train dataset: 6874 samples
   ```

4. **Training** (2-6 giờ trên M1 Pro Max)
   ```
   🚀 Bắt đầu training...
   Epoch 1/5: 100%|████████| 3437/3437 [45:23<00:00, loss=2.345]
   ...
   ```

5. **Lưu model** (vài phút)
   ```
   💾 Đang lưu model...
   ✅ Fine-tune BLIP Vietnamese hoàn thành!
   ```

**Thời gian ước tính:**
- M1 Pro Max: 2-6 giờ (tùy số lượng ảnh)
- CPU only: 8-12 giờ

---

## 📊 Bước 5: Kiểm tra Log và Loss (Sau khi train)

**Mục đích:**
- Xem model có học tốt không
- Kiểm tra loss có giảm dần không
- Đảm bảo không bị overfitting

**Các file log:**
- Training logs: `logs/` (nếu có)
- Console output: hiển thị loss sau mỗi epoch

**Loss mong đợi:**
- Epoch 1: Loss cao (~2.5-3.0)
- Epoch 2-3: Loss giảm dần (~1.5-2.0)
- Epoch 4-5: Loss ổn định (~1.0-1.5)

**Dấu hiệu tốt:**
- ✅ Loss giảm dần qua các epoch
- ✅ Validation loss không tăng (tránh overfitting)
- ✅ Model lưu thành công

**Dấu hiệu cần chú ý:**
- ⚠️ Loss không giảm → learning rate quá nhỏ hoặc data có vấn đề
- ⚠️ Loss tăng → learning rate quá lớn
- ⚠️ Validation loss tăng nhưng train loss giảm → overfitting

---

## 📝 Tóm tắt quy trình

```
1. Kiểm tra ảnh: ls data/images/ | wc -l
   → Đảm bảo có ~7638 ảnh

2. Kích hoạt venv: source venv/bin/activate
   → Terminal hiển thị (venv)

3. Vào thư mục train: cd train
   → Di chuyển vào thư mục train/

4. Chạy training: python train_blip_vietnamese.py
   → Chờ 2-6 giờ (M1 Pro Max)

5. Kiểm tra kết quả:
   → Xem loss trong console
   → Kiểm tra model trong models/blip_vietnamese/
```

---

## ⚠️ Lưu ý quan trọng

1. **Không tắt terminal** khi đang training
2. **Giữ máy không sleep** (Settings → Energy Saver)
3. **Đảm bảo đủ dung lượng** (~5-10GB cho model)
4. **Kiểm tra RAM** - nếu hết RAM, giảm batch size

---

## 🎯 Sau khi training xong

Model sẽ được lưu tại:
```
models/blip_vietnamese/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt
```

Sau đó có thể:
1. Chạy API: `uvicorn app.main:app --reload`
2. Test với Postman
3. Deploy lên server

