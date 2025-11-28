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

Trong thời đại công nghệ số hiện nay, thương mại điện tử đang phát triển mạnh mẽ với hàng triệu sản phẩm được đăng bán mỗi ngày trên các nền tảng như Shopee, Lazada, Tiki. Một trong những thách thức lớn nhất mà các sàn thương mại điện tử phải đối mặt là việc quản lý và tra cứu sản phẩm một cách hiệu quả. Hầu hết các sản phẩm được mô tả bằng ảnh, nhưng việc tìm kiếm và phân loại sản phẩm dựa trên ảnh vẫn còn nhiều hạn chế.

Vấn đề chính nằm ở việc thiếu mô tả văn bản tiếng Việt chính xác và tự nhiên cho các ảnh sản phẩm. Hiện tại, nhiều sản phẩm trên các sàn thương mại điện tử có caption gốc bằng tiếng Anh hoặc mô tả dạng SEO dài dòng, không phù hợp với nhu cầu của người dùng Việt Nam. Việc tạo caption tiếng Việt thủ công cho hàng nghìn sản phẩm là không khả thi về mặt thời gian và chi phí.

Hơn nữa, tiếng Việt là ngôn ngữ có dấu (accented language), điều này tạo ra thách thức bổ sung cho các mô hình AI. Nhiều mô hình image captioning hiện tại được thiết kế cho tiếng Anh và không hỗ trợ tốt tiếng Việt, đặc biệt là việc sinh caption có dấu chính xác. Việc tạo caption không dấu rồi sau đó phục hồi dấu là một giải pháp khả thi, nhưng đòi hỏi một pipeline xử lý phức tạp và chính xác.

Từ những vấn đề trên, chúng tôi nhận thấy nhu cầu cấp thiết phải xây dựng một hệ thống tự động tạo caption tiếng Việt có dấu, tự nhiên và ngắn gọn cho ảnh sản phẩm. Hệ thống này sẽ giúp:
- Tự động hóa quá trình tạo mô tả sản phẩm, tiết kiệm thời gian và chi phí
- Cải thiện trải nghiệm người dùng với caption tiếng Việt tự nhiên, dễ hiểu
- Hỗ trợ tra cứu và tìm kiếm sản phẩm hiệu quả hơn
- Mở rộng khả năng ứng dụng AI trong lĩnh vực thương mại điện tử tại Việt Nam

Với dataset gồm 7,638 ảnh sản phẩm từ Shopee được cung cấp, chúng tôi có cơ hội nghiên cứu và phát triển một giải pháp cụ thể cho bài toán này. Đây là một bài toán thực tế, có ý nghĩa và có thể ứng dụng ngay vào thực tế.

### 1.2. Mục tiêu nghiên cứu

#### 1.2.1. Mục tiêu chính

Dựa trên những vấn đề đã nêu, dự án này đặt ra các mục tiêu chính sau:

**Mục tiêu 1: Fine-tune mô hình BLIP cho tiếng Việt**
- Nghiên cứu và áp dụng mô hình BLIP (Bootstrapping Language-Image Pre-training) của Salesforce Research
- Fine-tune mô hình trên dataset tiếng Việt gồm 7,638 ảnh sản phẩm từ Shopee
- Đạt được khả năng sinh caption tiếng Việt không dấu, ngắn gọn và chính xác cho ảnh sản phẩm
- Chia dataset theo tỉ lệ 80% cho training và 20% cho testing theo yêu cầu

**Mục tiêu 2: Xây dựng pipeline phục hồi dấu tiếng Việt**
- Tích hợp mô hình Accent Restoration để chuyển đổi caption không dấu sang có dấu
- Sử dụng mô hình `peterhung/vietnamese-accent-marker-xlm-roberta` với độ chính xác cao
- Đảm bảo caption cuối cùng có dấu chính xác, tự nhiên và dễ đọc

**Mục tiêu 3: Xây dựng API prototype**
- Phát triển REST API sử dụng FastAPI framework
- Cung cấp các endpoint để sinh caption cho ảnh đơn lẻ và batch
- Tích hợp các tính năng: caching, rate limiting, error handling
- Tạo tài liệu API tự động với Swagger UI

**Mục tiêu 4: Đánh giá và phân tích kết quả**
- Đánh giá mô hình trên test set 20% (1,528 samples)
- Sử dụng nhiều metrics: BLEU, ROUGE-L, và SBERT Similarity
- Phân tích và so sánh kết quả với baseline
- Đưa ra nhận xét và đánh giá về hiệu quả của giải pháp

#### 1.2.2. Mục tiêu phụ

Ngoài các mục tiêu chính, dự án còn đặt ra các mục tiêu phụ nhằm nâng cao chất lượng và khả năng ứng dụng:

- **Tối ưu hóa cho macOS**: Tối ưu code và model để chạy hiệu quả trên macOS với Apple Silicon (M1/M2/M3), sử dụng MPS (Metal Performance Shaders) backend
- **Xây dựng hệ thống caching**: Implement caching mechanism để tăng tốc độ xử lý và giảm tải cho model
- **Rate limiting**: Bảo vệ API khỏi quá tải với rate limiting
- **Đánh giá đa metrics**: Sử dụng nhiều metrics khác nhau để đánh giá toàn diện chất lượng caption
- **Code quality**: Viết code rõ ràng, có cấu trúc, dễ maintain và mở rộng

### 1.3. Phạm vi nghiên cứu

#### 1.3.1. Phạm vi nghiên cứu

Dự án này tập trung vào các phạm vi sau:

**Về dữ liệu:**
- Dataset: 7,638 ảnh sản phẩm từ Shopee được cung cấp
- Format: CSV file chứa đường dẫn ảnh và caption tiếng Việt
- Loại ảnh: Ảnh sản phẩm thương mại điện tử (quần áo, giày dép, phụ kiện, v.v.)
- Chia dataset: 80% cho training (6,110 samples) và 20% cho testing (1,528 samples)

**Về mô hình:**
- Base model: BLIP (Salesforce/blip-image-captioning-base) - mô hình pre-trained trên 129M ảnh-caption pairs
- Accent Restoration: `peterhung/vietnamese-accent-marker-xlm-roberta` - mô hình Token Classification cho phục hồi dấu tiếng Việt
- Fine-tuning: Fine-tune BLIP trên dataset tiếng Việt với 5 epochs

**Về nền tảng:**
- Hệ điều hành: macOS với Apple Silicon (M1/M2/M3)
- Backend: MPS (Metal Performance Shaders) cho GPU acceleration
- Framework: FastAPI cho REST API, PyTorch cho deep learning
- Ngôn ngữ lập trình: Python 3.8+

**Về đánh giá:**
- Test set: 1,528 samples (20% của dataset)
- Metrics: BLEU Score, ROUGE-L F1, SBERT Similarity
- So sánh với: Pretrained BLIP (baseline) và caption gốc từ Shopee

#### 1.3.2. Giới hạn nghiên cứu

Dự án có các giới hạn sau:

**Về loại ảnh:**
- Chỉ xử lý ảnh sản phẩm thương mại điện tử, không phải ảnh tổng quát
- Không xử lý ảnh có nhiều đối tượng phức tạp hoặc ảnh nghệ thuật
- Ảnh đầu vào phải có chất lượng tối thiểu và rõ ràng

**Về độ dài caption:**
- Caption sinh ra có độ dài tối đa 50 tokens
- Caption ngắn gọn, tập trung vào mô tả sản phẩm chính
- Không bao gồm thông tin chi tiết như giá cả, thông số kỹ thuật

**Về ngôn ngữ:**
- Chỉ hỗ trợ tiếng Việt có dấu
- Không hỗ trợ đa ngôn ngữ hoặc tiếng Việt không dấu (sau khi restore accent)
- Không xử lý các từ viết tắt hoặc thuật ngữ chuyên ngành đặc biệt

**Về hiệu năng:**
- Inference time: ~0.8-1.2 giây mỗi ảnh (có thể tối ưu thêm)
- Batch processing: Tối đa 10 ảnh mỗi request
- Memory usage: Phụ thuộc vào kích thước ảnh và batch size

**Về đánh giá:**
- Metrics chủ yếu dựa trên semantic similarity (SBERT) vì caption gốc từ Shopee có format SEO dài, khác với caption sinh ra
- Không so sánh trực tiếp với các mô hình image captioning khác do thiếu dataset benchmark tiếng Việt

### 1.4. Cấu trúc báo cáo

Báo cáo được chia thành 6 chương chính và các phần phụ lục, cụ thể như sau:

**Chương 1: Giới thiệu**
- Đặt vấn đề, mục tiêu nghiên cứu, phạm vi và giới hạn nghiên cứu
- Cấu trúc báo cáo

**Chương 2: Cơ sở lý thuyết**
- Tổng quan về Image Captioning và các phương pháp hiện tại
- Giới thiệu chi tiết về mô hình BLIP: kiến trúc, cơ chế hoạt động, và khả năng ứng dụng
- Lý thuyết về Accent Restoration cho tiếng Việt
- Các metrics đánh giá: BLEU, ROUGE-L, và SBERT Similarity

**Chương 3: Phân tích và thiết kế hệ thống**
- Kiến trúc tổng thể của hệ thống
- Pipeline xử lý từ training đến inference
- Thiết kế API và các endpoint
- Cơ chế caching và rate limiting
- Xử lý lỗi và edge cases

**Chương 4: Cài đặt và triển khai**
- Quy trình chuẩn bị và tiền xử lý dataset
- Quá trình fine-tuning BLIP model với các tham số cụ thể
- Triển khai API với FastAPI
- Các kỹ thuật tối ưu hóa cho macOS và MPS backend
- Giải quyết các vấn đề kỹ thuật trong quá trình phát triển

**Chương 5: Đánh giá và kết quả**
- Setup đánh giá và các metrics sử dụng
- Kết quả thực nghiệm trên test set 20%
- Phân tích chi tiết từng metric và so sánh với baseline
- Ví dụ minh họa caption sinh ra
- Đánh giá hiệu năng và tốc độ xử lý

**Chương 6: Kết luận và hướng phát triển**
- Tổng kết những gì đã đạt được
- Nhận xét về hạn chế và thách thức
- Đề xuất hướng phát triển trong tương lai
- Đóng góp của dự án và khả năng ứng dụng thực tế

**Phần phụ lục:**
- Code snippets quan trọng
- Tài liệu API đầy đủ
- Kết quả đánh giá chi tiết
- Thống kê dataset

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Image Captioning

#### 2.1.1. Định nghĩa và Tổng quan

Image Captioning là một nhiệm vụ đa phương thức (multimodal) trong lĩnh vực trí tuệ nhân tạo, kết hợp giữa Computer Vision và Natural Language Processing. Nhiệm vụ này yêu cầu hệ thống tự động tạo ra một câu mô tả văn bản chính xác và tự nhiên cho một hình ảnh đầu vào. Khác với Image Classification (phân loại ảnh) hay Object Detection (phát hiện đối tượng), Image Captioning không chỉ nhận diện các đối tượng trong ảnh mà còn phải hiểu được mối quan hệ giữa chúng và diễn đạt bằng ngôn ngữ tự nhiên.

Bài toán Image Captioning có thể được mô tả như sau: Cho một hình ảnh I, tìm một chuỗi từ W = {w₁, w₂, ..., wₙ} sao cho W mô tả chính xác nội dung của I. Đây là một bài toán khó vì nó đòi hỏi hệ thống phải:
- Hiểu được nội dung hình ảnh (Computer Vision)
- Tạo ra câu văn có ngữ pháp và ngữ nghĩa đúng (Natural Language Processing)
- Kết hợp hai domain này một cách hiệu quả

#### 2.1.2. Lịch sử phát triển

Image Captioning đã trải qua nhiều giai đoạn phát triển:

**Giai đoạn 1: Template-based (2010-2014)**
- Sử dụng template cố định với các slot được điền bởi kết quả từ object detection
- Ví dụ: "A [object] is [action] in [location]"
- Hạn chế: Cứng nhắc, không tự nhiên

**Giai đoạn 2: Encoder-Decoder với RNN (2014-2017)**
- Encoder: CNN (VGG, ResNet) để trích xuất features từ ảnh
- Decoder: RNN/LSTM để sinh caption từ features
- Đột phá: Show and Tell (2014), Show, Attend and Tell (2015)
- Hạn chế: Vanishing gradient, khó xử lý long-range dependencies

**Giai đoạn 3: Attention Mechanism (2015-2019)**
- Thêm attention mechanism để decoder "nhìn" vào các vùng khác nhau của ảnh khi sinh từng từ
- Cải thiện đáng kể chất lượng caption
- Ví dụ: Show, Attend and Tell, Bottom-Up and Top-Down Attention

**Giai đoạn 4: Transformer-based (2019-nay)**
- Áp dụng Transformer architecture cho cả vision và language
- Vision Transformer (ViT) thay thế CNN
- Cross-modal attention để kết hợp thông tin ảnh và text
- Ví dụ: BLIP, CLIP, Flamingo, GPT-4V

#### 2.1.3. Kiến trúc cơ bản

Hầu hết các mô hình Image Captioning hiện đại đều tuân theo kiến trúc Encoder-Decoder:

**Encoder (Vision Encoder)**:
- Nhiệm vụ: Trích xuất đặc trưng từ hình ảnh
- Input: Hình ảnh I (thường là 224×224 hoặc 384×384 pixels)
- Output: Feature representation F = {f₁, f₂, ..., fₖ}
- Các phương pháp:
  - CNN-based: VGG, ResNet, EfficientNet
  - Transformer-based: ViT (Vision Transformer), Swin Transformer

**Decoder (Text Decoder)**:
- Nhiệm vụ: Sinh caption từ feature representation
- Input: Feature F từ encoder
- Output: Chuỗi từ W = {w₁, w₂, ..., wₙ}
- Các phương pháp:
  - RNN/LSTM/GRU
  - Transformer Decoder
  - GPT-style autoregressive generation

**Attention Mechanism**:
- Cho phép decoder "tập trung" vào các vùng khác nhau của ảnh khi sinh từng từ
- Công thức cơ bản:
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) × V
  ```
- Trong đó:
  - Q (Query): Từ decoder hiện tại
  - K, V (Key, Value): Features từ encoder
  - d_k: Dimension của key

#### 2.1.4. Ứng dụng thực tế

Image Captioning có nhiều ứng dụng quan trọng:

**Accessibility (Khả năng tiếp cận)**:
- Mô tả ảnh cho người khiếm thị qua screen reader
- Giúp người khiếm thị hiểu được nội dung ảnh trên web, social media

**E-commerce (Thương mại điện tử)**:
- Tự động tạo mô tả sản phẩm từ ảnh
- Cải thiện SEO và trải nghiệm người dùng
- Hỗ trợ tìm kiếm sản phẩm bằng hình ảnh

**Content Generation (Tạo nội dung)**:
- Tự động tạo caption cho ảnh trên social media
- Hỗ trợ nhà báo, blogger tạo mô tả ảnh nhanh chóng

**Image Search (Tìm kiếm ảnh)**:
- Chuyển đổi ảnh thành text để tìm kiếm dễ dàng hơn
- Hỗ trợ tìm kiếm semantic (theo nghĩa) thay vì chỉ theo từ khóa

### 2.2. BLIP Model

#### 2.2.1. Giới thiệu

BLIP (Bootstrapping Language-Image Pre-training) là một mô hình vision-language được phát triển bởi Salesforce Research và công bố vào năm 2022. BLIP được thiết kế để giải quyết vấn đề "noise" trong dữ liệu web-scale (như các caption tự động từ web thường không chính xác) bằng cách sử dụng một phương pháp bootstrapping để tạo ra dữ liệu training chất lượng cao.

**Đặc điểm nổi bật của BLIP**:
- Pre-trained trên 129 triệu ảnh-caption pairs từ web
- Hỗ trợ cả understanding (hiểu) và generation (sinh) tasks
- Có khả năng filter và tạo ra caption chất lượng cao từ dữ liệu noisy
- Đạt state-of-the-art trên nhiều benchmarks

**Các phiên bản BLIP**:
- **BLIP**: Base model với 224M parameters
- **BLIP-Large**: Large model với 990M parameters
- **BLIP-2**: Phiên bản cải tiến với Q-Former và frozen image encoder

#### 2.2.2. Kiến trúc

BLIP sử dụng kiến trúc multimodal encoder-decoder với 3 thành phần chính:

**1. Vision Encoder (ViT - Vision Transformer)**:
- Base: ViT-B/16 (Vision Transformer với patch size 16×16)
- Input: Ảnh được chia thành patches 16×16
- Process:
  - Linear projection của mỗi patch thành embedding
  - Thêm positional encoding
  - Qua các Transformer layers để tạo image features
- Output: Sequence of image features F_img = {f₁, f₂, ..., fₙ}

**2. Text Encoder (BERT-based)**:
- Base: BERT architecture
- Nhiệm vụ: Encode text input (cho understanding tasks)
- Sử dụng: Image-Text Retrieval, Image-Text Matching

**3. Text Decoder (BERT-based với Causal Masking)**:
- Base: BERT architecture nhưng với causal masking (chỉ nhìn được các từ trước đó)
- Nhiệm vụ: Generate caption từ image features
- Sử dụng: Image Captioning

**4. Cross-modal Attention**:
- Kết nối giữa Vision Encoder và Text Encoder/Decoder
- Cho phép text "nhìn" vào image features khi encode/decode
- Công thức:
  ```
  CrossAttention(Q_text, K_img, V_img) = softmax(Q_text × K_img^T / √d) × V_img
  ```

**Sơ đồ kiến trúc tổng quát**:
```
Input Image
    ↓
[Vision Encoder (ViT)]
    ↓ Image Features
[Cross-Modal Attention] ← Text Input (cho Encoder)
    ↓
[Text Encoder/Decoder (BERT)]
    ↓
Output: Caption
```

#### 2.2.3. Pre-training Strategy

BLIP sử dụng 3 objectives trong quá trình pre-training:

**1. Image-Text Contrastive Learning (ITC)**:
- Mục tiêu: Học alignment giữa image và text
- Cách hoạt động: Pull positive pairs (ảnh-caption đúng) lại gần nhau, push negative pairs ra xa
- Loss: Contrastive loss (InfoNCE)

**2. Image-Text Matching (ITM)**:
- Mục tiêu: Học fine-grained alignment
- Cách hoạt động: Binary classification - ảnh và text có match không?
- Loss: Binary cross-entropy

**3. Image-grounded Text Generation (ITG)**:
- Mục tiêu: Học generate caption từ ảnh
- Cách hoạt động: Given image, generate caption (autoregressive)
- Loss: Cross-entropy cho từng token

**Bootstrapping Capability**:
- BLIP có thể filter noisy captions từ web
- Tạo ra synthetic captions chất lượng cao
- Sử dụng chính model để cải thiện dữ liệu training

#### 2.2.4. Fine-tuning cho Image Captioning

Khi fine-tune BLIP cho image captioning:

**Input**:
- Image: Ảnh sản phẩm (resize về 224×224 hoặc 384×384)
- Caption: Caption tiếng Việt không dấu (ground truth)

**Process**:
1. Image qua Vision Encoder → Image features
2. Image features + Text prefix qua Text Decoder
3. Decoder sinh caption token by token (autoregressive)
4. Tính loss với ground truth caption

**Training Objective**:
```
Loss = -Σ log P(w_i | image, w_<i)
```
- w_i: Token thứ i
- w_<i: Các token trước đó

**Generation Parameters**:
- `max_new_tokens`: Số token tối đa (thường 50-77)
- `num_beams`: Số beams cho beam search (thường 3-5)
- `repetition_penalty`: Penalty cho lặp từ (thường 1.2)
- `length_penalty`: Khuyến khích độ dài (thường 1.0-1.2)

#### 2.2.5. Ưu điểm của BLIP

1. **Unified Architecture**: Một model cho nhiều tasks (captioning, retrieval, VQA)
2. **Bootstrapping**: Tự cải thiện dữ liệu training
3. **Efficient**: Tốc độ inference nhanh hơn so với các model lớn khác
4. **Flexible**: Dễ fine-tune cho các ngôn ngữ khác (như tiếng Việt)
5. **State-of-the-art**: Đạt kết quả tốt trên nhiều benchmarks

#### 2.2.6. Hạn chế

1. **Tokenizer**: Sử dụng BPE tokenizer tiếng Anh, không tối ưu cho tiếng Việt
2. **Vocabulary**: Vocabulary chủ yếu là tiếng Anh, cần fine-tune để học từ tiếng Việt
3. **Caption không dấu**: Khi fine-tune trên tiếng Việt, model thường sinh caption không dấu (do tokenizer không hỗ trợ tốt dấu tiếng Việt)

### 2.3. Accent Restoration cho Tiếng Việt

#### 2.3.1. Vấn đề

Tiếng Việt là một ngôn ngữ có dấu (accented language) với 5 loại dấu: sắc (´), huyền (`), hỏi (?), ngã (~), và nặng (.). Việc thiếu dấu có thể làm thay đổi hoàn toàn nghĩa của từ. Ví dụ:
- "ma" (ma quỷ) vs "má" (mẹ)
- "ban" (ban hành) vs "bàn" (cái bàn)

Khi fine-tune BLIP cho tiếng Việt, do tokenizer BPE được thiết kế cho tiếng Anh, model thường sinh ra caption không dấu. Điều này gây khó khăn cho người đọc và làm giảm chất lượng caption.

#### 2.3.2. Giải pháp: Accent Restoration

Accent Restoration (Phục hồi dấu) là nhiệm vụ tự động thêm dấu vào text tiếng Việt không dấu. Đây là một bài toán Token Classification trong NLP.

**Mô hình sử dụng**: `peterhung/vietnamese-accent-marker-xlm-roberta`

**Kiến trúc**:
- Base model: XLM-RoBERTa (Cross-lingual Language Model)
- Task: Token Classification
- Input: Text tiếng Việt không dấu (word-level hoặc subword-level)
- Output: Accent labels cho mỗi token

#### 2.3.3. XLM-RoBERTa

XLM-RoBERTa (Cross-lingual Language Model - RoBERTa) là một mô hình ngôn ngữ đa ngôn ngữ được pre-trained trên 100 ngôn ngữ, bao gồm tiếng Việt. Nó là phiên bản đa ngôn ngữ của RoBERTa.

**Đặc điểm**:
- Pre-trained trên dữ liệu đa ngôn ngữ
- Sử dụng SentencePiece tokenization với subword units
- Có prefix "▁" để đánh dấu đầu từ
- Hỗ trợ tốt tiếng Việt

#### 2.3.4. Cơ chế hoạt động

**Bước 1: Tokenization**
- Input: "ao khoac the thao nu mau den"
- Tokenize với XLM-RoBERTa tokenizer:
  ```
  ["▁ao", "▁khoac", "▁the", "▁thao", "▁nu", "▁mau", "▁den"]
  ```
- Prefix "▁" đánh dấu đây là đầu từ

**Bước 2: Model Prediction**
- Mỗi token được đưa qua XLM-RoBERTa
- Model predict một label (số nguyên) cho mỗi token
- Label này tương ứng với một accent pattern trong vocabulary

**Bước 3: Label Mapping**
- Labels được map sang accent patterns
- Format: "raw-vowel" (ví dụ: "ao-áo", "khoac-khoác")
- Nếu label không match, giữ nguyên token gốc

**Bước 4: Merge Tokens**
- Merge các subword tokens có cùng prefix "▁" thành một từ
- Ví dụ: ["▁kho", "ac"] → "khoac" → "khoác"

**Bước 5: Join Words**
- Join các từ đã được restore accent thành câu hoàn chỉnh
- Output: "áo khoác thể thao nữ màu đen"

**Pipeline chi tiết**:
```
Text không dấu: "ao khoac the thao nu mau den"
    ↓
Tokenize (XLM-RoBERTa)
    ↓
["▁ao", "▁khoac", "▁the", "▁thao", "▁nu", "▁mau", "▁den"]
    ↓
XLM-RoBERTa Token Classification
    ↓
Labels: [label_ao, label_khoac, label_the, ...]
    ↓
Map labels to accent patterns
    ↓
["áo", "khoác", "thể", "thao", "nữ", "màu", "đen"]
    ↓
Join
    ↓
"áo khoác thể thao nữ màu đen"
```

#### 2.3.5. Ưu điểm của phương pháp

1. **Accuracy cao**: Đạt 97%+ accuracy trên test set
2. **Nhanh**: Inference time ~0.1-0.2s cho một câu
3. **Nhẹ**: Model size nhỏ hơn nhiều so với các mô hình generation lớn
4. **Không lỗi ký tự**: Token Classification ít gây lỗi hơn so với generation
5. **Tương thích**: Hoạt động tốt với output từ BLIP (caption không dấu)

#### 2.3.6. Hạn chế

1. **Context-dependent**: Một số từ có thể có nhiều cách thêm dấu tùy ngữ cảnh
   - Ví dụ: "ban" có thể là "bàn" (cái bàn) hoặc "ban" (ban hành)
2. **Unknown words**: Từ mới hoặc từ ngoại lai có thể không được xử lý đúng
3. **Subword tokens**: Cần merge tokens đúng cách để tránh lỗi

### 2.4. Evaluation Metrics

Đánh giá chất lượng caption là một thách thức vì không có một metric "hoàn hảo" nào. Mỗi metric có ưu và nhược điểm riêng. Trong dự án này, chúng tôi sử dụng 3 metrics chính: BLEU, ROUGE-L, và SBERT Similarity.

#### 2.4.1. BLEU Score

**Định nghĩa**:
BLEU (Bilingual Evaluation Understudy) là một metric được phát triển để đánh giá chất lượng machine translation. Nó đo độ tương đồng n-gram giữa prediction và reference.

**Công thức chi tiết**:

1. **N-gram Precision**:
   ```
   P_n = (Số n-grams trong prediction xuất hiện trong reference) / (Tổng số n-grams trong prediction)
   ```

2. **Brevity Penalty (BP)**:
   ```
   BP = {
       1, nếu length(prediction) > length(reference)
       exp(1 - length(reference)/length(prediction)), nếu không
   }
   ```
   - BP phạt những prediction quá ngắn

3. **BLEU Score**:
   ```
   BLEU = BP × exp(Σ_{n=1}^N w_n × log(P_n))
   ```
   - N: Số n-gram tối đa (thường N=4)
   - w_n: Trọng số cho mỗi n-gram (thường đều nhau: w_n = 1/N)

**Ví dụ**:
- Reference: "áo khoác thể thao nữ màu đen"
- Prediction: "áo khoác thể thao nữ"
- 1-grams: "áo", "khoác", "thể", "thao", "nữ" → 5/5 = 1.0
- 2-grams: "áo khoác", "khoác thể", "thể thao", "thao nữ" → 4/4 = 1.0
- BP: exp(1 - 5/5) = 1.0
- BLEU ≈ 1.0

**Ưu điểm**:
- Đơn giản, dễ tính toán
- Phổ biến, được sử dụng rộng rãi
- Phù hợp khi prediction và reference có nhiều từ chung

**Hạn chế**:
- Chỉ đánh giá n-gram overlap, không đánh giá semantic similarity
- Thấp khi prediction và reference dùng từ khác nhau nhưng cùng nghĩa
- Phụ thuộc vào độ dài: prediction ngắn hơn reference thường có BLEU thấp
- Không phù hợp khi reference dài, nhiều từ thừa (như SEO keywords)

**Trong dự án này**:
- BLEU score thấp (0.0141) là bình thường vì:
  - Caption gốc từ Shopee dài, nhiều SEO keywords
  - Caption sinh ra ngắn gọn, tập trung vào mô tả chính
  - Mục tiêu khác nhau: SEO vs mô tả tự nhiên

#### 2.4.2. ROUGE-L

**Định nghĩa**:
ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence) đo độ tương đồng dựa trên Longest Common Subsequence (LCS) giữa prediction và reference.

**Longest Common Subsequence (LCS)**:
- LCS là chuỗi con dài nhất có thể tìm được trong cả prediction và reference
- Khác với Longest Common Substring, LCS không yêu cầu các phần tử liên tiếp

**Ví dụ**:
- Reference: "áo khoác thể thao nữ màu đen"
- Prediction: "áo khoác thể thao nữ"
- LCS: "áo khoác thể thao nữ" (độ dài 5)

**Công thức**:

1. **Precision**:
   ```
   P_LCS = |LCS(prediction, reference)| / |prediction|
   ```

2. **Recall**:
   ```
   R_LCS = |LCS(prediction, reference)| / |reference|
   ```

3. **F1 Score**:
   ```
   ROUGE-L = F1_LCS = 2 × (P_LCS × R_LCS) / (P_LCS + R_LCS)
   ```

**Ví dụ tính toán**:
- Reference: "áo khoác thể thao nữ màu đen" (6 từ)
- Prediction: "áo khoác thể thao nữ" (5 từ)
- LCS: "áo khoác thể thao nữ" (5 từ)
- P_LCS = 5/5 = 1.0
- R_LCS = 5/6 = 0.833
- ROUGE-L = 2 × (1.0 × 0.833) / (1.0 + 0.833) = 0.909

**Ưu điểm**:
- Đánh giá cấu trúc câu, không chỉ n-gram
- Không phụ thuộc vào thứ tự từ (LCS có thể không liên tiếp)
- Phù hợp khi prediction và reference có cấu trúc tương tự

**Hạn chế**:
- Vẫn dựa trên từ overlap, không đánh giá semantic
- Có thể cao ngay cả khi prediction và reference dùng từ khác nhau nhưng có cấu trúc tương tự

**Trong dự án này**:
- ROUGE-L = 0.1486 (trung bình)
- Phản ánh một phần cấu trúc chung giữa prediction và reference
- Thấp hơn SBERT vì caption gốc có nhiều từ SEO không có trong prediction

#### 2.4.3. SBERT Similarity

**Định nghĩa**:
SBERT (Sentence-BERT) Similarity đo semantic similarity (độ tương đồng về nghĩa) giữa prediction và reference bằng cách so sánh embeddings của chúng.

**Sentence-BERT**:
- Sentence-BERT là một biến thể của BERT được fine-tune để tạo ra sentence embeddings
- Sử dụng Siamese network architecture
- Embeddings có thể được so sánh bằng cosine similarity

**Cơ chế hoạt động**:

1. **Encode sentences**:
   ```
   emb_pred = SBERT_model.encode(prediction)
   emb_ref = SBERT_model.encode(reference)
   ```

2. **Cosine Similarity**:
   ```
   cosine_sim = (emb_pred · emb_ref) / (||emb_pred|| × ||emb_ref||)
   ```
   - Range: [-1, 1]
   - 1: Hoàn toàn giống nhau
   - 0: Không liên quan
   - -1: Đối lập hoàn toàn

3. **Normalize về [0, 1]**:
   ```
   similarity = (cosine_sim + 1) / 2
   ```
   - Range: [0, 1]
   - 1: Hoàn toàn giống nhau về nghĩa
   - 0: Không liên quan

**Model sử dụng**: `keepitreal/vietnamese-sbert`
- Model được fine-tune cho tiếng Việt
- Pre-trained trên dữ liệu tiếng Việt
- Đạt kết quả tốt trong các tasks semantic similarity

**Ví dụ**:
- Reference: "áo khoác thể thao nữ màu đen chất lượng cao giá rẻ"
- Prediction: "áo khoác thể thao nữ màu đen"
- Mặc dù prediction ngắn hơn và thiếu "chất lượng cao giá rẻ", nhưng semantic similarity vẫn cao vì cả hai đều mô tả cùng một sản phẩm

**Ưu điểm**:
- Đánh giá semantic similarity, không chỉ từ overlap
- Phù hợp khi prediction và reference dùng từ khác nhau nhưng cùng nghĩa
- Phù hợp với mục tiêu thực tế: caption đúng nghĩa sản phẩm
- Không bị ảnh hưởng bởi độ dài hoặc format (SEO keywords)

**Hạn chế**:
- Phụ thuộc vào chất lượng của SBERT model
- Cần model được fine-tune cho ngôn ngữ cụ thể (tiếng Việt)
- Tính toán chậm hơn BLEU/ROUGE (cần encode embeddings)

**Trong dự án này**:
- SBERT = 0.6330 (khá tốt)
- Đây là metric quan trọng nhất vì:
  - Mục tiêu là tạo caption đúng nghĩa sản phẩm
  - Caption gốc có nhiều SEO keywords không cần thiết
  - SBERT đánh giá được semantic similarity tốt hơn

#### 2.4.4. So sánh các Metrics

| Metric | Phạm vi | Ưu điểm | Nhược điểm | Phù hợp khi |
|--------|---------|---------|------------|-------------|
| **BLEU** | [0, 1] | Đơn giản, phổ biến | Chỉ n-gram overlap | Prediction và reference có nhiều từ chung |
| **ROUGE-L** | [0, 1] | Đánh giá cấu trúc | Vẫn dựa trên từ overlap | Cấu trúc tương tự |
| **SBERT** | [0, 1] | Đánh giá semantic | Phụ thuộc model | Cùng nghĩa nhưng khác từ |

**Kết luận**:
- Trong dự án này, **SBERT là metric quan trọng nhất** vì nó đánh giá được semantic similarity
- BLEU và ROUGE-L vẫn hữu ích để đánh giá một phần, nhưng không phản ánh đầy đủ chất lượng caption
- Kết hợp cả 3 metrics cho đánh giá toàn diện

---

## 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Kiến trúc tổng thể

#### 3.1.1. Tổng quan kiến trúc

Hệ thống được thiết kế theo kiến trúc client-server với REST API, sử dụng mô hình microservices để tách biệt các thành phần chức năng. Kiến trúc này đảm bảo tính mở rộng, dễ bảo trì và hiệu suất cao.

**Nguyên tắc thiết kế**:
- **Separation of Concerns**: Tách biệt rõ ràng giữa API layer, business logic, và model layer
- **Modularity**: Mỗi module có trách nhiệm riêng, dễ test và maintain
- **Scalability**: Có thể mở rộng theo chiều ngang (horizontal scaling)
- **Performance**: Tối ưu với caching, batch processing, và memory management

#### 3.1.2. Sơ đồ kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Web App     │  │  Mobile App  │  │  API Client  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │ HTTP/REST API
                             │ (JSON, Multipart)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           FastAPI Application Server                 │   │
│  │  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │   Routes     │  │  Middleware  │                │   │
│  │  │  - Caption   │  │  - CORS       │                │   │
│  │  │  - Batch     │  │  - Auth       │                │   │
│  │  │  - Health    │  │  - Rate Limit │                │   │
│  │  └──────────────┘  └──────────────┘                │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Caption Service                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │ Image        │  │ Cache        │                │   │
│  │  │ Preprocess   │  │ Manager      │                │   │
│  │  │ - Resize     │  │ - Check      │                │   │
│  │  │ - Convert    │  │ - Store      │                │   │
│  │  │ - Validate   │  │ - Invalidate │                │   │
│  │  └──────────────┘  └──────────────┘                │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌───────────────────────────┐  ┌───────────────────────────┐
│      MODEL LAYER           │  │   ACCENT RESTORATION       │
│  ┌─────────────────────┐   │  │   ┌─────────────────────┐ │
│  │  BLIP Model         │   │  │   │ XLM-RoBERTa         │ │
│  │  - Vision Encoder   │───┼──┼──▶│ Token Classifier    │ │
│  │  - Text Decoder     │   │  │   │ Accent Predictor    │ │
│  │  - Processor        │   │  │   └─────────────────────┘ │
│  └─────────────────────┘   │  │                           │
│                            │  │   ┌─────────────────────┐ │
│  ┌─────────────────────┐   │  │   │ Token Merger        │ │
│  │  Model Loader       │   │  │   │ Label Mapper        │ │
│  │  - Load weights     │   │  │   └─────────────────────┘ │
│  │  - Device mgmt      │   │  │                           │
│  └─────────────────────┘   │  └───────────────────────────┘
└─────────────────────────────┘
```

#### 3.1.3. Mô tả các thành phần

**1. Client Layer (Lớp khách hàng)**:
- **Web App**: Ứng dụng web có thể tích hợp API
- **Mobile App**: Ứng dụng di động (iOS/Android)
- **API Client**: Các service khác gọi API
- Giao tiếp với server qua HTTP/REST API

**2. API Gateway Layer (Lớp cổng API)**:
- **FastAPI Server**: Framework web hiện đại, tự động generate documentation
- **Routes**: Định tuyến các endpoint
  - `/api/caption`: Caption không dấu (single)
  - `/api/caption/batch`: Caption không dấu (batch)
  - `/api/caption_full`: Caption có dấu (single)
  - `/api/caption_full/batch`: Caption có dấu (batch)
  - `/api/accent/restore`: Restore accent cho text
  - `/api/health`: Health check
  - `/api/cache/clear`: Clear cache
- **Middleware**: Xử lý cross-cutting concerns
  - CORS: Cho phép cross-origin requests
  - Authentication: Xác thực API key (optional)
  - Rate Limiting: Giới hạn số request

**3. Business Logic Layer (Lớp logic nghiệp vụ)**:
- **Caption Service**: Xử lý logic chính
  - Image Preprocessing: Resize, convert format, validate
  - Cache Management: Check cache, store results
  - Error Handling: Xử lý lỗi và trả về response phù hợp
  - Orchestration: Điều phối giữa BLIP và Accent Restoration

**4. Model Layer (Lớp mô hình)**:
- **BLIP Model**: 
  - Vision Encoder: Trích xuất features từ ảnh
  - Text Decoder: Sinh caption từ features
  - Processor: Xử lý ảnh và text
- **Model Loader**: 
  - Load model weights từ disk hoặc HuggingFace
  - Quản lý device (MPS/CUDA/CPU)
  - Model initialization và optimization

**5. Accent Restoration Layer (Lớp phục hồi dấu)**:
- **XLM-RoBERTa Token Classifier**: Predict accent labels
- **Token Merger**: Merge subword tokens thành words
- **Label Mapper**: Map labels sang accent patterns

#### 3.1.4. Luồng xử lý request

**Luồng xử lý một request đơn giản**:

1. **Client gửi request**:
   - POST `/api/caption_full`
   - Content-Type: `multipart/form-data`
   - Body: Image file

2. **API Gateway nhận request**:
   - FastAPI parse request
   - Middleware xử lý (CORS, Auth, Rate Limit)
   - Route handler nhận request

3. **Business Logic xử lý**:
   - Caption Service nhận image
   - Check cache (hash image)
   - Nếu có cache → Return ngay
   - Nếu không → Tiếp tục

4. **Image Preprocessing**:
   - Validate image format
   - Resize nếu > 512px
   - Convert RGB
   - Normalize

5. **BLIP Generation**:
   - Load image vào processor
   - Vision Encoder trích xuất features
   - Text Decoder sinh caption (không dấu)
   - Decode tokens thành text

6. **Accent Restoration**:
   - Tokenize caption không dấu
   - XLM-RoBERTa predict accent labels
   - Merge tokens và apply accents
   - Join thành caption có dấu

7. **Cache và Response**:
   - Cache kết quả (hash image → caption)
   - Format response JSON
   - Return cho client

### 3.2. Pipeline xử lý

#### 3.2.1. Training Pipeline

Training pipeline mô tả quá trình từ dữ liệu thô đến model đã được fine-tune:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: DATA COLLECTION                                │
│  ┌───────────────────────────────────────────────────┐ │
│  │ train_bilingual_clean_v2.csv                      │ │
│  │ - 7,638 samples                                    │ │
│  │ - Format: image, caption_vi                       │ │
│  │ - Images: data/images/ (7,443 files)             │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: DATA SPLITTING                                │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Shuffle với random_state=42                      │ │
│  │ Split 80/20                                       │ │
│  │                                                   │ │
│  │ Train: 6,110 samples (80%)                        │ │
│  │ Test: 1,528 samples (20%)                        │ │
│  │                                                   │ │
│  │ Output:                                          │ │
│  │ - train_80.csv                                   │ │
│  │ - test_20.csv                                    │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: DATA PREPROCESSING                            │
│  ┌───────────────────────────────────────────────────┐ │
│  │ For each sample:                                 │ │
│  │   1. Load image từ data/images/                  │ │
│  │   2. Convert to RGB                              │ │
│  │   3. Resize to 224×224 hoặc 384×384             │ │
│  │   4. Normalize pixel values                      │ │
│  │   5. Tokenize caption với BLIP processor        │ │
│  │   6. Create attention mask                       │ │
│  │   7. Create labels (ignore padding tokens)       │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: MODEL INITIALIZATION                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Load pretrained BLIP:                             │ │
│  │ - Salesforce/blip-image-captioning-base          │ │
│  │ - Vision Encoder: ViT-B/16                        │ │
│  │ - Text Decoder: BERT-based                        │ │
│  │ - Processor: BlipProcessor                        │ │
│  │                                                   │ │
│  │ Move to device:                                   │ │
│  │ - MPS (macOS) hoặc CUDA (Linux)                   │ │
│  │ - Set model.eval() → model.train()                │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: FINE-TUNING                                    │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Training Arguments:                              │ │
│  │ - Epochs: 5                                       │ │
│  │ - Batch size: 2                                  │ │
│  │ - Learning rate: 5e-5                            │ │
│  │ - Warmup steps: 500                              │ │
│  │ - Max length: 77                                 │ │
│  │                                                   │ │
│  │ Training Loop:                                   │ │
│  │ For each epoch:                                  │ │
│  │   For each batch:                                │ │
│  │     1. Forward pass                              │ │
│  │     2. Calculate loss                            │ │
│  │     3. Backward pass                             │ │
│  │     4. Update weights                            │ │
│  │   Evaluate on validation set                    │ │
│  │   Save best model                                │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6: MODEL SAVING                                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Save to: models/blip_vietnamese_80_20/           │ │
│  │ - config.json                                    │ │
│  │ - pytorch_model.bin                              │ │
│  │ - tokenizer_config.json                          │ │
│  │ - vocab.txt                                      │ │
│  │ - ...                                            │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Chi tiết các bước**:

**Bước 1: Data Collection**
- Dataset gốc: `train_bilingual_clean_v2.csv` với 7,638 samples
- Mỗi sample gồm: đường dẫn ảnh và caption tiếng Việt không dấu
- Ảnh được lưu trong thư mục `data/images/`

**Bước 2: Data Splitting**
- Shuffle dataset với `random_state=42` để đảm bảo reproducibility
- Chia 80/20: 6,110 samples cho training, 1,528 samples cho testing
- Lưu thành 2 file CSV riêng biệt

**Bước 3: Data Preprocessing**
- Load ảnh từ disk và convert sang RGB
- Resize về kích thước chuẩn (224×224 hoặc 384×384)
- Normalize pixel values về [0, 1] hoặc standardize
- Tokenize caption với BLIP processor
- Tạo attention mask và labels (ignore padding tokens với -100)

**Bước 4: Model Initialization**
- Load pretrained BLIP từ HuggingFace
- Chuyển model sang device (MPS cho macOS, CUDA cho Linux)
- Chuyển từ eval mode sang train mode

**Bước 5: Fine-tuning**
- Sử dụng HuggingFace Trainer với các tham số đã định
- Training loop: forward → loss → backward → update
- Evaluate mỗi epoch và lưu best model

**Bước 6: Model Saving**
- Lưu model weights, config, và tokenizer
- Model sẵn sàng cho inference

#### 3.2.2. Inference Pipeline

Inference pipeline mô tả quá trình xử lý một ảnh đầu vào để sinh caption:

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Image File                                      │
│  ┌───────────────────────────────────────────────────┐ │
│  │ - Format: JPEG, PNG, etc.                          │ │
│  │ - Size: Variable                                    │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: CACHE CHECK                                   │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. Load image vào memory                          │ │
│  │ 2. Convert to PNG format (standardize)           │ │
│  │ 3. Calculate MD5 hash                              │ │
│  │ 4. Check cache với hash key                        │ │
│  │                                                   │ │
│  │ IF cache hit AND not expired:                    │ │
│  │   → Return cached caption (skip to STEP 6)       │ │
│  │ ELSE:                                             │ │
│  │   → Continue to STEP 2                           │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: IMAGE PREPROCESSING                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. Validate image format                          │ │
│  │ 2. Convert to RGB (if needed)                     │ │
│  │ 3. Resize if max dimension > 512px               │ │
│  │    - Maintain aspect ratio                        │ │
│  │    - Use LANCZOS resampling                       │ │
│  │ 4. Normalize pixel values                          │ │
│  │                                                   │ │
│  │ Output: PIL Image object (RGB, normalized)      │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: BLIP GENERATION                                │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 3.1. Process image với BLIP processor            │ │
│  │      - Convert to tensor                          │ │
│  │      - Add batch dimension                        │ │
│  │      - Move to device (MPS/CUDA/CPU)            │ │
│  │                                                   │ │
│  │ 3.2. Vision Encoder                               │ │
│  │      - ViT processes image patches                │ │
│  │      - Output: Image features F_img               │ │
│  │                                                   │ │
│  │ 3.3. Text Decoder (Autoregressive)                │ │
│  │      For each token:                             │ │
│  │        - Cross-attention với image features      │ │
│  │        - Self-attention với previous tokens      │ │
│  │        - Predict next token                       │ │
│  │      - Use beam search (num_beams=3)             │ │
│  │      - Apply repetition penalty (1.2)            │ │
│  │                                                   │ │
│  │ 3.4. Decode tokens                                │ │
│  │      - Convert token IDs to text                 │ │
│  │      - Remove special tokens                     │ │
│  │                                                   │ │
│  │ Output: Caption không dấu (string)               │ │
│  │ Example: "ao khoac the thao nu mau den"          │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: ACCENT RESTORATION                             │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 4.1. Tokenize với XLM-RoBERTa tokenizer          │ │
│  │      - Split text thành words                    │ │
│  │      - Tokenize với is_split_into_words=True     │ │
│  │      - Add special tokens ([CLS], [SEP])        │ │
│  │                                                   │ │
│  │ 4.2. Predict accent labels                       │ │
│  │      - Forward pass qua XLM-RoBERTa              │ │
│  │      - Get logits for each token                 │ │
│  │      - Argmax to get predicted label             │ │
│  │                                                   │ │
│  │ 4.3. Merge subword tokens                         │ │
│  │      - Identify word boundaries (prefix "▁")     │ │
│  │      - Merge tokens belonging to same word      │ │
│  │                                                   │ │
│  │ 4.4. Apply accent labels                          │ │
│  │      - Map labels to accent patterns             │ │
│  │      - Format: "raw-vowel" (e.g., "ao-áo")     │ │
│  │      - Replace raw text with accented text      │ │
│  │                                                   │ │
│  │ 4.5. Join words                                   │ │
│  │      - Join accented words into sentence         │ │
│  │                                                   │ │
│  │ Output: Caption có dấu (string)                  │ │
│  │ Example: "áo khoác thể thao nữ màu đen"          │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: CACHE STORAGE                                 │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. Create cache entry:                            │ │
│  │    {                                                │ │
│  │      "caption": "áo khoác thể thao nữ màu đen",  │ │
│  │      "expires_at": current_time + TTL,            │ │
│  │      "created_at": current_time                   │ │
│  │    }                                               │ │
│  │ 2. Store với hash key                              │ │
│  │ 3. TTL: 24 hours (86400 seconds)                  │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6: RESPONSE FORMATTING                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Format JSON response:                             │ │
│  │ {                                                  │ │
│  │   "success": true,                                │ │
│  │   "caption_vi": "áo khoác thể thao nữ màu đen",  │ │
│  │   "caption_vi_no_accent": "ao khoac...",         │ │
│  │   "accent_restored": true,                        │ │
│  │   "device": "mps",                                │ │
│  │   "cached": false,                                │ │
│  │   "processing_time": 0.78                          │ │
│  │ }                                                 │ │
│  │                                                   │ │
│  │ Return to client                                  │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Chi tiết các bước**:

**Bước 1: Cache Check**
- Hash ảnh bằng MD5 để tạo cache key
- Kiểm tra cache, nếu có và chưa hết hạn → return ngay
- Giảm tải cho model và tăng tốc độ response

**Bước 2: Image Preprocessing**
- Validate format và kích thước
- Resize nếu quá lớn để giảm memory usage
- Chuẩn hóa format (RGB, normalized)

**Bước 3: BLIP Generation**
- Process ảnh với BLIP processor
- Vision Encoder trích xuất features
- Text Decoder sinh caption với beam search
- Decode tokens thành text

**Bước 4: Accent Restoration**
- Tokenize và predict accent labels
- Merge tokens và apply accents
- Join thành caption có dấu

**Bước 5: Cache Storage**
- Lưu kết quả vào cache với TTL 24h
- Giúp các request sau nhanh hơn

**Bước 6: Response Formatting**
- Format JSON response với đầy đủ thông tin
- Return cho client

#### 3.2.3. Batch Processing Pipeline

Khi xử lý nhiều ảnh cùng lúc (batch):

```
Input: [Image1, Image2, ..., ImageN] (N ≤ 10)
  ↓
For each image:
  ├─ Check cache
  ├─ If cached → Use cached result
  └─ If not cached:
      ├─ Preprocess
      ├─ BLIP generation
      ├─ Accent restoration
      └─ Cache result
  ↓
  Cleanup memory every 5 images
  ↓
Aggregate results
  ↓
Return: {
  "success": true,
  "total": N,
  "results": [
    {image1_result},
    {image2_result},
    ...
  ],
  "processing_time": total_time
}
```

**Tối ưu hóa batch processing**:
- Xử lý tuần tự từng ảnh (không parallel) để tránh memory overflow
- Cleanup memory mỗi 5 ảnh
- Cache từng ảnh riêng biệt
- Trả về kết quả cho tất cả ảnh, kể cả ảnh lỗi

### 3.3. API Design

#### 3.3.1. RESTful API Principles

API được thiết kế theo nguyên tắc RESTful:
- **Stateless**: Mỗi request độc lập, không lưu state
- **Resource-based**: URLs đại diện cho resources
- **HTTP Methods**: Sử dụng đúng HTTP methods (GET, POST)
- **JSON Format**: Request/Response dùng JSON
- **Error Handling**: Trả về HTTP status codes phù hợp

#### 3.3.2. Endpoints Chi tiết

**1. POST `/api/caption` - Single Caption (Không dấu)**

**Mục đích**: Sinh caption tiếng Việt không dấu cho một ảnh

**Request**:
```http
POST /api/caption
Content-Type: multipart/form-data

file: [image file]
```

**Response** (Success - 200):
```json
{
  "success": true,
  "caption_vi": "ao khoac the thao nu mau den",
  "device": "mps",
  "cached": false,
  "processing_time": 0.45
}
```

**Response** (Error - 400):
```json
{
  "detail": "Invalid image format. Supported formats: JPEG, PNG, etc."
}
```

**2. POST `/api/caption/batch` - Batch Caption (Không dấu)**

**Mục đích**: Sinh caption không dấu cho nhiều ảnh (tối đa 10)

**Request**:
```http
POST /api/caption/batch
Content-Type: multipart/form-data

files: [image1, image2, ..., imageN]
```

**Response** (Success - 200):
```json
{
  "success": true,
  "total": 3,
  "results": [
    {
      "index": 0,
      "filename": "image1.jpg",
      "caption_vi": "ao khoac the thao nu",
      "success": true,
      "cached": false
    },
    {
      "index": 1,
      "filename": "image2.jpg",
      "caption_vi": "giay the thao mau trang",
      "success": true,
      "cached": true
    },
    {
      "index": 2,
      "filename": "image3.jpg",
      "success": false,
      "error": "Invalid image format"
    }
  ],
  "device": "mps",
  "processing_time": 2.34
}
```

**3. POST `/api/caption_full` - Single Caption (Có dấu)**

**Mục đích**: Sinh caption tiếng Việt có dấu cho một ảnh (pipeline đầy đủ)

**Request**:
```http
POST /api/caption_full
Content-Type: multipart/form-data

file: [image file]
```

**Response** (Success - 200):
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

**4. POST `/api/caption_full/batch` - Batch Caption (Có dấu)**

**Mục đích**: Sinh caption có dấu cho nhiều ảnh (tối đa 10)

**Request/Response**: Tương tự `/api/caption/batch` nhưng có thêm `caption_vi_no_accent` và `accent_restored`

**5. POST `/api/accent/restore` - Restore Accent cho Text**

**Mục đích**: Test accent restoration với text không cần ảnh

**Request**:
```http
POST /api/accent/restore
Content-Type: application/json

{
  "text": "ao khoac the thao mau den"
}
```

**Response** (Success - 200):
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

**6. GET `/api/health` - Health Check**

**Mục đích**: Kiểm tra trạng thái API và models

**Request**:
```http
GET /api/health
```

**Response** (Success - 200):
```json
{
  "status": "healthy",
  "device": "mps",
  "blip_model_loaded": true,
  "accent_model_loaded": true,
  "cache_enabled": true,
  "cache_stats": {
    "total_entries": 150,
    "valid_entries": 148,
    "expired_entries": 2
  }
}
```

**7. POST `/api/cache/clear` - Clear Cache**

**Mục đích**: Xóa toàn bộ cache

**Request**:
```http
POST /api/cache/clear
```

**Response** (Success - 200):
```json
{
  "success": true,
  "message": "Cache đã được xóa"
}
```

#### 3.3.3. Request/Response Format

**Request Headers**:
- `Content-Type`: `multipart/form-data` (cho image upload) hoặc `application/json` (cho text)
- `Authorization`: `Bearer <api_key>` (nếu enable auth)

**Response Headers**:
- `Content-Type`: `application/json`
- `Access-Control-Allow-Origin`: `*` (CORS)

**Response Structure**:
- **Success**: `{"success": true, ...data...}`
- **Error**: `{"detail": "error message"}`

**HTTP Status Codes**:
- `200 OK`: Request thành công
- `400 Bad Request`: Request không hợp lệ (invalid format, missing file, etc.)
- `429 Too Many Requests`: Vượt quá rate limit
- `500 Internal Server Error`: Lỗi server (model not loaded, etc.)

### 3.4. Database/Cache Design

#### 3.4.1. Cache Strategy

Hệ thống sử dụng in-memory caching để tăng tốc độ xử lý và giảm tải cho model.

**Lý do sử dụng cache**:
- **Performance**: Tránh regenerate caption cho cùng một ảnh
- **Cost**: Giảm computation cost (không cần chạy model lại)
- **User Experience**: Response time nhanh hơn (0.01s vs 0.8s)

**Cache Architecture**:
```
┌─────────────────────────────────────────┐
│         Cache Storage                    │
│  ┌───────────────────────────────────┐ │
│  │  In-Memory Dictionary              │ │
│  │  {                                  │ │
│  │    "hash1": {                      │ │
│  │      "caption": "...",             │ │
│  │      "expires_at": timestamp,      │ │
│  │      "created_at": timestamp       │ │
│  │    },                               │ │
│  │    "hash2": {...},                 │ │
│  │    ...                              │ │
│  │  }                                  │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Cache Key Generation**:
- Input: Image file
- Process:
  1. Load image vào memory
  2. Convert to PNG format (standardize)
  3. Calculate MD5 hash
  4. Use hash as cache key
- Example: `"a1b2c3d4e5f6..."`

**Cache Value Structure**:
```python
{
  "caption": "áo khoác thể thao nữ màu đen",
  "expires_at": 1234567890.0,  # Unix timestamp
  "created_at": 1234560000.0   # Unix timestamp
}
```

**TTL (Time To Live)**:
- Default: 24 hours (86400 seconds)
- Có thể config qua environment variable `CACHE_TTL`
- Sau khi hết hạn, entry tự động bị xóa khi check

#### 3.4.2. Cache Operations

**1. Cache Lookup (Check)**:
```python
def get_cached_caption(image_hash: str) -> Optional[str]:
    if image_hash not in cache:
        return None
    
    entry = cache[image_hash]
    if time.time() > entry['expires_at']:
        # Expired, delete and return None
        del cache[image_hash]
        return None
    
    return entry['caption']
```

**2. Cache Store (Save)**:
```python
def set_cached_caption(image_hash: str, caption: str, ttl: int = 86400):
    cache[image_hash] = {
        'caption': caption,
        'expires_at': time.time() + ttl,
        'created_at': time.time()
    }
```

**3. Cache Invalidation (Clear)**:
```python
def clear_cache():
    cache.clear()
```

**4. Cache Statistics**:
```python
def get_cache_stats():
    now = time.time()
    valid = sum(1 for e in cache.values() if e['expires_at'] > now)
    expired = len(cache) - valid
    return {
        'total_entries': len(cache),
        'valid_entries': valid,
        'expired_entries': expired
    }
```

#### 3.4.3. Cache Flow Diagram

```
Request Image
    ↓
Load Image → Convert to PNG → Calculate MD5 Hash
    ↓
Check Cache with Hash Key
    ↓
    ├─ Cache Hit?
    │   ├─ Yes → Check Expiry
    │   │   ├─ Valid → Return Cached Caption (0.01s)
    │   │   └─ Expired → Delete Entry → Continue
    │   └─ No → Continue
    ↓
Generate Caption (BLIP + Accent Restoration)
    ↓
Store in Cache (Hash → Caption + Metadata)
    ↓
Return Caption
```

#### 3.4.4. Cache Limitations và Future Improvements

**Limitations hiện tại**:
- In-memory: Mất cache khi restart server
- Single server: Không share cache giữa multiple servers
- No persistence: Không lưu vào disk

**Future Improvements**:
- **Redis**: Sử dụng Redis để persistent cache và share giữa servers
- **Disk Cache**: Lưu cache vào disk để survive server restart
- **Distributed Cache**: Share cache giữa multiple API servers

### 3.5. Error Handling

#### 3.5.1. Error Categories

Hệ thống xử lý các loại lỗi sau:

**1. Client Errors (4xx)**:
- **400 Bad Request**: Request không hợp lệ
  - Invalid image format
  - Missing file
  - Image too large
  - Empty file
- **429 Too Many Requests**: Vượt quá rate limit

**2. Server Errors (5xx)**:
- **500 Internal Server Error**: Lỗi server
  - Model not loaded
  - Inference error
  - Memory error

**3. Business Logic Errors**:
- Image processing error
- Model generation error
- Accent restoration error
- Cache error (non-fatal, continue without cache)

#### 3.5.2. Error Handling Strategy

**1. Validation Errors (400)**:
```python
# Invalid image format
if not image.format in ['JPEG', 'PNG', 'WEBP']:
    raise HTTPException(
        status_code=400,
        detail="Invalid image format. Supported: JPEG, PNG, WEBP"
    )

# Image too large
if image.size[0] * image.size[1] > MAX_IMAGE_SIZE:
    raise HTTPException(
        status_code=400,
        detail=f"Image too large. Max size: {MAX_IMAGE_SIZE} pixels"
    )
```

**2. Rate Limiting Errors (429)**:
```python
# Rate limit exceeded
if rate_limit_exceeded(request):
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Please try again later."
    )
```

**3. Server Errors (500)**:
```python
# Model not loaded
if model is None:
    raise HTTPException(
        status_code=500,
        detail="Model not loaded. Please check server logs."
    )

# Inference error
try:
    caption = generate_caption(image)
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Error during caption generation: {str(e)}"
    )
```

**4. Non-Fatal Errors (Continue)**:
```python
# Cache error (non-fatal)
try:
    cached = get_cached_caption(hash)
except Exception:
    # Continue without cache
    cached = None

# Accent restoration error (fallback to no accent)
try:
    caption_with_accent = restore_accent(caption_no_accent)
except Exception:
    # Fallback to no accent
    caption_with_accent = caption_no_accent
    accent_restored = False
```

#### 3.5.3. Error Response Format

**Standard Error Response**:
```json
{
  "detail": "Error message here"
}
```

**Detailed Error Response** (for debugging):
```json
{
  "detail": "Error message",
  "error_type": "ValidationError",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "abc123"
}
```

#### 3.5.4. Error Logging

Tất cả errors được log để debugging:
- Log level: ERROR
- Include: Error message, stack trace, request details
- Log destination: File hoặc console (configurable)

**Example Log**:
```
[ERROR] 2024-01-01 12:00:00 - Caption generation failed
  Request: POST /api/caption_full
  Error: CUDA out of memory
  Stack trace: ...
```

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

