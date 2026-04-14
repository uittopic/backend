# Template Phân Tích Lỗi Caption

Mục tiêu: dùng template này cho phần "chuẩn bị kỹ hơn phần thực nghiệm" trong chuyên đề/khóa luận.

## 1) Cấu hình thực nghiệm

- EXP ID:
- Model path:
- Dataset test:
- Decode params: `num_beams=`, `no_repeat_ngram_size=`, `repetition_penalty=`
- Có accent restoration: `true/false`
- Metrics tổng: BLEU, ROUGE-L, SBERT

## 2) Phân loại lỗi

| Nhóm lỗi | Mô tả | Ví dụ điển hình | Tần suất (%) | Mức ảnh hưởng |
|---|---|---|---:|---|
| Sai loại sản phẩm | Caption nhận sai category chính | "áo" thành "túi" |  | Cao |
| Thiếu thuộc tính | Bỏ sót màu/chất liệu/đối tượng | thiếu "nữ", thiếu "màu đen" |  | Trung bình |
| Lặp từ | Lặp cụm nhiều lần | "dynamax dynamax ..." |  | Trung bình |
| Trộn ngôn ngữ | EN/ID lẫn tiếng Việt | "men's watches", "wanita" |  | Trung bình |
| Lỗi dấu/typo | Có lỗi chính tả hoặc dấu | "đong" thay vì "đồng" |  | Thấp |

## 3) Bảng mẫu lỗi chi tiết (ít nhất 20 mẫu)

| STT | image | Ground truth | Prediction | Loại lỗi chính | Nhận xét nguyên nhân | Hướng xử lý |
|---:|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |

## 4) Root cause theo hệ thống

- Dữ liệu:
- Mô hình:
- Decode:
- Accent restoration:
- API pipeline:

## 5) Kế hoạch cải tiến theo ưu tiên

| Ưu tiên | Việc cần làm | Kỳ vọng tác động | Cách đo |
|---:|---|---|---|
| P1 | Chạy ablation decode (beam/no-repeat/repetition) | tăng ROUGE-L, giảm lặp từ | compare bảng metrics |
| P1 | Làm sạch dữ liệu train (lọc caption lỗi/ngôn ngữ) | tăng BLEU + SBERT | rerun cùng split |
| P2 | Tăng robustness API (fallback caption rỗng/rác) | giảm lỗi runtime/demo | tỉ lệ lỗi endpoint |
| P2 | Phân tích lỗi top 20 mẫu mỗi EXP | rõ nguyên nhân để chọn hướng | error log + template |

## 6) Kết luận mỗi vòng thực nghiệm

- EXP tốt nhất vòng này:
- Vì sao chọn:
- Điểm còn yếu:
- Quyết định vòng kế tiếp:
