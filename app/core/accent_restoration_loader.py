"""
Accent Restoration Model Loader - peterhung/vietnamese-accent-marker-xlm-roberta
Model này chuyển caption tiếng Việt KHÔNG DẤU → CÓ DẤU
Sử dụng Token Classification model - nhẹ, nhanh, chính xác
Tối ưu cho macOS M1 với MPS backend

CẢI TIẾN v3:
  - merge_tokens_and_preds: xử lý edge cases (empty prefix, special tokens)
  - get_accented_words: ưu tiên match dài nhất trong subword
  - restore_accent: cải thiện logic chuyển đổi
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from typing import Optional
from app.core.config import get_device, ACCENT_MODEL_NAME, ACCENT_MODEL_PATH, synchronize_device

print(f"🔄 Đang load Accent Restoration Model ({ACCENT_MODEL_NAME})...")
print(f"📱 Device: {get_device()}")

# Global variables
accent_tokenizer = None
accent_model = None
label_list = None
device = get_device()

def load_tags_from_huggingface(model_name: str) -> Optional[list]:
    """
    Tải file selected_tags_names.txt từ HuggingFace
    """
    try:
        tags_file = hf_hub_download(
            repo_id=model_name,
            filename="selected_tags_names.txt",
            cache_dir=None
        )
        with open(tags_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"✅ Đã tải {len(labels)} tags từ HuggingFace")
        return labels
    except Exception as e:
        print(f"⚠️  Không thể tải tags từ HuggingFace: {e}")
        print("⚠️  Sẽ thử load từ local hoặc dùng fallback")
        return None

# Load tokenizer và model
try:
    print(f"📥 Đang tải model từ HuggingFace: {ACCENT_MODEL_NAME}")
    accent_tokenizer = AutoTokenizer.from_pretrained(
        ACCENT_MODEL_NAME,
        add_prefix_space=True
    )
    accent_model = AutoModelForTokenClassification.from_pretrained(ACCENT_MODEL_NAME)

    # Load tags
    label_list = load_tags_from_huggingface(ACCENT_MODEL_NAME)

    if label_list is None:
        tags_local = ACCENT_MODEL_PATH / "selected_tags_names.txt"
        if tags_local.exists():
            with open(tags_local, 'r', encoding='utf-8') as f:
                label_list = [line.strip() for line in f if line.strip()]
            print(f"✅ Đã load {len(label_list)} tags từ local")
        else:
            raise Exception("Không thể load tags. Vui lòng tải file selected_tags_names.txt từ HuggingFace.")

    # Move model to device
    accent_model.to(device)
    accent_model.eval()
    synchronize_device()

    print(f"✅ Accent Restoration Model đã được load và chuyển sang {device}")
    print("📝 Model này chuyển caption KHÔNG DẤU → CÓ DẤU (Token Classification)")

except Exception as e:
    print(f"❌ Lỗi khi load Accent Restoration Model: {e}")
    print("⚠️  API /api/caption_full sẽ không hoạt động")
    accent_tokenizer = None
    accent_model = None
    label_list = None


def merge_tokens_and_preds(tokens, predictions):
    """
    Merge subword tokens thành words và collect predictions.
    CẢI TIẾN: xử lý edge cases tốt hơn.
    """
    merged_tokens_preds = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        label_indexes = {int(predictions[i])}

        # XLM-RoBERTa dùng prefix "▁" để đánh dấu đầu từ
        if tok.startswith("▁"):
            tok_no_prefix = tok[1:]
            cur_word_toks = [tok_no_prefix] if tok_no_prefix else []
            j = i + 1

            # Collect tất cả subword tokens của từ này
            while j < len(tokens) and not tokens[j].startswith("▁"):
                cur_word_toks.append(tokens[j])
                label_indexes.add(int(predictions[j]))
                j += 1

            if tok_no_prefix:
                cur_word = tok_no_prefix
            elif cur_word_toks:
                cur_word = ''.join(cur_word_toks)
            else:
                cur_word = tok
            merged_tokens_preds.append((cur_word, label_indexes))
            i = j
        else:
            # Bỏ qua special tokens
            if tok not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                merged_tokens_preds.append((tok, label_indexes))
            i += 1

    return merged_tokens_preds


def get_accented_words(merged_tokens_preds, label_list):
    """
    Áp dụng labels để chuyển từ không dấu → có dấu.
    CẢI TIẾN: ưu tiên match dài nhất.
    """
    accented_words = []

    for word_raw, label_indexes in merged_tokens_preds:
        if not word_raw:
            continue

        word_accented = word_raw
        best_match = None
        best_raw_len = 0

        for label_index in label_indexes:
            if label_index < len(label_list):
                tag_name = label_list[int(label_index)]

                if "-" in tag_name:
                    raw, vowel = tag_name.split("-", 1)
                    # Ưu tiên match có độ dài lớn hơn
                    if raw and len(raw) >= best_raw_len and raw in word_raw:
                        best_match = vowel
                        best_raw_len = len(raw)

        if best_match:
            word_accented = word_raw.replace(word_raw[:best_raw_len], best_match, 1)

        accented_words.append(word_accented)

    return accented_words


def restore_accent(text_no_accent: str, max_length: int = 512) -> str:
    """
    Chuyển text tiếng Việt không dấu → có dấu.

    CẢI TIẾN:
      - Thêm fallback khi model predict sai tag
      - Cải thiện handling empty/whitespace input
      - Tối ưu memory bằng cách xóa tensor sau khi dùng
    """
    if accent_model is None or accent_tokenizer is None or label_list is None:
        return text_no_accent

    try:
        tokens = text_no_accent.strip().split()

        if not tokens:
            return text_no_accent

        # Tokenize với is_split_into_words=True
        inputs = accent_tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = accent_model(**inputs)
            synchronize_device()

        predictions = outputs["logits"].cpu().numpy()
        predictions = np.argmax(predictions, axis=2)[0]

        input_ids = inputs['input_ids'][0]
        token_tokens = accent_tokenizer.convert_ids_to_tokens(input_ids)

        # Cleanup memory
        del inputs, outputs

        # Skip special tokens
        if len(token_tokens) > 2:
            token_tokens = token_tokens[1:-1]
            predictions = predictions[1:-1]

        # Merge tokens và apply accents
        merged_tokens_preds = merge_tokens_and_preds(token_tokens, predictions)
        accented_words = get_accented_words(merged_tokens_preds, label_list)

        accented_text = ' '.join(accented_words)
        return accented_text.strip()

    except Exception as e:
        print(f"⚠️  Lỗi khi restore accent: {e}")
        return text_no_accent
