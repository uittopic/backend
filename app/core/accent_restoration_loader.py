"""
Accent Restoration Model Loader - peterhung/vietnamese-accent-marker-xlm-roberta
Model chuyển caption tiếng Việt KHÔNG DẤU → CÓ DẤU
Sử dụng Token Classification model - nhẹ, nhanh, chính xác
Tối ưu cho macOS M1 với MPS backend

FIX v4: Dùng is_split_into_words=False để tokenizer tự tách word boundaries
bằng ▁ prefix. Group tokens theo ▁ prefix (không theo is_split_into_words).
Lấy prediction = mode của tất cả tokens trong group.
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

    label_list = load_tags_from_huggingface(ACCENT_MODEL_NAME)

    if label_list is None:
        tags_local = ACCENT_MODEL_PATH / "selected_tags_names.txt"
        if tags_local.exists():
            with open(tags_local, 'r', encoding='utf-8') as f:
                label_list = [line.strip() for line in f if line.strip()]
            print(f"✅ Đã load {len(label_list)} tags từ local")
        else:
            raise Exception("Không thể load tags.")

    accent_model.to(device)
    accent_model.eval()
    synchronize_device()

    print(f"✅ Accent Restoration Model đã được load và chuyển sang {device}")
    print("📝 Model chuyển caption KHÔNG DẤU → CÓ DẤU (Token Classification)")

except Exception as e:
    print(f"❌ Lỗi khi load Accent Restoration Model: {e}")
    print("⚠️  API /api/caption_full sẽ không hoạt động")
    accent_tokenizer = None
    accent_model = None
    label_list = None


def restore_accent(text_no_accent: str, max_length: int = 512) -> str:
    """
    Chuyển text tiếng Việt không dấu → có dấu.

    Strategy: Dùng is_split_into_words=False để tokenizer tự tách word
    boundaries bằng ▁ prefix. Group tokens theo ▁ prefix, dùng mode
    prediction của group để apply accent cho toàn bộ word.
    """
    if accent_model is None or accent_tokenizer is None or label_list is None:
        return text_no_accent

    try:
        text = text_no_accent.strip()
        if not text:
            return text_no_accent

        # Tokenize full text (is_split_into_words=False)
        # XLM-RoBERTa SentencePiece tự thêm ▁ cho word starts
        inputs = accent_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = accent_model(**inputs)
            synchronize_device()

        logits = outputs["logits"].cpu().numpy()
        predictions = np.argmax(logits, axis=2)[0]  # (seq_len,)

        token_tokens = accent_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        del inputs, outputs

        # Skip BOS/EOS tokens
        token_tokens = token_tokens[1:]
        predictions = predictions[1:]

        # Remove trailing padding/special tokens
        filtered_tokens = []
        filtered_preds = []
        for tok, pred in zip(token_tokens, predictions):
            if tok in ('<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]'):
                continue
            filtered_tokens.append(tok)
            filtered_preds.append(pred)

        # Group tokens by word: each group starts with ▁ prefix
        # All consecutive tokens until next ▁ belong to the same word
        word_groups = []  # list of (tokens, predictions)
        current_tokens = []
        current_preds = []

        for tok, pred in zip(filtered_tokens, filtered_preds):
            if tok.startswith("▁"):
                # Save previous group if any
                if current_tokens:
                    word_groups.append((current_tokens, current_preds))
                # Start new word
                current_tokens = [tok]
                current_preds = [pred]
            else:
                # Subword of current word
                current_tokens.append(tok)
                current_preds.append(pred)

        # Don't forget the last group
        if current_tokens:
            word_groups.append((current_tokens, current_preds))

        # Apply accent per word group
        result_words = []
        for toks, preds in word_groups:
            # Merge subword tokens (strip ▁ from first token)
            word_str = toks[0][1:]  # remove ▁
            for tp in toks[1:]:
                word_str += tp.lstrip("▁")

            # Prediction = mode of all tokens in group
            if preds:
                label = int(np.median(preds))
            else:
                label = 0

            # Apply accent using label
            word_accented = word_str
            if 0 <= label < len(label_list):
                tag_name = label_list[label]
                if "-" in tag_name:
                    raw_base, vowel = tag_name.split("-", 1)
                    if raw_base and raw_base in word_str:
                        word_accented = word_str.replace(raw_base, vowel, 1)

            result_words.append(word_accented)

        return " ".join(result_words).strip()

    except Exception as e:
        print(f"⚠️  Lỗi khi restore accent: {e}")
        return text_no_accent
