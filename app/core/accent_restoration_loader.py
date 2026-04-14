"""
Accent Restoration Model Loader - peterhung/vietnamese-accent-marker-xlm-roberta
Model này chuyển caption tiếng Việt KHÔNG DẤU → CÓ DẤU
Sử dụng Token Classification model - nhẹ, nhanh, chính xác
Tối ưu cho macOS M1 với MPS backend
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
    # Load từ HuggingFace (model này không cần fine-tune local)
    print(f"📥 Đang tải model từ HuggingFace: {ACCENT_MODEL_NAME}")
    accent_tokenizer = AutoTokenizer.from_pretrained(
        ACCENT_MODEL_NAME,
        add_prefix_space=True
    )
    accent_model = AutoModelForTokenClassification.from_pretrained(ACCENT_MODEL_NAME)
    
    # Load tags
    label_list = load_tags_from_huggingface(ACCENT_MODEL_NAME)
    
    if label_list is None:
        # Fallback: thử load từ local nếu có
        tags_local = ACCENT_MODEL_PATH / "selected_tags_names.txt"
        if tags_local.exists():
            with open(tags_local, 'r', encoding='utf-8') as f:
                label_list = [line.strip() for line in f if line.strip()]
            print(f"✅ Đã load {len(label_list)} tags từ local")
        else:
            raise Exception("Không thể load tags. Vui lòng tải file selected_tags_names.txt từ HuggingFace.")
    
    # Move model to device
    accent_model.to(device)  # type: ignore
    accent_model.eval()
    
    # Synchronize device sau khi load model (quan trọng cho MPS)
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
    Merge subword tokens thành words và collect predictions
    """
    merged_tokens_preds = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        label_indexes = {predictions[i]}
        
        # XLM-RoBERTa dùng prefix "▁" để đánh dấu đầu từ
        if tok.startswith("▁"):
            tok_no_prefix = tok[1:]
            cur_word_toks = [tok_no_prefix]
            j = i + 1
            
            # Collect tất cả subword tokens của từ này
            while j < len(tokens) and not tokens[j].startswith("▁"):
                cur_word_toks.append(tokens[j])
                label_indexes.add(predictions[j])
                j += 1
            
            cur_word = ''.join(cur_word_toks)
            merged_tokens_preds.append((cur_word, label_indexes))
            i = j
        else:
            merged_tokens_preds.append((tok, label_indexes))
            i += 1
    
    return merged_tokens_preds

def get_accented_words(merged_tokens_preds, label_list):
    """
    Áp dụng labels để chuyển từ không dấu → có dấu
    """
    accented_words = []
    
    for word_raw, label_indexes in merged_tokens_preds:
        word_accented = word_raw  # Default: giữ nguyên nếu không tìm thấy label
        
        # Tìm label phù hợp nhất
        for label_index in label_indexes:
            if label_index < len(label_list):
                tag_name = label_list[int(label_index)]
                
                # Format: "raw-vowel" (ví dụ: "ao-áo")
                if "-" in tag_name:
                    raw, vowel = tag_name.split("-", 1)
                    if raw and raw in word_raw:
                        word_accented = word_raw.replace(raw, vowel)
                        break
        
        accented_words.append(word_accented)
    
    return accented_words

def restore_accent(text_no_accent: str, max_length: int = 512) -> str:
    """
    Chuyển text tiếng Việt không dấu → có dấu
    
    Args:
        text_no_accent: Text tiếng Việt không dấu
        max_length: Độ dài tối đa (model giới hạn 512 tokens)
    
    Returns:
        Text tiếng Việt có dấu
    """
    if accent_model is None or accent_tokenizer is None or label_list is None:
        # Nếu model chưa load được, trả về text gốc
        return text_no_accent
    
    try:
        # Split text thành words (model expect word-level input)
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
            # Synchronize device để đảm bảo computation hoàn thành (quan trọng cho MPS)
            synchronize_device()
        
        # Get predicted labels - move to CPU trước khi convert numpy
        predictions = outputs["logits"].cpu().numpy()
        predictions = np.argmax(predictions, axis=2)[0]
        
        # Convert token ids to tokens (skip special tokens)
        input_ids = inputs['input_ids'][0]
        token_tokens = accent_tokenizer.convert_ids_to_tokens(input_ids)
        
        # Cleanup: Delete tensors để giải phóng memory
        del inputs, outputs
        
        # Skip [CLS] và [SEP] tokens
        token_tokens = token_tokens[1:-1] if len(token_tokens) > 2 else token_tokens[1:]
        predictions = predictions[1:-1] if len(predictions) > 2 else predictions[1:]
        
        # Merge tokens và apply accents
        merged_tokens_preds = merge_tokens_and_preds(token_tokens, predictions)
        accented_words = get_accented_words(merged_tokens_preds, label_list)
        
        # Join words thành text
        accented_text = ' '.join(accented_words)
        return accented_text.strip()
    
    except Exception as e:
        print(f"⚠️  Lỗi khi restore accent: {e}")
        # Trả về text gốc nếu có lỗi
        return text_no_accent
