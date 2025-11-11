"""
Fine-tune BLIP model cho tiếng Việt
Tối ưu cho macOS M1 Pro Max với MPS backend
"""
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from PIL import Image
import pandas as pd
import torch
import os
from pathlib import Path

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Tạo thư mục nếu chưa có
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

def get_device():
    """Xác định device tối ưu cho macOS M1"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

print("=" * 60)
print("🚀 Bắt đầu Fine-tune BLIP cho tiếng Việt")
print("=" * 60)

# === Load dataset ===
print("\n📂 Đang load dataset...")
csv_path = DATA_DIR / "train_bilingual_clean_v2.csv"
df = pd.read_csv(csv_path)

print(f"✅ Đã load {len(df)} samples")
print(f"📊 Các cột: {df.columns.tolist()}")

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split train/val (90/10)
split = int(0.9 * len(df))
train_df = df[:split].copy()
val_df = df[split:].copy()

print(f"📈 Train: {len(train_df)} samples")
print(f"📈 Val: {len(val_df)} samples")

# === Load BLIP model ===
print("\n🤖 Đang load BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = get_device()
print(f"📱 Device: {device}")

# Chuyển model sang device
model.to(device)
print("✅ Model đã được load")

# === Preprocess function ===
def preprocess(batch):
    """Xử lý batch ảnh và caption"""
    images, texts = [], []
    
    for img_name, caption in zip(batch["image"], batch["caption_vi"]):
        img_path = DATA_DIR / "images" / img_name
        
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            texts.append(caption)
        except Exception as e:
            print(f"⚠️  Lỗi khi load ảnh {img_name}: {e}")
            continue
    
    if len(images) == 0:
        return {}
    
    # Process với processor
    inputs = processor(
        images=images, 
        text=texts, 
        padding="max_length", 
        truncation=True, 
        max_length=77,
        return_tensors="pt"
    )
    
    # Labels cho training
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

# === Tạo datasets ===
print("\n🔄 Đang tạo datasets...")
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print("🔄 Đang preprocess train dataset...")
train_dataset = train_dataset.map(
    preprocess, 
    batched=True, 
    batch_size=10,
    remove_columns=train_dataset.column_names
)

print("🔄 Đang preprocess val dataset...")
val_dataset = val_dataset.map(
    preprocess, 
    batched=True, 
    batch_size=10,
    remove_columns=val_dataset.column_names
)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Val dataset: {len(val_dataset)} samples")

# === Training Arguments ===
# Tối ưu cho M1: batch size nhỏ hơn, không dùng fp16 (MPS chưa hỗ trợ tốt)
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "blip_vietnamese"),
    per_device_train_batch_size=2,  # Nhỏ hơn cho M1
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_dir=str(LOG_DIR),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,  # MPS chưa hỗ trợ fp16 tốt
    dataloader_num_workers=0,  # Tránh lỗi multiprocessing trên macOS
    report_to="none",  # Không gửi lên wandb/tensorboard
)

# === Trainer ===
print("\n🏋️  Đang khởi tạo Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === Train ===
print("\n🚀 Bắt đầu training...")
print("=" * 60)
trainer.train()

# === Save model ===
print("\n💾 Đang lưu model...")
final_model_path = MODEL_DIR / "blip_vietnamese"
model.save_pretrained(str(final_model_path))
processor.save_pretrained(str(final_model_path))

print("=" * 60)
print(f"✅ Fine-tune BLIP Vietnamese hoàn thành!")
print(f"📁 Model đã được lưu tại: {final_model_path}")
print("=" * 60)

