"""
Sanity Check - Train 1 epoch với cleaned data
Để xem loss và sample caption trước khi train full
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from PIL import Image

from app.core.config import get_device

# === CONFIG ===
DEVICE = get_device()
MODEL_PATH = Path("models/blip_vietnamese")  # Dùng model local đã train
IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("models/blip_vietnamese_cleaned_v1_sanity")

# Dataset cleaned (đã loại SEO words)
TRAIN_CSV = "data/train_80_cleaned.csv"
VAL_CSV = "data/test_20.csv"

# Training config
NUM_EPOCHS = 1  # Sanity check = 1 epoch
BATCH_SIZE = 2
MAX_LENGTH = 60

print("=" * 60)
print("🔍 SANITY CHECK - Train 1 epoch với Cleaned Data")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Train CSV: {TRAIN_CSV}")
print(f"Epochs: {NUM_EPOCHS}")
print()

# === LOAD DATA ===
print("📂 Load dataset...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Normalize
train_df["caption_vi_cleaned"] = train_df["caption_vi_cleaned"].fillna("")
train_df = train_df[train_df["caption_vi_cleaned"] != ""].copy()
val_df["caption_vi"] = val_df["caption_vi"].fillna("")
val_df = val_df[val_df["caption_vi"] != ""].copy()

# Giới hạn cho sanity check
MAX_TRAIN = 200
MAX_VAL = 50
train_df = train_df.head(MAX_TRAIN)
val_df = val_df.head(MAX_VAL)

print(f"  Train samples: {len(train_df)}")
print(f"  Val samples: {len(val_df)}")

# === LOAD MODEL (từ local) ===
print()
print("📦 Load pretrained model từ local...")
if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
    processor = BlipProcessor.from_pretrained(str(MODEL_PATH))
    model = BlipForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    print(f"  ✅ Load từ {MODEL_PATH}")
else:
    print(f"  ⚠️  Model local không tồn tại, dùng pretrained...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

model.to(DEVICE)
model.train()
print(f"  ✅ Model loaded on {DEVICE}")

# === CREATE DATASET ===
def create_dataset(df: pd.DataFrame, caption_col: str) -> Dataset:
    """Tạo HuggingFace Dataset từ DataFrame"""
    images = []
    captions = []
    
    for _, row in df.iterrows():
        img_path = IMAGE_DIR / row["image"]
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                captions.append(str(row[caption_col]))
            except:
                pass
    
    return Dataset.from_dict({"image": images, "caption": captions})

def preprocess(example):
    """Preprocess cho training"""
    inputs = processor(
        images=example["image"],
        text=example["caption"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["pixel_values"] = inputs["pixel_values"].squeeze(0) if len(inputs["pixel_values"].shape) == 4 else inputs["pixel_values"]
    
    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs["pixel_values"],
        "labels": inputs["labels"].squeeze(0)
    }

print()
print("🔧 Tạo datasets...")
train_dataset = create_dataset(train_df, "caption_vi_cleaned")
val_dataset = create_dataset(val_df, "caption_vi")

# Map preprocessing
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=["image", "caption"],
    batched=False,
    desc="Preprocessing train"
)
val_dataset = val_dataset.map(
    preprocess,
    remove_columns=["image", "caption"],
    batched=False,
    desc="Preprocessing val"
)

print(f"  Train dataset: {len(train_dataset)}")
print(f"  Val dataset: {len(val_dataset)}")

# === TRAINING ARGUMENTS ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=10,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    report_to="none",
    fp16=False if DEVICE == "mps" else True,
)

# === TRAINER ===
print()
print("🏋️ Bắt đầu training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === TRAIN ===
trainer.train()

# === SAVE MODEL ===
print()
print(f"💾 Lưu model: {OUTPUT_DIR}")
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))

# === SAMPLE OUTPUT ===
print()
print("=" * 60)
print("📝 SAMPLE CAPTION TỪ MODEL MỚI (sau 1 epoch)")
print("=" * 60)

model.eval()
for i in range(min(5, len(val_dataset))):
    sample = val_dataset[i]
    
    # Convert tensor
    pixel_values = sample["pixel_values"].unsqueeze(0).to(DEVICE) if sample["pixel_values"].ndim == 3 else torch.tensor(sample["pixel_values"]).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=60,
            num_beams=3,
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    gt = val_df.iloc[i]["caption_vi"]
    
    print(f"\n[{i+1}] GT:  {gt[:60]}")
    print(f"    Pred: {caption[:60]}")

print()
print("✅ Sanity check hoàn thành!")
print(f"📁 Model: {OUTPUT_DIR}")