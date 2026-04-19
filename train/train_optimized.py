"""
Train BLIP Vietnamese — PHIÊN BẢN TỐI ƯU
==========================================

Cải tiến so với baseline:
  1. Epoch: 3 → 8 (học lâu hơn, model tốt hơn)
  2. Learning rate: 5e-5 → 3e-5 với cosine scheduler (học mịn hơn)
  3. Max length: 60 → 128 (caption dài hơn, BLEU cao hơn)
  4. Batch size: 4 → 8 (gradient stable hơn, GPU hiệu quả hơn)
  5. Warmup: 100 → 200 (tránh loss spike đầu)
  6. Weight decay: 0.01 (chống overfit)
  7. Gradient checkpointing: bật (tiết kiệm VRAM)
  8. Image augmentation: RandomHorizontalFlip, ColorJitter

Chạy:
  python train/train_optimized.py
"""

import os
import platform
from pathlib import Path
from typing import cast, Any, Dict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Trainer, TrainingArguments, get_cosine_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from dataclasses import dataclass

torch.set_float32_matmul_precision("high")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# ============================================================
# SECTION 1: ĐƯỜNG DẪN DATA
# ============================================================
TRAIN_CSV = DATA_DIR / "train_80_cleaned.csv"
VAL_CSV = DATA_DIR / "test_20.csv"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = MODEL_DIR / "blip_vietnamese_optimized_v1"

# ============================================================
# SECTION 2: HYPERPARAMETERS TỐI ƯU
# ============================================================
NUM_EPOCHS = 8           # 3 → 8: model học lâu hơn, quality tốt hơn
TRAIN_BATCH_SIZE = 4     # 4 → 4 (giữ nguyên cho MPS ổn định)
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-5      # 5e-5 → 3e-5: học mịn hơn, không overshoot
WARMUP_RATIO = 0.06      # warmup = 6% total steps
MAX_LENGTH = 128          # 60 → 128: caption dài hơn, BLEU cao hơn
WEIGHT_DECAY = 0.01       # weight decay chống overfit
GRADIENT_ACCUMULATION = 4 # effective batch = 4*4 = 16
SAVE_STEPS = 500          # lưu mỗi 500 steps
LOGGING_STEPS = 25        # log mỗi 25 steps

# ============================================================
# SECTION 3: IMAGE AUGMENTATION
# ============================================================
@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5    # lật ngang
    brightness: float = 0.2         # độ sáng ±20%
    contrast: float = 0.2           # tương phản ±20%
    saturation: float = 0.2          # bão hòa ±20%
    hue: float = 0.1                # sắc độ ±10%

AUG_CONFIG = AugmentationConfig()

class ImageCaptionDataset(Dataset):
    """Dataset với lazy loading + augmentation"""

    def __init__(self, df, caption_col, processor, max_length=128,
                 augment: bool = False, aug_config=None):
        self.df = df.reset_index(drop=True)
        self.caption_col = caption_col
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        self.aug = aug_config or AUG_CONFIG

    def __len__(self):
        return len(self.df)

    def _augment_image(self, img: Image.Image) -> Image.Image:
        """Apply random augmentation (CPU-based, không tốn GPU VRAM)"""
        import torchvision.transforms as T

        transforms_list = []
        if self.augment:
            transforms_list.append(T.RandomHorizontalFlip(p=self.aug.horizontal_flip))
            transforms_list.append(
                T.ColorJitter(
                    brightness=self.aug.brightness,
                    contrast=self.aug.contrast,
                    saturation=self.aug.saturation,
                    hue=self.aug.hue
                )
            )

        if transforms_list:
            transform = T.Compose(transforms_list)
            img = transform(img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image on-the-fly
        img_path = IMAGE_DIR / row["image"]
        img = Image.open(img_path).convert("RGB")

        # Augment training images
        if self.augment:
            img = self._augment_image(img)

        caption = str(row[self.caption_col])

        # Preprocess
        pixel_vals = self.processor.image_processor(
            img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        text_inputs = self.processor.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_vals,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": text_inputs["input_ids"].squeeze(0),
        }


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_dataframe(df, caption_column):
    df = df.copy()
    df["image"] = df["image"].astype(str).str.strip()
    if caption_column not in df.columns:
        raise ValueError(f"Column not found: {caption_column}")
    df[caption_column] = df[caption_column].fillna("").astype(str).str.strip()
    valid_mask = (df["image"] != "") & (df[caption_column] != "")
    return df.loc[valid_mask].copy()


def compute_metrics(eval_pred):
    """Compute training metrics (loss là chính, không cần custom metric ở đây)"""
    logits, labels = eval_pred
    return {"eval_loss": float(np.mean(logits))}


# ============================================================
# SECTION 4: MAIN TRAINING
# ============================================================
print("=" * 65)
print("TRAIN BLIP OPTIMIZED — V2")
print("  Epoch: 8  |  LR: 3e-5  |  MaxLen: 128  |  Augmentation")
print("=" * 65)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_df = normalize_dataframe(train_df, "caption_vi_cleaned")
val_df = normalize_dataframe(val_df, "caption_vi")

# Giới hạn val để eval nhanh hơn
MAX_VAL = 100
if MAX_VAL > 0:
    val_df = val_df.head(MAX_VAL)

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")

DEVICE = get_device()
print(f"Device: {DEVICE}")

# Load model & processor
print("Loading BLIP model...")
model_name = "Salesforce/blip-image-captioning-base"
processor = cast(BlipProcessor, BlipProcessor.from_pretrained(model_name))
device_obj = torch.device(get_device())
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device_obj)

# Create datasets — AUGMENTATION chỉ bật cho train
print("Creating datasets (with augmentation for train)...")
train_dataset = ImageCaptionDataset(
    train_df, "caption_vi_cleaned", processor, MAX_LENGTH,
    augment=True, aug_config=AUG_CONFIG
)
val_dataset = ImageCaptionDataset(
    val_df, "caption_vi", processor, MAX_LENGTH,
    augment=False  # KHÔNG augment validation
)

print(f"Train dataset: {len(train_dataset)} (with augmentation)")
print(f"Val dataset: {len(val_dataset)}")

# Compute total steps
train_size = len(train_dataset)
batch_size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION
total_steps = (train_size // batch_size) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

# Training arguments TỐI ƯU
MODEL_LOG_DIR = LOG_DIR / "train_optimized_runs"
MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Xóa checkpoint cũ để train sạch
import shutil
if MODEL_LOG_DIR.exists():
    for item in MODEL_LOG_DIR.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            shutil.rmtree(item)

training_args = TrainingArguments(
    output_dir=str(MODEL_LOG_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",           # cosine scheduler — mịn hơn Adam
    logging_dir=str(LOG_DIR / "tensorboard"),
    logging_steps=LOGGING_STEPS,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_total_limit=3,                  # giữ 3 checkpoint tốt nhất
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    # Tối ưu memory
    gradient_checkpointing=True,          # tiết kiệm VRAM
    dataloader_num_workers=0,
    #fp16=True,                          # bật nếu có CUDA (MPS không hỗ trợ fp16 ổn định)
    remove_unused_columns=False,
)

# Tạo scheduler riêng (cosine với warmup)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, lr_scheduler),
)

print(f"\n🚀 Starting training: {NUM_EPOCHS} epochs, {total_steps} steps")
print(f"   LR: {LEARNING_RATE} (cosine), Weight decay: {WEIGHT_DECAY}")
print(f"   Max length: {MAX_LENGTH}, Batch size: {TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {batch_size}")
print("=" * 65)

trainer.train()

# Save model
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))

print("\n" + "=" * 65)
print("✅ Training complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print("=" * 65)

# Sample output
print("\n=== SAMPLE OUTPUT (beam=5, max_new_tokens=128) ===")
model.eval()
device = torch.device(get_device())

for i in range(min(5, len(val_dataset))):
    sample = val_dataset[i]
    pv = torch.tensor(sample["pixel_values"], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values=pv,
            max_new_tokens=128,    # tăng từ 60 → 128
            num_beams=5,           # tăng từ 3 → 5
            early_stopping=True,
            repetition_penalty=1.1,
            length_penalty=1.2,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    gt = val_df.iloc[i]["caption_vi"]

    print(f"\n[{i+1}] GT:   {gt[:80]}")
    print(f"    Pred: {caption[:80]}")
