"""
Train BLIP voi Cleaned Data - Phien ban MEMORY EFFICIENT
Load anh on-the-fly thay vi load het vao RAM
"""
import os
import platform
from pathlib import Path
from typing import cast

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

torch.set_float32_matmul_precision("high")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

TRAIN_CSV = DATA_DIR / "train_80_cleaned.csv"
VAL_CSV = DATA_DIR / "test_20.csv"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = MODEL_DIR / "blip_vietnamese_cleaned_v1"
CAPTION_COLUMN = "caption_vi_cleaned"

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
MAX_LENGTH = 60
MAX_TRAIN_SAMPLES = None  # FULL DATA
MAX_VAL_SAMPLES = 50

DATALOADER_NUM_WORKERS = 0

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
TRAINING_OUTPUT_DIR = LOG_DIR / "train_cleaned_v1_runs"
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ImageCaptionDataset(Dataset):
    """Dataset load anh on-the-fly, khong load het vao RAM"""
    
    def __init__(self, df, caption_col, processor, max_length=60):
        self.df = df.reset_index(drop=True)
        self.caption_col = caption_col
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load anh on-the-fly
        img_path = IMAGE_DIR / row["image"]
        img = Image.open(img_path).convert("RGB")
        
        caption = str(row[self.caption_col])
        
        # Preprocess
        pixel_vals = self.processor.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
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


print("=" * 60)
print("TRAIN BLIP CLEANED - MEMORY EFFICIENT")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_df = normalize_dataframe(train_df, CAPTION_COLUMN)
val_df = normalize_dataframe(val_df, "caption_vi")

if MAX_TRAIN_SAMPLES:
    train_df = train_df.head(MAX_TRAIN_SAMPLES)
if MAX_VAL_SAMPLES:
    val_df = val_df.head(MAX_VAL_SAMPLES)

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")

DEVICE = get_device()
print("Device:", DEVICE)

# Xoa checkpoint cu de train moi hoan toan
import shutil
if TRAINING_OUTPUT_DIR.exists():
    for item in TRAINING_OUTPUT_DIR.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            shutil.rmtree(item)
            print(f"Da xoa checkpoint cu: {item}")

# Load model & processor
print("Loading model...")
model_name = "Salesforce/blip-image-captioning-base"
processor = cast(BlipProcessor, BlipProcessor.from_pretrained(model_name))
device = torch.device(get_device())
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)  # type: ignore

# Create datasets (lazy loading - khong load anh vao RAM)
print("Creating datasets...")
train_dataset = ImageCaptionDataset(train_df, CAPTION_COLUMN, processor, MAX_LENGTH)
val_dataset = ImageCaptionDataset(val_df, "caption_vi", processor, MAX_LENGTH)

print(f"Train dataset: {len(train_dataset)}")
print(f"Val dataset: {len(val_dataset)}")

training_args = TrainingArguments(
    output_dir=str(TRAINING_OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    report_to="none",
    gradient_accumulation_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nStarting training...")
print("=" * 60)

last_checkpoint = get_last_checkpoint(str(TRAINING_OUTPUT_DIR))
if last_checkpoint is not None:
    print(f"Resume from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Save model
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 60)

# Sample output
print("\n=== SAMPLE OUTPUT ===")
model.eval()
for i in range(min(5, len(val_dataset))):
    sample = val_dataset[i]
    pv = torch.tensor(sample["pixel_values"], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(
            pixel_values=pv,
            max_new_tokens=60,
            num_beams=3
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    gt = val_df.iloc[i]["caption_vi"]
    
    print(f"\n[{i+1}] GT:   {gt[:80]}")
    print(f"    Pred: {caption[:80]}")
