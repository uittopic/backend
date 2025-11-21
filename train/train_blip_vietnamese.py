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
DEFAULT_TRAIN_CSV = DATA_DIR / "train_80.csv"
DEFAULT_VAL_CSV = DATA_DIR / "test_20.csv"
BASE_DATASET_CSV = DATA_DIR / "train_bilingual_clean_v2.csv"
DEFAULT_MODEL_SUBDIR = "blip_vietnamese_80_20"
DEFAULT_OUTPUT_DIR = MODEL_DIR / DEFAULT_MODEL_SUBDIR

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


def get_train_ratio() -> float:
    """Đọc tỉ lệ train từ env (mặc định 80%)."""
    env_value = os.environ.get("TRAIN_SPLIT_RATIO", "0.8")
    try:
        ratio = float(env_value)
    except ValueError:
        print(f"⚠️  TRAIN_SPLIT_RATIO='{env_value}' không hợp lệ, dùng mặc định 0.8")
        return 0.8

    if not 0.0 < ratio < 1.0:
        print(f"⚠️  TRAIN_SPLIT_RATIO={ratio} nằm ngoài (0,1), dùng mặc định 0.8")
        return 0.8
    return ratio


def resolve_csv_path(env_key: str, default_path: Path) -> Path:
    """Lấy đường dẫn CSV từ env nếu có, ngược lại dùng default."""
    env_value = os.environ.get(env_key)
    return Path(env_value) if env_value else default_path


def load_or_create_splits(train_csv: Path, val_csv: Path, train_ratio: float):
    """Load train/val CSV nếu có, nếu chưa có thì tạo từ dataset gốc."""
    if train_csv.exists() and val_csv.exists():
        print(f"✅ Đã tìm thấy split có sẵn:\n   • Train: {train_csv}\n   • Val/Test: {val_csv}")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        return train_df, val_df

    if not BASE_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset gốc tại {BASE_DATASET_CSV}. "
            "Vui lòng kiểm tra lại đường dẫn hoặc đồng bộ dữ liệu."
        )

    print("\n🆕 Chưa có split 80/20. Đang tạo mới từ dataset gốc...")
    df = pd.read_csv(BASE_DATASET_CSV)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError(f"train_ratio={train_ratio} tạo ra split rỗng. Hãy chọn giá trị trong (0,1).")

    train_df = df[:split_idx].copy()
    val_df = df[split_idx:].copy()

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"✅ Đã tạo train/test split:\n   • Train ({len(train_df)} samples): {train_csv}\n   • Test  ({len(val_df)} samples): {val_csv}")
    return train_df, val_df

print("=" * 60)
print("🚀 Bắt đầu Fine-tune BLIP cho tiếng Việt")
print("=" * 60)

# === Load dataset ===
print("\n📂 Đang chuẩn bị dataset 80/20...")
TRAIN_CSV_PATH = resolve_csv_path("TRAIN_CSV_PATH", DEFAULT_TRAIN_CSV)
VAL_CSV_PATH = resolve_csv_path("VAL_CSV_PATH", DEFAULT_VAL_CSV)
TRAIN_RATIO = get_train_ratio()
MODEL_OUTPUT_DIR = Path(os.environ.get("MODEL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

print(f"⚙️  Train CSV path: {TRAIN_CSV_PATH}")
print(f"⚙️  Val/Test CSV path: {VAL_CSV_PATH}")
print(f"⚙️  Train ratio: {TRAIN_RATIO:.2f}")
print(f"📦 Model output dir: {MODEL_OUTPUT_DIR}")

train_df, val_df = load_or_create_splits(TRAIN_CSV_PATH, VAL_CSV_PATH, TRAIN_RATIO)

print(f"📈 Train samples: {len(train_df)}")
print(f"📈 Val/Test samples: {len(val_df)}")
print(f"📊 Các cột: {train_df.columns.tolist()}")

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

    # Labels cho training (bỏ qua padding tokens khi tính loss)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    inputs["labels"] = labels
    
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
    output_dir=str(MODEL_OUTPUT_DIR),
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
    remove_unused_columns=False,
    overwrite_output_dir=True,
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
final_model_path = MODEL_OUTPUT_DIR
model.save_pretrained(str(final_model_path))
processor.save_pretrained(str(final_model_path))

print("=" * 60)
print(f"✅ Fine-tune BLIP Vietnamese hoàn thành!")
print(f"📁 Model đã được lưu tại: {final_model_path}")
print("=" * 60)

