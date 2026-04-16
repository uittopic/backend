"""
Fine-tune BLIP model cho tiếng Việt
Tối ưu cho macOS M1 Max / Apple Silicon với MPS backend
"""

from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from PIL import Image
import pandas as pd
import torch
import os
import inspect
import platform
import re
from pathlib import Path
from typing import Any, Dict, cast

# Bật fallback cho các op chưa hỗ trợ tốt trên MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

DEFAULT_TRAIN_CSV = DATA_DIR / "train_80.csv"
DEFAULT_VAL_CSV = DATA_DIR / "test_20.csv"
DEFAULT_BASE_DATASET_CSV = DATA_DIR / "train_bilingual_clean_v2.csv"
DEFAULT_IMAGE_DIR = DATA_DIR / "images"
DEFAULT_CAPTION_COLUMN = "caption_vi"
DEFAULT_MODEL_SUBDIR = "blip_vietnamese_80_20"
DEFAULT_OUTPUT_DIR = MODEL_DIR / DEFAULT_MODEL_SUBDIR

# Thêm support cho data cleaned v2
USE_CLEANED_V2 = os.environ.get("USE_CLEANED_V2", "false").strip().lower() == "true"
if USE_CLEANED_V2:
    DEFAULT_TRAIN_CSV = DATA_DIR / "train_80_cleaned_v2.csv"
    DEFAULT_VAL_CSV = DATA_DIR / "test_20.csv"
    DEFAULT_CAPTION_COLUMN = "caption_vi_cleaned_v2"
    DEFAULT_VAL_CAPTION_COLUMN = "caption_vi"
    print("⚙️  Sử dụng TRAIN data cleaned v2 (train_80_cleaned_v2.csv)")
    print("⚙️  Val/Test dùng cột: caption_vi")
else:
    DEFAULT_VAL_CAPTION_COLUMN = DEFAULT_CAPTION_COLUMN

# Tạo thư mục nếu chưa có
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def get_device() -> str:
    """Xác định device tối ưu cho macOS M1 / GPU / CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
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


def to_absolute_path(path: Path) -> Path:
    """Chuẩn hóa đường dẫn tương đối thành tuyệt đối theo BASE_DIR."""
    return path if path.is_absolute() else (BASE_DIR / path)


def load_or_create_splits(
    train_csv: Path,
    val_csv: Path,
    train_ratio: float,
    base_dataset_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/val CSV nếu có, nếu chưa có thì tạo từ dataset gốc."""
    if train_csv.exists() and val_csv.exists():
        print(f"✅ Đã tìm thấy split có sẵn:\n   • Train: {train_csv}\n   • Val/Test: {val_csv}")
        train_df = cast(pd.DataFrame, pd.read_csv(train_csv))
        val_df = cast(pd.DataFrame, pd.read_csv(val_csv))
        return train_df, val_df

    if not base_dataset_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset gốc tại {base_dataset_csv}. "
            "Vui lòng kiểm tra lại đường dẫn hoặc đồng bộ dữ liệu."
        )

    print("\n🆕 Chưa có split 80/20. Đang tạo mới từ dataset gốc...")
    df = cast(pd.DataFrame, pd.read_csv(base_dataset_csv))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError(f"train_ratio={train_ratio} tạo ra split rỗng. Hãy chọn giá trị trong (0,1).")

    train_df = cast(pd.DataFrame, df[:split_idx].copy())
    val_df = cast(pd.DataFrame, df[split_idx:].copy())

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(
        f"✅ Đã tạo train/test split:\n"
        f"   • Train ({len(train_df)} samples): {train_csv}\n"
        f"   • Test  ({len(val_df)} samples): {val_csv}"
    )
    return train_df, val_df


def normalize_dataframe(df: pd.DataFrame, caption_column: str, split_name: str) -> pd.DataFrame:
    """
    Làm sạch dữ liệu cơ bản trước khi đưa vào Trainer:
    - Bỏ dòng thiếu image hoặc thiếu caption
    - Ép caption về string
    """
    required_cols = {"image", caption_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV {split_name} thiếu cột bắt buộc: {sorted(missing_cols)}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df.copy()
    df["image"] = df["image"].astype(str).str.strip()
    df[caption_column] = df[caption_column].fillna("").astype(str).str.strip()

    before = len(df)
    valid_mask = cast(Any, (df["image"] != "") & (df[caption_column] != ""))
    df = df.loc[valid_mask].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"⚠️  {split_name}: đã loại {removed} dòng thiếu image/caption")

    return df


VI_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]"
)


def estimate_vietnamese_ratio(df: pd.DataFrame, caption_column: str) -> float:
    texts = df[caption_column].fillna("").astype(str)
    if len(texts) == 0:
        return 0.0
    has_vi = texts.map(lambda s: bool(VI_DIACRITIC_RE.search(str(s))))
    return float(has_vi.mean())


print("=" * 60)
print("🚀 Bắt đầu Fine-tune BLIP cho tiếng Việt")
print("=" * 60)

# === Load dataset ===
print("\n📂 Đang chuẩn bị dataset 80/20...")
TRAIN_CSV_PATH = to_absolute_path(resolve_csv_path("TRAIN_CSV_PATH", DEFAULT_TRAIN_CSV))
VAL_CSV_PATH = to_absolute_path(resolve_csv_path("VAL_CSV_PATH", DEFAULT_VAL_CSV))
BASE_DATASET_CSV = to_absolute_path(resolve_csv_path("BASE_DATASET_CSV", DEFAULT_BASE_DATASET_CSV))
IMAGE_DIR = to_absolute_path(resolve_csv_path("IMAGE_DIR", DEFAULT_IMAGE_DIR))

CAPTION_COLUMN = os.environ.get("CAPTION_COLUMN", DEFAULT_CAPTION_COLUMN).strip() or DEFAULT_CAPTION_COLUMN
VAL_CAPTION_COLUMN = os.environ.get("VAL_CAPTION_COLUMN", DEFAULT_VAL_CAPTION_COLUMN).strip() or DEFAULT_VAL_CAPTION_COLUMN
print(f"⚙️  Train caption column: {CAPTION_COLUMN}")
print(f"⚙️  Val/Test caption column: {VAL_CAPTION_COLUMN}")

TRAIN_RATIO = get_train_ratio()
model_output_env = os.environ.get("MODEL_OUTPUT_DIR")
MODEL_OUTPUT_DIR = to_absolute_path(Path(model_output_env) if model_output_env else DEFAULT_OUTPUT_DIR)

# Tạo output dir lúc save/checkpoint; không ép mkdir sớm để tránh nhầm resume/output trống
MODEL_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

device = get_device()

# Mặc định tối ưu cho Mac M1 Max 64GB, nhưng vẫn cho phép override qua env
default_train_bs = "4" if device == "mps" else "2"
default_eval_bs = "4" if device == "mps" else "2"
default_map_bs = "20" if device == "mps" else "10"
default_grad_acc = "2" if device == "mps" else "1"

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", default_train_bs))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", default_eval_bs))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "77"))
MAP_BATCH_SIZE = int(os.environ.get("MAP_BATCH_SIZE", default_map_bs))
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
MAX_VAL_SAMPLES = int(os.environ.get("MAX_VAL_SAMPLES", "0"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", default_grad_acc))
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "2"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "25"))
SAVE_STRATEGY = os.environ.get("SAVE_STRATEGY", "steps").strip().lower()
EVAL_STRATEGY = os.environ.get("EVAL_STRATEGY", SAVE_STRATEGY).strip().lower()
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "4"))
AUTO_RESUME = os.environ.get("AUTO_RESUME", "true").strip().lower() == "true"
OVERWRITE_OUTPUT_DIR = os.environ.get("OVERWRITE_OUTPUT_DIR", "false").strip().lower() == "true"
REQUIRE_VI_TARGET = os.environ.get("REQUIRE_VI_TARGET", "true").strip().lower() == "true"
VI_RATIO_THRESHOLD = float(os.environ.get("VI_RATIO_THRESHOLD", "0.60"))
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "false").strip().lower() == "true"

if SAVE_STRATEGY not in {"steps", "epoch"}:
    print(f"⚠️  SAVE_STRATEGY='{SAVE_STRATEGY}' không hợp lệ, dùng 'steps'")
    SAVE_STRATEGY = "steps"

if EVAL_STRATEGY not in {"steps", "epoch"}:
    print(f"⚠️  EVAL_STRATEGY='{EVAL_STRATEGY}' không hợp lệ, dùng theo SAVE_STRATEGY")
    EVAL_STRATEGY = SAVE_STRATEGY

if EVAL_STRATEGY != SAVE_STRATEGY:
    # load_best_model_at_end yêu cầu eval/save cùng strategy.
    print("⚠️  EVAL_STRATEGY khác SAVE_STRATEGY, tự động đồng bộ theo SAVE_STRATEGY")
    EVAL_STRATEGY = SAVE_STRATEGY

# Trên macOS, dataloader workers > 0 dễ phát sinh lỗi spawn với script train dạng này.
if platform.system() == "Darwin" and DATALOADER_NUM_WORKERS > 0:
    print("⚠️  macOS detected: ép DATALOADER_NUM_WORKERS=0 để tránh lỗi multiprocessing khi train.")
    DATALOADER_NUM_WORKERS = 0

print(f"📱 Device: {device}")
print(f"⚙️  Train CSV path: {TRAIN_CSV_PATH}")
print(f"⚙️  Val/Test CSV path: {VAL_CSV_PATH}")
print(f"⚙️  Base dataset path: {BASE_DATASET_CSV}")
print(f"⚙️  Image directory: {IMAGE_DIR}")
print(f"⚙️  Caption column: {CAPTION_COLUMN}")
print(f"⚙️  Train ratio: {TRAIN_RATIO:.2f}")
print(f"📦 Model output dir: {MODEL_OUTPUT_DIR}")
print(
    f"⚙️  Hyperparams: epochs={NUM_EPOCHS}, train_bs={TRAIN_BATCH_SIZE}, "
    f"eval_bs={EVAL_BATCH_SIZE}, lr={LEARNING_RATE}, warmup={WARMUP_STEPS}, max_len={MAX_LENGTH}"
)
print(
    f"⚙️  Runtime: grad_acc={GRADIENT_ACCUMULATION_STEPS}, workers={DATALOADER_NUM_WORKERS}, "
    f"map_bs={MAP_BATCH_SIZE}, save={SAVE_STRATEGY}, save_steps={SAVE_STEPS}, "
    f"eval={EVAL_STRATEGY}, eval_steps={EVAL_STEPS}, auto_resume={AUTO_RESUME}, "
    f"overwrite={OVERWRITE_OUTPUT_DIR}, compile={ENABLE_TORCH_COMPILE}"
)
print(f"⚙️  Language gate: require_vi={REQUIRE_VI_TARGET}, vi_ratio_threshold={VI_RATIO_THRESHOLD:.2f}")
if MAX_TRAIN_SAMPLES > 0 or MAX_VAL_SAMPLES > 0:
    print(f"⚙️  Sample caps: max_train={MAX_TRAIN_SAMPLES}, max_val={MAX_VAL_SAMPLES}")

train_df, val_df = load_or_create_splits(
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TRAIN_RATIO,
    BASE_DATASET_CSV,
)
train_df = cast(pd.DataFrame, train_df)
val_df = cast(pd.DataFrame, val_df)

train_df = normalize_dataframe(train_df, CAPTION_COLUMN, "train")
val_df = normalize_dataframe(val_df, VAL_CAPTION_COLUMN, "val/test")

if MAX_TRAIN_SAMPLES > 0:
    train_df = train_df.head(MAX_TRAIN_SAMPLES).copy()
if MAX_VAL_SAMPLES > 0:
    val_df = val_df.head(MAX_VAL_SAMPLES).copy()

vi_ratio_train = estimate_vietnamese_ratio(train_df, CAPTION_COLUMN)
print(f"🌐 Ước lượng tỷ lệ caption có dấu tiếng Việt (train): {vi_ratio_train:.4f}")
if REQUIRE_VI_TARGET and vi_ratio_train < VI_RATIO_THRESHOLD:
    raise ValueError(
        f"Caption column '{CAPTION_COLUMN}' có tỷ lệ tiếng Việt quá thấp "
        f"({vi_ratio_train:.4f} < {VI_RATIO_THRESHOLD:.2f}). "
        "Để đảm bảo output tiếng Việt cho Postman, hãy dùng cột caption tiếng Việt trước khi train."
    )

print(f"📈 Train samples: {len(train_df)}")
print(f"📈 Val/Test samples: {len(val_df)}")
print(f"📊 Các cột train: {train_df.columns.tolist()}")

# === Load BLIP model ===
print("\n🤖 Đang load BLIP model...")
processor_loaded = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
if isinstance(processor_loaded, tuple):
    processor_loaded = processor_loaded[0]
processor = cast(BlipProcessor, processor_loaded)

model_loaded = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
if isinstance(model_loaded, tuple):
    model_loaded = model_loaded[0]
model = cast(BlipForConditionalGeneration, model_loaded)

# Tối ưu phép nhân ma trận float32 cho tốc độ tốt hơn
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Optional: compile model để tận dụng M1 Max tốt hơn
if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        print("⚡ Đang compile model để tăng tốc training...")
        model = cast(BlipForConditionalGeneration, torch.compile(model))
        print("✅ torch.compile thành công")
    except Exception as exc:
        print(f"⚠️  torch.compile thất bại, bỏ qua: {exc}")

print("✅ Model đã được load")

# === Preprocess functions ===
def preprocess_train(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Xử lý batch ảnh và caption cho train (dùng CAPTION_COLUMN)."""
    images, texts = [], []

    for img_name, caption in zip(batch["image"], batch[CAPTION_COLUMN]):
        img_path = IMAGE_DIR / str(img_name)

        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            texts.append(str(caption))
        except Exception as exc:
            print(f"⚠️  Lỗi khi load ảnh {img_name}: {exc}")
            continue

    if len(images) == 0:
        return {}

    processor_callable = cast(Any, processor)
    inputs = processor_callable(
        images=images,
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    inputs["labels"] = labels
    return inputs


def preprocess_val(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Xử lý batch ảnh và caption cho val/test (dùng VAL_CAPTION_COLUMN)."""
    images, texts = [], []

    for img_name, caption in zip(batch["image"], batch[VAL_CAPTION_COLUMN]):
        img_path = IMAGE_DIR / str(img_name)

        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            texts.append(str(caption))
        except Exception as exc:
            print(f"⚠️  Lỗi khi load ảnh {img_name}: {exc}")
            continue

    if len(images) == 0:
        return {}

    processor_callable = cast(Any, processor)
    inputs = processor_callable(
        images=images,
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

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
    preprocess_train,
    batched=True,
    batch_size=MAP_BATCH_SIZE,
    remove_columns=train_dataset.column_names,
)

print("🔄 Đang preprocess val dataset...")
val_dataset = val_dataset.map(
    preprocess_val,
    batched=True,
    batch_size=MAP_BATCH_SIZE,
    remove_columns=val_dataset.column_names,
)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Val dataset: {len(val_dataset)} samples")

# === Training Arguments ===
training_kwargs = {
    "output_dir": str(MODEL_OUTPUT_DIR),
    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "warmup_steps": WARMUP_STEPS,
    "save_strategy": SAVE_STRATEGY,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "logging_dir": str(LOG_DIR),
    "logging_steps": LOGGING_STEPS,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": False,  # MPS chưa hỗ trợ fp16 ổn định
    "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    "dataloader_pin_memory": False,  # pin_memory không hiệu quả trên MPS
    "report_to": "none",
    "remove_unused_columns": False,
    "overwrite_output_dir": OVERWRITE_OUTPUT_DIR,
}

# Transformers mới dùng eval_strategy, bản cũ dùng evaluation_strategy
training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in training_arg_params:
    training_kwargs["evaluation_strategy"] = EVAL_STRATEGY
elif "eval_strategy" in training_arg_params:
    training_kwargs["eval_strategy"] = EVAL_STRATEGY

if SAVE_STRATEGY == "steps":
    training_kwargs["save_steps"] = SAVE_STEPS

if EVAL_STRATEGY == "steps":
    training_kwargs["eval_steps"] = EVAL_STEPS

if "use_mps_device" in training_arg_params and device == "mps":
    training_kwargs["use_mps_device"] = True

training_args = TrainingArguments(**training_kwargs)

# === Trainer ===
print("\n🏋️  Đang khởi tạo Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Tránh truyền num_items_in_batch vào model.forward của BLIP
if hasattr(trainer, "model_accepts_loss_kwargs"):
    setattr(trainer, "model_accepts_loss_kwargs", False)

# === Train ===
print("\n🚀 Bắt đầu training...")
print("=" * 60)
resume_checkpoint = None
if AUTO_RESUME:
    resume_checkpoint = get_last_checkpoint(str(MODEL_OUTPUT_DIR))
    if resume_checkpoint:
        print(f"♻️  Tìm thấy checkpoint gần nhất, sẽ resume từ: {resume_checkpoint}")
    else:
        print("🆕 Không có checkpoint cũ, bắt đầu train mới.")

if resume_checkpoint:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    trainer.train()

# === Save model ===
print("\n💾 Đang lưu model...")
final_model_path = MODEL_OUTPUT_DIR
model.save_pretrained(str(final_model_path))
processor.save_pretrained(str(final_model_path))

print("=" * 60)
print("✅ Fine-tune BLIP Vietnamese hoàn thành!")
print(f"📁 Model đã được lưu tại: {final_model_path}")
print("=" * 60)