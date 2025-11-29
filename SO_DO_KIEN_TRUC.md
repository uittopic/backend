# 📐 SƠ ĐỒ KIẾN TRÚC HỆ THỐNG

> Tài liệu này chứa các sơ đồ kiến trúc hệ thống chi tiết để sử dụng trong báo cáo chuyên đề.

---

## 1. SYSTEM ARCHITECTURE DIAGRAM

### 1.1. Kiến trúc Tổng thể (High-Level)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Web Browser │    │  Mobile App  │    │  API Client  │          │
│  │  (React/Vue) │    │ (iOS/Android)│    │  (Python/JS) │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
└─────────┼────────────────────┼────────────────────┼──────────────────┘
          │                    │                    │
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    HTTP/REST API (JSON, Multipart)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              FastAPI Application Server                      │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  Routes Handler                                       │  │   │
│  │  │  • POST /api/caption                                  │  │   │
│  │  │  • POST /api/caption/batch                            │  │   │
│  │  │  • POST /api/caption_full                             │  │   │
│  │  │  • POST /api/caption_full/batch                       │  │   │
│  │  │  • POST /api/accent/restore                          │  │   │
│  │  │  • GET  /api/health                                  │  │   │
│  │  │  • POST /api/cache/clear                             │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  Middleware                                            │  │   │
│  │  │  • CORS Handler                                        │  │   │
│  │  │  • Authentication (Optional)                           │  │   │
│  │  │  • Rate Limiting                                       │  │   │
│  │  │  • Error Handling                                      │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Caption Service                                  │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │  Image Preprocessor  │  │   Cache Manager      │         │   │
│  │  │  • Validate format   │  │   • Hash generation  │         │   │
│  │  │  • Resize (>512px)   │  │   • Cache lookup     │         │   │
│  │  │  • Convert RGB       │  │   • Cache store      │         │   │
│  │  │  • Normalize         │  │   • TTL management   │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  │                                                               │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │  Orchestration Logic                                 │    │   │
│  │  │  • Coordinate BLIP + Accent Restoration             │    │   │
│  │  │  • Error handling & fallback                         │    │   │
│  │  │  • Response formatting                               │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│      MODEL LAYER              │  │  ACCENT RESTORATION LAYER     │
│  ┌─────────────────────────┐  │  │  ┌─────────────────────────┐ │
│  │  BLIP Model             │  │  │  │  XLM-RoBERTa Model      │ │
│  │  ┌───────────────────┐  │  │  │  │  ┌───────────────────┐ │ │
│  │  │ Vision Encoder    │  │  │  │  │  │ Token Classifier  │ │ │
│  │  │ (ViT-B/16)        │  │  │  │  │  │ • Predict labels   │ │ │
│  │  └─────────┬─────────┘  │  │  │  │  └───────────────────┘ │ │
│  │            │            │  │  │  │                        │ │
│  │            ▼            │  │  │  │  ┌───────────────────┐ │ │
│  │  ┌───────────────────┐  │  │  │  │  │ Token Merger      │ │ │
│  │  │ Cross-Modal       │  │  │  │  │  │ • Merge subwords   │ │ │
│  │  │ Attention         │  │  │  │  │  └───────────────────┘ │ │
│  │  └─────────┬─────────┘  │  │  │  │                        │ │
│  │            │            │  │  │  │  ┌───────────────────┐ │ │
│  │            ▼            │  │  │  │  │ Label Mapper      │ │ │
│  │  ┌───────────────────┐  │  │  │  │  │ • Apply accents   │ │ │
│  │  │ Text Decoder      │  │  │  │  │  └───────────────────┘ │ │
│  │  │ (BERT-based)     │  │  │  │  └─────────────────────────┘ │
│  │  └───────────────────┘  │  │  └───────────────────────────────┘
│  │                         │  │
│  │  ┌───────────────────┐  │  │
│  │  │ BLIP Processor    │  │  │
│  │  │ • Image process   │  │  │
│  │  │ • Tokenize        │  │  │
│  │  └───────────────────┘  │  │
│  └─────────────────────────┘  │
│                                │
│  ┌─────────────────────────┐  │
│  │ Model Loader            │  │
│  │ • Load weights          │  │
│  │ • Device management     │  │
│  │ • Memory optimization   │  │
│  └─────────────────────────┘  │
└────────────────────────────────┘
```

### 1.2. Kiến trúc Chi tiết (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL LAYER                                │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   Client 1   │  │   Client 2   │  │   Client N   │                 │
│  │  (Web/Mobile)│  │  (Web/Mobile)│  │  (Web/Mobile)│                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
└─────────┼──────────────────┼──────────────────┼─────────────────────────┘
          │                  │                  │
          │  HTTP/REST       │  HTTP/REST       │  HTTP/REST
          │  (Port 8000)     │  (Port 8000)     │  (Port 8000)
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI APPLICATION                              │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Request Handler                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │  │
│  │  │   Router     │  │  Middleware  │  │  Exception   │         │  │
│  │  │  (Routes)    │  │  (CORS/Auth) │  │  Handler     │         │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │  │
│  └─────────┼──────────────────┼──────────────────┼─────────────────┘  │
└────────────┼──────────────────┼──────────────────┼─────────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                                      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Caption Service                                                  │  │
│  │  ┌──────────────────┐  ┌──────────────────┐                     │  │
│  │  │ Image Processor  │  │  Cache Service   │                     │  │
│  │  │                  │  │                  │                     │  │
│  │  │ • Validate       │  │  • Hash (MD5)    │                     │  │
│  │  │ • Resize         │  │  • Lookup        │                     │  │
│  │  │ • Convert        │  │  • Store         │                     │  │
│  │  │ • Normalize      │  │  • TTL Check     │                     │  │
│  │  └──────────────────┘  └──────────────────┘                     │  │
│  │                                                                   │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │  Pipeline Orchestrator                                    │   │  │
│  │  │  1. Check Cache → Return if found                         │   │  │
│  │  │  2. Preprocess Image                                      │   │  │
│  │  │  3. BLIP Generation (no accent)                          │   │  │
│  │  │  4. Accent Restoration                                    │   │  │
│  │  │  5. Cache Result                                          │   │  │
│  │  │  6. Format Response                                       │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│  BLIP MODEL COMPONENT             │  │  ACCENT MODEL COMPONENT          │
│                                   │  │                                   │
│  ┌─────────────────────────────┐  │  │  ┌─────────────────────────────┐ │
│  │  Model Loader               │  │  │  │  Model Loader               │ │
│  │  • Load from disk/HF        │  │  │  │  • Load from HuggingFace    │ │
│  │  • Device assignment        │  │  │  │  • Device assignment        │ │
│  │  • Memory optimization      │  │  │  │  • Memory optimization      │ │
│  └──────────────┬──────────────┘  │  │  └──────────────┬──────────────┘ │
│                 │                 │  │                 │                │
│                 ▼                 │  │                 ▼                │
│  ┌─────────────────────────────┐  │  │  ┌─────────────────────────────┐ │
│  │  BLIP Processor             │  │  │  │  XLM-RoBERTa Tokenizer     │ │
│  │  • Image → Tensor           │  │  │  │  • Text → Tokens            │ │
│  │  • Tokenize                 │  │  │  │  • Add special tokens      │ │
│  └──────────────┬──────────────┘  │  │  └──────────────┬──────────────┘ │
│                 │                 │  │                 │                │
│                 ▼                 │  │                 ▼                │
│  ┌─────────────────────────────┐  │  │  ┌─────────────────────────────┐ │
│  │  Vision Encoder (ViT)       │  │  │  │  XLM-RoBERTa Model          │ │
│  │  • Patch embedding          │  │  │  │  • Forward pass             │ │
│  │  • Transformer layers       │  │  │  │  • Get logits               │ │
│  │  • Image features           │  │  │  └──────────────┬──────────────┘ │
│  └──────────────┬──────────────┘  │  │                 │                │
│                 │                 │  │                 ▼                │
│                 ▼                 │  │  ┌─────────────────────────────┐ │
│  ┌─────────────────────────────┐  │  │  │  Token Classification       │ │
│  │  Cross-Modal Attention      │  │  │  │  • Argmax labels           │ │
│  │  • Image-Text alignment     │  │  │  │  • Merge subword tokens     │ │
│  └──────────────┬──────────────┘  │  │  │  • Apply accent patterns   │ │
│                 │                 │  │  └─────────────────────────────┘ │
│                 ▼                 │  │                                   │
│  ┌─────────────────────────────┐  │  │                                   │
│  │  Text Decoder (BERT)        │  │  │                                   │
│  │  • Autoregressive gen       │  │  │                                   │
│  │  • Beam search              │  │  │                                   │
│  │  • Caption (no accent)      │  │  │                                   │
│  └─────────────────────────────┘  │  │                                   │
│                                   │  │                                   │
│  Output: "ao khoac the thao..."   │  │  Output: "áo khoác thể thao..."   │
└───────────────────────────────────┘  └───────────────────────────────────┘
```

---

## 2. DATA FLOW DIAGRAM

### 2.1. Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: Dataset CSV (7,638 samples)                                │
│  File: train_bilingual_clean_v2.csv                                │
│  Format: image, caption_vi                                          │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Data Splitting        │
                    │  • Shuffle (seed=42)   │
                    │  • Split 80/20         │
                    └────────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Train Set           │  │  Test Set              │
        │  6,110 samples (80%) │  │  1,528 samples (20%)  │
        │  train_80.csv        │  │  test_20.csv          │
        └──────────┬───────────┘  └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Data Preprocessing  │
        │  • Load images       │
        │  • Convert RGB       │
        │  • Resize 224×224    │
        │  • Tokenize captions │
        │  • Create labels     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Model Initialization│
        │  • Load pretrained   │
        │    BLIP              │
        │  • Move to device    │
        │  • Set train mode    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Training Loop       │
        │  For each epoch:     │
        │    • Forward pass    │
        │    • Calculate loss  │
        │    • Backward pass   │
        │    • Update weights  │
        │    • Evaluate        │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Model Saving        │
        │  • Save weights      │
        │  • Save config       │
        │  • Save tokenizer    │
        │  Output:             │
        │  models/blip_        │
        │  vietnamese_80_20/   │
        └──────────────────────┘
```

### 2.2. Inference Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: Image File (JPEG/PNG)                                       │
│  Source: Client upload via API                                      │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Image Validation      │
                    │  • Check format        │
                    │  • Check size          │
                    │  • Validate content    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Cache Check           │
                    │  • Hash image (MD5)    │
                    │  • Lookup in cache     │
                    └────────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            Cache Hit?                    Cache Miss?
                    │                         │
                    │                         ▼
                    │            ┌────────────────────────┐
                    │            │  Image Preprocessing   │
                    │            │  • Resize if >512px   │
                    │            │  • Convert RGB        │
                    │            │  • Normalize          │
                    │            └────────────┬──────────┘
                    │                         │
                    │                         ▼
                    │            ┌────────────────────────┐
                    │            │  BLIP Generation       │
                    │            │  • Process image      │
                    │            │  • Vision encoder     │
                    │            │  • Text decoder       │
                    │            │  • Beam search        │
                    │            │  Output: Caption      │
                    │            │    (no accent)        │
                    │            └────────────┬──────────┘
                    │                         │
                    │                         ▼
                    │            ┌────────────────────────┐
                    │            │  Accent Restoration    │
                    │            │  • Tokenize            │
                    │            │  • Predict accents    │
                    │            │  • Merge tokens        │
                    │            │  • Apply accents       │
                    │            │  Output: Caption      │
                    │            │    (with accent)       │
                    │            └────────────┬───────────┘
                    │                         │
                    │                         ▼
                    │            ┌────────────────────────┐
                    │            │  Cache Storage         │
                    │            │  • Store result        │
                    │            │  • Set TTL (24h)       │
                    │            └────────────┬───────────┘
                    │                         │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Response Formatting    │
                    │  • Format JSON          │
                    │  • Add metadata         │
                    │  • Calculate time      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  OUTPUT: JSON Response  │
                    │  {                     │
                    │    "success": true,    │
                    │    "caption_vi": "...",│
                    │    "processing_time":  │
                    │  }                     │
                    └────────────────────────┘
```

### 2.3. Batch Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: Multiple Images (max 10)                                    │
│  files: [image1.jpg, image2.jpg, ..., imageN.jpg]                  │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Validate Batch        │
                    │  • Check count ≤ 10    │
                    │  • Validate each file  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Initialize Results    │
                    │  results = []         │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  For each image:       │
                    │  ┌──────────────────┐ │
                    │  │ 1. Check cache   │ │
                    │  │ 2. Preprocess    │ │
                    │  │ 3. BLIP gen      │ │
                    │  │ 4. Accent restore│ │
                    │  │ 5. Cache result  │ │
                    │  │ 6. Append result  │ │
                    │  └──────────────────┘ │
                    │                        │
                    │  Every 5 images:      │
                    │  • Cleanup memory     │
                    │  • Clear device cache │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Aggregate Results     │
                    │  • Combine all results │
                    │  • Calculate total    │
                    │    processing time     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  OUTPUT: Batch Response │
                    │  {                     │
                    │    "success": true,    │
                    │    "total": N,         │
                    │    "results": [...],   │
                    │    "processing_time":  │
                    │  }                     │
                    └────────────────────────┘
```

---

## 3. SEQUENCE DIAGRAM

### 3.1. Single Caption Request Sequence

```
Client          API Gateway      Caption Service    Cache        BLIP Model    Accent Model
  │                  │                  │            │              │              │
  │  POST /api/      │                  │            │              │              │
  │  caption_full    │                  │            │              │              │
  │  (image)         │                  │            │              │              │
  ├─────────────────>│                  │            │              │              │
  │                  │                  │            │              │              │
  │                  │  Validate        │            │              │              │
  │                  │  Request         │            │              │              │
  │                  ├─────────────────>│            │              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Hash      │              │              │
  │                  │                  │  Image     │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Check     │              │              │
  │                  │                  │  Cache     │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Cache     │              │              │
  │                  │                  │  Miss      │              │              │
  │                  │                  │<───────────┤              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Preprocess│              │              │
  │                  │                  │  Image     │              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Generate  │              │              │
  │                  │                  │  Caption   │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │            │  Process     │              │
  │                  │                  │            │  Image       │              │
  │                  │                  │            │              │              │
  │                  │                  │            │  Generate    │              │
  │                  │                  │            │  Caption     │              │
  │                  │                  │            │  (no accent) │              │
  │                  │                  │            │<─────────────┤              │
  │                  │                  │            │              │              │
  │                  │                  │  Caption   │              │              │
  │                  │                  │  (no accent)              │              │
  │                  │                  │<───────────┤              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Restore   │              │              │
  │                  │                  │  Accent    │              │              │
  │                  │                  ├──────────────────────────>│              │
  │                  │                  │            │              │              │
  │                  │                  │            │  Tokenize    │              │
  │                  │                  │            │  & Predict   │              │
  │                  │                  │            │              │              │
  │                  │                  │            │  Caption     │              │
  │                  │                  │            │  (with accent)              │
  │                  │                  │            │<───────────────────────────┤
  │                  │                  │            │              │              │
  │                  │                  │  Caption   │              │              │
  │                  │                  │  (with accent)            │              │
  │                  │                  │<───────────────────────────┤              │
  │                  │                  │            │              │              │
  │                  │                  │  Store in  │              │              │
  │                  │                  │  Cache     │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Format    │              │              │
  │                  │                  │  Response  │              │              │
  │                  │                  │            │              │              │
  │                  │  JSON Response  │              │              │              │
  │                  │<─────────────────┤            │              │              │
  │                  │                  │            │              │              │
  │  JSON Response   │                  │            │              │              │
  │<─────────────────┤                  │            │              │              │
  │                  │                  │            │              │              │
```

### 3.2. Cached Request Sequence (Fast Path)

```
Client          API Gateway      Caption Service    Cache        BLIP Model    Accent Model
  │                  │                  │            │              │              │
  │  POST /api/      │                  │            │              │              │
  │  caption_full    │                  │            │              │              │
  │  (image)         │                  │            │              │              │
  ├─────────────────>│                  │            │              │              │
  │                  │                  │            │              │              │
  │                  │  Validate        │            │              │              │
  │                  │  Request         │            │              │              │
  │                  ├─────────────────>│            │              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Hash      │              │              │
  │                  │                  │  Image     │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Check     │              │              │
  │                  │                  │  Cache     │              │              │
  │                  │                  ├───────────>│              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Cache     │              │              │
  │                  │                  │  Hit!      │              │              │
  │                  │                  │<───────────┤              │              │
  │                  │                  │            │              │              │
  │                  │                  │  Format    │              │              │
  │                  │                  │  Response  │              │              │
  │                  │                  │            │              │              │
  │                  │  JSON Response  │              │              │              │
  │                  │  (cached=true)  │              │              │              │
  │                  │<─────────────────┤            │              │              │
  │                  │                  │            │              │              │
  │  JSON Response   │                  │            │              │              │
  │  (fast: ~0.01s)  │                  │            │              │              │
  │<─────────────────┤                  │            │              │              │
  │                  │                  │            │              │              │
```

---

## 4. COMPONENT DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BLIP VIETNAMESE CAPTIONING API                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│  API COMPONENT                 │  │  SERVICE COMPONENT            │
│                                │  │                                │
│  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │  FastAPI App             │  │  │  │  Caption Service        │ │
│  │  • Routes                │  │  │  │  • Image preprocessing  │ │
│  │  • Middleware            │  │  │  │  • Pipeline orchestration│ │
│  │  • Error handling        │  │  │  │  • Response formatting   │ │
│  └──────────────────────────┘  │  │  └──────────────────────────┘ │
│                                │  │                                │
│  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │  Route Handlers          │  │  │  │  Cache Service          │ │
│  │  • /api/caption          │  │  │  │  • Hash generation       │ │
│  │  • /api/caption/batch    │  │  │  │  • Cache lookup          │ │
│  │  • /api/caption_full     │  │  │  │  • Cache storage         │ │
│  │  • /api/health           │  │  │  │  • TTL management        │ │
│  └──────────────────────────┘  │  │  └──────────────────────────┘ │
└───────────────────────────────┘  └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│  MODEL COMPONENT              │  │  UTILITY COMPONENT             │
│                               │  │                                │
│  ┌──────────────────────────┐ │  │  ┌──────────────────────────┐ │
│  │  BLIP Model              │ │  │  │  Rate Limiter           │ │
│  │  • Model Loader          │ │  │  │  • Request tracking     │ │
│  │  • Vision Encoder        │ │  │  │  • Limit enforcement     │ │
│  │  • Text Decoder          │ │  │  └──────────────────────────┘ │
│  │  • Processor             │ │  │                                │
│  └──────────────────────────┘ │  │  ┌──────────────────────────┐ │
│                               │  │  │  Auth Middleware         │ │
│  ┌──────────────────────────┐ │  │  │  • API key validation    │ │
│  │  Accent Model            │ │  │  │  • Token verification     │ │
│  │  • Model Loader          │ │  │  └──────────────────────────┘ │
│  │  • XLM-RoBERTa          │ │  │                                │
│  │  • Token Classifier     │ │  │  ┌──────────────────────────┐ │
│  │  • Token Merger         │ │  │  │  Logger                  │ │
│  │  • Label Mapper         │ │  │  │  • Error logging         │ │
│  └──────────────────────────┘ │  │  │  • Performance logging   │ │
└───────────────────────────────┘  │  └──────────────────────────┘ │
                                   └───────────────────────────────┘
```

---

## 5. DEPLOYMENT DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION ENVIRONMENT                        │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Load Balancer (Nginx)                                        │  │
│  │  • SSL/TLS termination                                        │  │
│  │  • Request routing                                            │  │
│  │  • Health checks                                              │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│        ┌─────────────────┼─────────────────┐                        │
│        │                 │                 │                        │
│        ▼                 ▼                 ▼                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
│  │ API      │    │ API      │    │ API      │                     │
│  │ Server 1 │    │ Server 2 │    │ Server N │                     │
│  │          │    │          │    │          │                     │
│  │ FastAPI  │    │ FastAPI  │    │ FastAPI  │                     │
│  │ Uvicorn  │    │ Uvicorn  │    │ Uvicorn  │                     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                     │
│       │               │               │                            │
│       └───────┬───────┴───────┬───────┘                            │
│               │               │                                     │
│               ▼               ▼                                     │
│  ┌───────────────────────────────────────┐                        │
│  │  Shared Storage                       │                        │
│  │  ┌──────────────┐  ┌──────────────┐  │                        │
│  │  │ Model Files  │  │  Redis Cache │  │                        │
│  │  │ (NFS/S3)     │  │  (Shared)    │  │                        │
│  │  └──────────────┘  └──────────────┘  │                        │
│  └───────────────────────────────────────┘                        │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Monitoring & Logging                                         │  │
│  │  • Prometheus (Metrics)                                       │  │
│  │  • Grafana (Visualization)                                    │  │
│  │  • ELK Stack (Logging)                                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. CACHE ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CACHE ARCHITECTURE                           │
│                                                                       │
│  Request Image                                                       │
│       │                                                               │
│       ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Image Hash Generator                                          │  │
│  │  • Load image                                                  │  │
│  │  • Convert to PNG (standardize)                               │  │
│  │  • Calculate MD5 hash                                         │  │
│  │  Output: "a1b2c3d4e5f6..."                                    │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Cache Lookup                                                  │  │
│  │  Key: MD5 hash                                                 │  │
│  │  Storage: In-Memory Dictionary                                 │  │
│  │                                                                │  │
│  │  {                                                             │  │
│  │    "hash1": {                                                  │  │
│  │      "caption": "áo khoác thể thao nữ màu đen",              │  │
│  │      "expires_at": 1234567890.0,                              │  │
│  │      "created_at": 1234560000.0                                │  │
│  │    },                                                          │  │
│  │    "hash2": {...},                                             │  │
│  │    ...                                                         │  │
│  │  }                                                             │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│          ┌───────────────┴───────────────┐                         │
│          │                               │                         │
│    Cache Hit?                      Cache Miss?                     │
│          │                               │                         │
│          │                               ▼                         │
│          │              ┌──────────────────────────┐              │
│          │              │  Generate Caption        │              │
│          │              │  (BLIP + Accent)         │              │
│          │              └──────────────┬───────────┘              │
│          │                             │                         │
│          │                             ▼                         │
│          │              ┌──────────────────────────┐              │
│          │              │  Cache Storage           │              │
│          │              │  • Store caption         │              │
│          │              │  • Set TTL (24h)        │              │
│          │              │  • Update expires_at    │              │
│          │              └──────────────┬───────────┘              │
│          │                             │                         │
│          └─────────────────────────────┘                         │
│                          │                                       │
│                          ▼                                       │
│              ┌──────────────────────────┐                        │
│              │  Return Caption         │                        │
│              │  (cached or new)        │                        │
│              └──────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## 7. MODEL ARCHITECTURE DIAGRAM

### 7.1. BLIP Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BLIP MODEL ARCHITECTURE                       │
│                                                                       │
│  Input Image (224×224 or 384×384)                                    │
│       │                                                               │
│       ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Vision Encoder (ViT-B/16)                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Patch Embedding                                        │  │  │
│  │  │  • Split into 16×16 patches                            │  │  │
│  │  │  • Linear projection                                    │  │  │
│  │  │  • Positional encoding                                  │  │  │
│  │  └───────────────────────┬─────────────────────────────────┘  │  │
│  │                          │                                     │  │
│  │                          ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Transformer Layers (12 layers)                         │  │  │
│  │  │  • Self-attention                                       │  │  │
│  │  │  • Feed-forward                                         │  │  │
│  │  │  • Layer normalization                                 │  │  │
│  │  └───────────────────────┬─────────────────────────────────┘  │  │
│  │                          │                                     │  │
│  │                          ▼                                     │  │
│  │  Image Features: F_img = {f₁, f₂, ..., fₙ}                    │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Cross-Modal Attention                                        │  │
│  │  • Image features (Key, Value)                               │  │
│  │  • Text tokens (Query)                                       │  │
│  │  • Attention(Q_text, K_img, V_img)                           │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Text Decoder (BERT-based)                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Autoregressive Generation                              │  │  │
│  │  │  For each token:                                        │  │  │
│  │  │    • Cross-attention with image features                │  │  │
│  │  │    • Self-attention with previous tokens               │  │  │
│  │  │    • Predict next token                                 │  │  │
│  │  └───────────────────────┬─────────────────────────────────┘  │  │
│  │                          │                                     │  │
│  │                          ▼                                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Beam Search (num_beams=3)                              │  │  │
│  │  │  • Generate multiple candidates                         │  │  │
│  │  │  • Select best sequence                                 │  │  │
│  │  └───────────────────────┬─────────────────────────────────┘  │  │
│  │                          │                                     │  │
│  │                          ▼                                     │  │
│  │  Token IDs: [101, 2345, 6789, ..., 102]                      │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Token Decoder                                                │  │
│  │  • Convert token IDs to text                                 │  │
│  │  • Remove special tokens                                     │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  Output: Caption (no accent)                                         │
│  Example: "ao khoac the thao nu mau den"                             │
└───────────────────────────────────────────────────────────────────────┘
```

### 7.2. Accent Restoration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ACCENT RESTORATION ARCHITECTURE                     │
│                                                                       │
│  Input: Text không dấu                                               │
│  Example: "ao khoac the thao nu mau den"                             │
│       │                                                               │
│       ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Tokenization (XLM-RoBERTa Tokenizer)                        │  │
│  │  • Split into words                                           │  │
│  │  • Tokenize with is_split_into_words=True                     │  │
│  │  • Add special tokens ([CLS], [SEP])                         │  │
│  │  Output: ["[CLS]", "▁ao", "▁khoac", "▁the", ..., "[SEP]"]    │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  XLM-RoBERTa Model                                            │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Forward Pass                                           │  │  │
│  │  │  • Embedding layer                                      │  │  │
│  │  │  • Transformer layers (12 layers)                      │  │  │
│  │  │  • Classification head                                 │  │  │
│  │  └───────────────────────┬─────────────────────────────────┘  │  │
│  │                          │                                     │  │
│  │                          ▼                                     │  │
│  │  Logits: [batch_size, seq_len, num_labels]                    │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Label Prediction                                            │  │
│  │  • Argmax to get predicted label for each token              │  │
│  │  • Remove special tokens ([CLS], [SEP])                     │  │
│  │  Output: [label_ao, label_khoac, label_the, ...]            │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Token Merger                                                 │  │
│  │  • Identify word boundaries (prefix "▁")                     │  │
│  │  • Merge subword tokens belonging to same word               │  │
│  │  Output: [("ao", {label_ao}), ("khoac", {label_khoac}), ...]│  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Label Mapper                                                 │  │
│  │  • Map labels to accent patterns                             │  │
│  │  • Format: "raw-vowel" (e.g., "ao-áo")                      │  │
│  │  • Replace raw text with accented text                       │  │
│  │  Output: ["áo", "khoác", "thể", "thao", "nữ", ...]          │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Join Words                                                   │  │
│  │  • Join accented words into sentence                         │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│  Output: Text có dấu                                                 │
│  Example: "áo khoác thể thao nữ màu đen"                            │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 8. HƯỚNG DẪN SỬ DỤNG

### 8.1. Cách sử dụng trong báo cáo

**1. Copy trực tiếp**:
- Các diagram ASCII art có thể copy trực tiếp vào báo cáo
- Format: Markdown code block với syntax highlighting

**2. Vẽ lại bằng công cụ**:
- Sử dụng các diagram này làm reference
- Vẽ lại bằng:
  - Draw.io / diagrams.net
  - Lucidchart
  - PlantUML
  - Mermaid
  - PowerPoint / Keynote

**3. Chuyển đổi format**:
- ASCII art → PNG/SVG bằng tools như:
  - `ascii-art-to-image`
  - Online converters

### 8.2. Gợi ý cho từng diagram

**System Architecture**: Sử dụng trong Chương 3 (Phân tích và thiết kế)
**Data Flow**: Sử dụng trong Chương 3 và 4
**Sequence Diagram**: Sử dụng trong Chương 3 (API Design)
**Component Diagram**: Sử dụng trong Chương 3 (Kiến trúc)
**Deployment Diagram**: Sử dụng trong Chương 4 (Triển khai)
**Cache Architecture**: Sử dụng trong Chương 3 (Cache Design)
**Model Architecture**: Sử dụng trong Chương 2 (Cơ sở lý thuyết)

---

**Lưu ý**: Các diagram này có thể được vẽ lại bằng các công cụ chuyên nghiệp để có hình ảnh đẹp hơn cho báo cáo.

