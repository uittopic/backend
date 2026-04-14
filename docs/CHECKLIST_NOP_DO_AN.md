# CHECKLIST_NOP_DO_AN

Checklist chot nop do an cho project:
`/Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend`

## 1) Setup moi terminal

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
```

Kiem tra nhanh:

```bash
pwd
ls tools
```

## 2) Chay API de demo

```bash
export DEVICE=mps
export MODEL_PATH=/Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend/models/blip_vietnamese_80_20
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Postman:
- `POST http://127.0.0.1:8000/api/caption_full`
- Body -> `form-data` -> key `file` (type `File`)

## 3) Tao predictions tren tap test 20%

Neu chua co `outputs/predictions_test.csv`, chay:

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
python tools/run_inference_full_test.py
```

File output:
- `outputs/predictions_test.csv`

## 4) Danh gia metrics

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
python tools/evaluate_test_20.py --predictions outputs/predictions_test.csv --ground-truth data/test_20.csv --create-comparison
```

Files output:
- `outputs/evaluation_metrics.csv`
- `outputs/comparison_test_20.csv`
- `outputs/ground_truth_simple.csv`

## 5) Ghi ket qua EXP vao bang tong hop

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
python tools/record_experiment_result.py --exp-id EXP-01-BASELINE --metrics-csv outputs/evaluation_metrics.csv --objective "Baseline decode default" --model-path models/blip_vietnamese_80_20 --num-beams 3 --no-repeat-ngram-size 3 --repetition-penalty 1.2 --use-accent true --notes "Official baseline for report"
```

Files output:
- `outputs/experiment_results.csv`
- `configs/experiment_matrix.csv`

## 6) Xuat bao cao xep hang tu dong

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
python tools/export_experiment_report.py --results-csv outputs/experiment_results.csv
```

Files output:
- `outputs/experiment_summary_latest.csv`
- `outputs/experiment_summary_latest.md`

## 7) Phan tich loi de viet bao cao

Dung template:
- `docs/ERROR_ANALYSIS_TEMPLATE.md`
- `outputs/error_analysis_sheet_template.csv`

Muc tieu:
- Chon 30-50 mau loi tieu bieu.
- Gan nhan loi: sai loai san pham, sai mau, sai thuong hieu, lap tu, lai Anh-Viet.
- Viet hanh dong khac phuc cho tung nhom loi.

## 8) Bo file can nop cho thay

Toi thieu can co:
- `README.md`
- `docs/CHECKLIST_NOP_DO_AN.md`
- `docs/ERROR_ANALYSIS_TEMPLATE.md`
- `outputs/evaluation_metrics.csv`
- `outputs/comparison_test_20.csv`
- `outputs/experiment_results.csv`
- `outputs/experiment_summary_latest.csv`
- `outputs/experiment_summary_latest.md`
- Anh demo va screenshot Postman (2-4 case)

## 9) Lenh dong goi nhanh truoc khi nop

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De
zip -r CHUYEN_DE_Backend_submission.zip CHUYEN_DE_Backend \
  -x "*/.venv/*" "*/__pycache__/*" "*/.DS_Store"
```

## 10) 4 cau tra loi khi bao ve

1. Bai toan: sinh caption san pham tieng Viet co dau tu anh.
2. Pipeline: BLIP sinh khong dau -> accent restoration -> API FastAPI.
3. Danh gia: BLEU/ROUGE-L/SBERT + bang so sanh prediction-ground truth.
4. Han che va huong phat trien: mo rong du lieu, them thu nghiem decode, phan tich loi chi tiet.
