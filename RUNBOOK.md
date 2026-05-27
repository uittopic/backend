# RUNBOOK - Official Eval v2

## 1) Activate environment

```bash
cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
source .venv/bin/activate
```

## 2) Run full evaluation

```bash
.venv/bin/python tools/eval_new_model.py \
  --model-path models/blip_shopee_vicaps_v1_0_1_mps \
  --test-csv data/test_clean_v3_final_noleak_filtered.csv \
  --output outputs/official_eval_v2/predictions.csv \
  --summary outputs/official_eval_v2/metrics_summary.json
```

Expected runtime on CPU: around 4 hours.

## 3) Verify outputs

```bash
ls -lh outputs/official_eval_v2/predictions.csv outputs/official_eval_v2/metrics_summary.json
wc -l outputs/official_eval_v2/predictions.csv
```

Expected lines in `predictions.csv`: `3338` (`3337` samples + header).

## 4) Validate schema + summary consistency

```bash
.venv/bin/python - <<'PY'
import json, pandas as pd

s = json.load(open('outputs/official_eval_v2/metrics_summary.json', encoding='utf-8'))
df = pd.read_csv('outputs/official_eval_v2/predictions.csv')

print('rows=', len(df), 'nan=', int(df.isna().sum().sum()))
print('bleu_keys=', list(s['bleu_scores'].keys()))
print('csv_bleu4_sem=', round(df['bleu4_semantic_no_accent'].mean(), 4))
print('sum_bleu4_sem=', s['bleu_scores']['bleu4_semantic_no_accent'])
print('csv_bleu4_acc=', round(df['bleu4_accented_exact'].mean(), 4))
print('sum_bleu4_acc=', s['bleu_scores']['bleu4_accented_exact'])
PY
```

Expected:
- no NaN
- CSV mean and summary values match

## 5) Freeze final artifacts

```bash
sudo chown -R $(whoami):staff outputs/official_eval_v2
mkdir -p outputs/freeze_final_v2
cp outputs/official_eval_v2/predictions.csv outputs/freeze_final_v2/
cp outputs/official_eval_v2/metrics_summary.json outputs/freeze_final_v2/
shasum -a 256 outputs/freeze_final_v2/predictions.csv outputs/freeze_final_v2/metrics_summary.json
```

Reference SHA256 (current freeze):
- `predictions.csv`: `eb03856b8d9f971be04163b11ce574e4a47d90527693df0d576ffa521a52a158`
- `metrics_summary.json`: `759314699a08f8ea48bdaca93c8425f99fc443204068b045639d1c045f0d9f92`

## 6) Freeze policy

Do not retrain, do not change dataset/metric logic after freeze.
