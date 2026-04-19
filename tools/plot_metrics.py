import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data for BLEU distribution
labels = ['BLEU-4=0', '0-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.10', '0.10-1.00']
counts = [887, 62, 148, 370, 136, 136]
percentages = [58.6, 4.1, 9.8, 24.4, 9.0, 9.0]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, counts, color='#2E86AB', edgecolor='black')

# Add value labels
for bar, cnt, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,
            f'{cnt}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('BLEU-4 Score Range', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('BLEU-4 Distribution on 1,514 Test Samples\n(blip_vietnamese_cleaned_v1 - Baseline Config)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)
plt.tight_layout()
out1 = Path('outputs/bleu_distribution.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f'✅ Saved: {out1}')
plt.close()

# Comparison: Baseline vs Tuned
metrics = ['BLEU-1', 'BLEU-4', 'Time\n(s/img)']
baseline_vals = [0.1326, 0.0457, 1.4]
tuned_vals   = [0.1320, 0.0459, 1.7]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline\n(beam=3, rep=1.05)', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x + width/2, tuned_vals, width, label='Tuned\n(beam=5, rep=1.2)', color='#A23B72', edgecolor='black')

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.4f}' if height < 2 else f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

ax.set_ylabel('Score / Time (s)', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs Tuned Config Comparison\n(1,514 test samples)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
out2 = Path('outputs/baseline_vs_tuned.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'✅ Saved: {out2}')
plt.close()

print('✅ All charts generated')
