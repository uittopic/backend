import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== BẢNG: PIPELINE SUMMARY ====================
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

# Data: 3 columns — Thành phần | Mô tả | Vai trò
data = [
    ["Thành phần", "Mô tả", "Vai trò"],
    ["BLIP\n(fine-tuned)", "Vision-language model\ntrained trên UIT-ViIC", "Sinh caption\ntiếng Việt (không dấu)"],
    ["Accent Restoration\n(XLM-R)", "Language model\nfine-tuned để khôi phục dấu", "Thêm dấu\ncho caption"],
    ["Config\n(beam=3)", "Beam search với k=3", "Điều chỉnh\nchất lượng output"]
]

# Column widths
col_widths = [0.25, 0.45, 0.30]

# Table
table = ax.table(
    cellText=data,
    loc='center',
    cellLoc='center',
    colWidths=col_widths
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)

# Header row (row 0)
for j in range(3):
    cell = table[(0, j)]
    cell.set_facecolor('#2E86AB')
    cell.set_text_props(weight='bold', color='white')
    cell.set_height(0.25)  # header cao hơn

# Data rows (rows 1-3)
colors = ['#A23B72', '#F18F01', '#95A5A6']
for i in range(1, 4):
    for j in range(3):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i-1])
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.22)

# Borders
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(1.5)

ax.set_title("Bảng 1. Tóm tắt pipeline và vai trò của từng thành phần",
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/pipeline_summary_table.png', dpi=150, bbox_inches='tight')
print('✅ Saved: outputs/pipeline_summary_table.png')
plt.close()

print('✅ Pipeline summary table generated')
