import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# FIG 1 — PROBLEM DEFINITION
# ============================================
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.5)
ax.axis('off')

# Input box (trái)
x_in, y_in, w, h = 1.0, 1.0, 3.0, 1.5
box_in = FancyBboxPatch((x_in, y_in), w, h, boxstyle="round,pad=0.15",
                        facecolor='#2E86AB', edgecolor='black', linewidth=2.5)
ax.add_patch(box_in)
ax.text(x_in + w/2, y_in + h/2, "INPUT\nIMAGE SẢN PHẨM", ha='center', va='center',
        fontsize=18, fontweight='bold', color='white')

# Arrow Input → Output
x_arrow_end = x_in + w + 0.5
ax.annotate('', xy=(x_arrow_end, y_in + h/2), xytext=(x_in + w, y_in + h/2),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# Output box (phải)
x_out = x_arrow_end + 1.0
box_out = FancyBboxPatch((x_out, y_in), 4.0, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#C73E1D', edgecolor='black', linewidth=2.5)
ax.add_patch(box_out)
ax.text(x_out + 2.0, y_in + h/2, "OUTPUT\nCAPTION TIẾNG VIỆT\nCÓ DẤU", ha='center', va='center',
        fontsize=18, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('outputs/fig1_problem.png', dpi=300, bbox_inches='tight')
print('✅ fig1_problem')
plt.close()

# ============================================
# FIG 2 — PIPELINE SIMPLE (IMPORTANT)
# ============================================
fig, ax = plt.subplots(figsize=(16, 3))
ax.set_xlim(0, 16)
ax.set_ylim(0, 3.5)
ax.axis('off')

steps = [
    (0.8, 1.0, 2.5, 1.5, "IMAGE", '#2E86AB'),
    (4.0, 1.0, 2.5, 1.5, "BLIP\nCAPTIONING", '#95A5A6'),
    (7.2, 1.0, 2.5, 1.5, "CAPTION\nKHÔNG DẤU", '#95A5A6'),
    (10.4, 1.0, 2.5, 1.5, "ACCENT\nRESTORATION\n(XLM-R)", '#95A5A6'),
    (13.6, 1.0, 2.5, 1.5, "FINAL\nCAPTION\nCÓ DẤU", '#C73E1D')
]

for x, y, w, h, label, color in steps:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='black', linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')

# Arrows
for i in range(len(steps)-1):
    x1 = steps[i][0] + steps[i][2]
    y1 = steps[i][1] + steps[i][3]/2
    x2 = steps[i+1][0]
    ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=4))

plt.tight_layout()
plt.savefig('outputs/fig2_pipeline_simple.png', dpi=300, bbox_inches='tight')
print('✅ fig2_pipeline_simple')
plt.close()

# ============================================
# FIG 3 — PIPELINE DETAIL (2 STAGES)
# ============================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis('off')

# ---------- STAGE 1: BLIP ----------
# Input Image (top-left)
x1, y1, w1, h1 = 0.8, 3.8, 2.4, 1.4
box_in = FancyBboxPatch((x1, y1), w1, h1, boxstyle="round,pad=0.15",
                        facecolor='#2E86AB', edgecolor='black', linewidth=2.5)
ax.add_patch(box_in)
ax.text(x1 + w1/2, y1 + h1/2, "INPUT\nIMAGE", ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Arrow Input → BLIP Model
ax.annotate('', xy=(3.8, 4.5), xytext=(3.2, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# BLIP Model (top-right)
x2, y2, w2, h2 = 4.0, 3.8, 2.8, 1.4
box_blip = FancyBboxPatch((x2, y2), w2, h2, boxstyle="round,pad=0.15",
                          facecolor='#2E86AB', edgecolor='black', linewidth=2.5)
ax.add_patch(box_blip)
ax.text(x2 + w2/2, y2 + h2/2, "BLIP MODEL\nVision Encoder + Text Decoder", ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Arrow BLIP → Caption (down)
ax.annotate('', xy=(x2 + w2/2, 3.2), xytext=(x2 + w2/2, y2),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# Caption no accent (bottom-right, same level as accent input)
x3, y3, w3, h3 = 4.0, 1.8, 2.8, 1.4
box_cap = FancyBboxPatch((x3, y3), w3, h3, boxstyle="round,pad=0.15",
                         facecolor='#95A5A6', edgecolor='black', linewidth=2.5)
ax.add_patch(box_cap)
ax.text(x3 + w3/2, y3 + h3/2, "CAPTION\nKHÔNG DẤU", ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Arrow Caption → Accent Input (left)
ax.annotate('', xy=(x1 + w1/2, 2.5), xytext=(x1 + w1/2, y3 + h3),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# Accent Input (bottom-left, same level as caption)
x4, y4 = 0.8, 1.8
box_acc_in = FancyBboxPatch((x4, y4), w1, h1, boxstyle="round,pad=0.15",
                            facecolor='#A23B72', edgecolor='black', linewidth=2.5)
ax.add_patch(box_acc_in)
ax.text(x4 + w1/2, y4 + h1/2, "INPUT CAPTION\n(KHÔNG DẤU)", ha='center', va='center',
        fontsize=15, fontweight='bold', color='white')

# Arrow Accent Input → Accent Model (down)
ax.annotate('', xy=(x4 + w1/2, 1.0), xytext=(x4 + w1/2, y4),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# ---------- STAGE 2: ACCENT RESTORATION ----------
# Accent Model (bottom-center)
x5, y5, w5, h5 = 4.0, 0.2, 2.8, 1.4
box_acc = FancyBboxPatch((x5, y5), w5, h5, boxstyle="round,pad=0.15",
                         facecolor='#A23B72', edgecolor='black', linewidth=2.5)
ax.add_patch(box_acc)
ax.text(x5 + w5/2, y5 + h5/2, "ACCENT RESTORATION\n(XLM-RoBERTa)", ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Arrow Accent Model → Output (left)
ax.annotate('', xy=(x4 + w1/2, y5 + h5/2), xytext=(x5 + w5/2, y5 + h5/2),
            arrowprops=dict(arrowstyle='->', color='black', lw=4))

# Output with accent (bottom-left, final)
x6, y6 = 0.8, 0.2
box_out = FancyBboxPatch((x6, y6), w1, h1, boxstyle="round,pad=0.15",
                         facecolor='#C73E1D', edgecolor='black', linewidth=2.5)
ax.add_patch(box_out)
ax.text(x6 + w1/2, y6 + h1/2, "OUTPUT\nCAPTION\nCÓ DẤU", ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('outputs/fig3_pipeline_detail.png', dpi=300, bbox_inches='tight')
print('✅ fig3_pipeline_detail')
plt.close()

# ============================================
# FIG 4 — BLIP ARCHITECTURE (CRITICAL)
# ============================================
fig, ax = plt.subplots(figsize=(13, 3))
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.5)
ax.axis('off')

boxes = [
    (0.6, 1.0, 2.0, 1.5, "INPUT\nIMAGE", '#2E86AB'),
    (3.2, 1.0, 2.2, 1.5, "VISION\nENCODER\n(ViT)", '#2E86AB'),
    (6.0, 1.0, 2.0, 1.5, "IMAGE\nFEATURES", '#95A5A6'),
    (8.6, 1.0, 2.2, 1.5, "TEXT\nDECODER\n(Transformer)", '#F18F01'),
    (11.4, 1.0, 1.8, 1.5, "GENERATED\nCAPTION", '#95A5A6')
]

for x, y, w, h, label, color in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='black', linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=17, fontweight='bold', color='white')

# Arrows
arrows = [(0.6+2.0, 1.75, 3.2, 1.75),  # Input → Encoder
          (3.2+2.2, 1.75, 6.0, 1.75),  # Encoder → Features
          (6.0+2.0, 1.75, 8.6, 1.75),  # Features → Decoder
          (8.6+2.2, 1.75, 11.4, 1.75)] # Decoder → Caption
for x_start, y_start, x_end, y_end in arrows:
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color='black', lw=4))

plt.tight_layout()
plt.savefig('outputs/fig4_blip_architecture.png', dpi=300, bbox_inches='tight')
print('✅ fig4_blip_architecture')
plt.close()

# ============================================
# FIG 5 — ACCENT RESTORATION
# ============================================
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3)
ax.axis('off')

steps5 = [
    (0.8, 0.7, 2.6, 1.6, "INPUT\nCAPTION\nKHÔNG DẤU", '#95A5A6'),
    (4.2, 0.7, 2.4, 1.6, "XLM-RoBERTa\nMODEL\n(Token Classification)", '#A23B72'),
    (7.8, 0.7, 2.6, 1.6, "OUTPUT\nCAPTION\nCÓ DẤU", '#C73E1D')
]

for i, (x, y, w, h, label, color) in enumerate(steps5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='black', linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=17, fontweight='bold', color='white')

    if i < len(steps5)-1:
        x_next = steps5[i+1][0]
        y_center = y + h/2
        ax.annotate('', xy=(x_next, y_center), xytext=(x + w, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=4))

plt.tight_layout()
plt.savefig('outputs/fig5_accent_restoration.png', dpi=300, bbox_inches='tight')
print('✅ fig5_accent_restoration')
plt.close()

print('🎯 ALL 5 FIGURES — CHUẨN ACADEMIC, DONE')
