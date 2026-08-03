"""
Generates publication-quality figures (Figures 3, 4, 5) for the manuscript:
- Figure 3: Bar chart of wound closure (AI vs ImageJ) by time point
- Figure 4: Correlation scatter plot
- Figure 5: Bland-Altman agreement plot

Requirements:
    pip install pandas matplotlib seaborn numpy scipy

Usage:
    python3 generate_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats

# Style
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13

# Load data (CSV expected in ../paired_data/)
here = Path(__file__).parent
csv = here.parent / 'paired_data' / 'paired_imagej_vs_ai_clean.csv'
df = pd.read_csv(csv)

# Figure 3: Bar chart by time point
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
tps = [8, 12, 24]
x = np.arange(len(tps))
width = 0.35

ij_means = [df[df['timepoint_h'] == t]['imagej'].mean() * 100 for t in tps]
ij_sds = [df[df['timepoint_h'] == t]['imagej'].std(ddof=1) * 100 for t in tps]
ai_means = [df[df['timepoint_h'] == t]['ai'].mean() * 100 for t in tps]
ai_sds = [df[df['timepoint_h'] == t]['ai'].std(ddof=1) * 100 for t in tps]

ax.bar(x - width/2, ij_means, width, yerr=ij_sds, label='ImageJ',
       color='#888888', capsize=5, edgecolor='black')
ax.bar(x + width/2, ai_means, width, yerr=ai_sds, label='AI',
       color='#CCCCCC', capsize=5, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels([f'{t} h' for t in tps])
ax.set_ylabel('Wound closure (%)')
ax.set_xlabel('Time post-scratch')
ax.set_ylim(0, 105)
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('Figure_3_method_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure_3_method_comparison.png")

# Figure 4: Correlation scatter
fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
colors = {8: '#1f77b4', 12: '#ff7f0e', 24: '#2ca02c'}
for tp in tps:
    sub = df[df['timepoint_h'] == tp]
    ax.scatter(sub['imagej'], sub['ai'], c=colors[tp], label=f'{tp} h',
               alpha=0.6, s=30, edgecolor='white', linewidth=0.5)

slope, intercept, r, p, _ = stats.linregress(df['imagej'], df['ai'])
xx = np.linspace(df['imagej'].min(), df['imagej'].max(), 100)
ax.plot(xx, slope*xx + intercept, color='red', linewidth=1.5,
        label=f'Regression (r={r:.3f})')
ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Line of identity')

ax.set_xlabel('ImageJ wound closure fraction')
ax.set_ylabel('AI model wound closure fraction')
ax.legend(loc='upper left', frameon=False, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('Figure_4_correlation_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Figure_4_correlation_scatter.png")

# Figure 5: Bland-Altman
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
mean_v = df['mean_imagej_ai']
diff_v = df['diff_ai_imagej']
bias = diff_v.mean()
sd_diff = diff_v.std(ddof=1)
loa_lo = bias - 1.96 * sd_diff
loa_hi = bias + 1.96 * sd_diff

for tp in tps:
    sub = df[df['timepoint_h'] == tp]
    ax.scatter(sub['mean_imagej_ai'], sub['diff_ai_imagej'], c=colors[tp],
               label=f'{tp} h', alpha=0.6, s=30, edgecolor='white', linewidth=0.5)

ax.axhline(bias, color='red', linewidth=1.5, label=f'Bias = {bias:+.4f}')
ax.axhline(loa_lo, color='gray', linewidth=1, linestyle='--')
ax.axhline(loa_hi, color='gray', linewidth=1, linestyle='--',
           label=f'95% LoA: [{loa_lo:+.3f}, {loa_hi:+.3f}]')

ax.set_xlabel('Mean of AI and ImageJ wound closure fraction')
ax.set_ylabel('Difference (AI − ImageJ)')
ax.legend(loc='upper right', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('Figure_5_bland_altman.png', dpi=300, bbox_inches='tight')
print("✓ Figure_5_bland_altman.png")

print(f"\nDone. All figures saved as 300 DPI PNG.")
