"""Make figure of overlap with Pfam domains."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_table('../../IDR_evolution/analysis/IDRpred/pfam_overlap/out/overlaps.tsv')
df['fraction'] = df['overlap'] / df['length']
disorder = df[df['disorder']]
order = df[~df['disorder']]

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 3.5))
gs = plt.GridSpec(1, 2)

# === MAIN FIGURE ===
# --- PANEL A: Bar charts ---
hs_disorder = [(disorder['fraction'] == 0).sum(), (disorder['fraction'] != 0).sum()]
hs_order = [(order['fraction'] == 0).sum(), (order['fraction'] != 0).sum()]
hs_stack = list(zip(hs_disorder, hs_order))
hs_labels = ['No overlap', 'Overlap']
hs_colors = ['white', 'darkgray']
hs_hatches = [None, None]

xs = list(range(len(hs_stack)))
xs_labels = ['disorder', 'order']
xs_lim = [-0.75, 1.75]

subfig = fig.add_subfigure(gs[0, 0])
ax = subfig.subplots(gridspec_kw={'left': 0.2})
bs = [0 for _ in range(len(hs_stack))]
for hs, label, color, hatch in zip(hs_stack, hs_labels, hs_colors, hs_hatches):
    ax.bar(xs, hs, bottom=bs, width=0.5, label=label, color=color, hatch=hatch, linewidth=1.25, edgecolor='black')
    bs = [h + b for h, b in zip(hs, bs)]
ax.set_xlim(xs_lim)
ax.set_xticks(xs, xs_labels)
ax.set_ylabel('Number of regions')
ax.legend()
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# --- PANEL B: Histograms of non-zero overlap ---
subfig = fig.add_subfigure(gs[0, 1])
axs = subfig.subplots(2, gridspec_kw={'bottom': 0.15})
axs[0].hist(disorder.loc[disorder['fraction'] != 0, 'fraction'], bins=50, label='disorder', color='C0')
axs[1].hist(order.loc[order['fraction'] != 0, 'fraction'], bins=50, label='order', color='C1')
axs[1].set_xlabel('Fraction overlap with Pfam domain')
for ax in axs:
    ax.set_ylabel('Number of regions')
    ax.legend()
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/pfam.png', dpi=300)
fig.savefig('out/pfam.tiff', dpi=300)
plt.close()

# === CHI-SQUARED TEST ===
hs_disorder = [(disorder['fraction'] == 0).sum(), (disorder['fraction'] != 0).sum()]
hs_order = [(order['fraction'] == 0).sum(), (order['fraction'] != 0).sum()]
hs_stack = list(zip(hs_disorder, hs_order))

chi2, p, dof, expected = chi2_contingency(hs_stack)
output = f"""\
chi2: {chi2}
p: {p}
dof: {dof}
"""
with open('out/chi2.txt', 'w') as file:
    file.write(output)
