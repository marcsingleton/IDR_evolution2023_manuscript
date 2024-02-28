"""Make figure of results from fitting substitution models."""

import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import skbio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from src.utils import read_iqtree, read_paml

dpi = 300

paml_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
labels = ['0R_disorder', '50R_disorder', '100R_disorder', '0R_order', '50R_order', '100R_order']
labels_suffix = r'_[0-9]+'
Record = namedtuple('Record', ['label', 'ematrix', 'rmatrix', 'freqs', 'length'])

# Load LG model
ematrix, freqs = read_paml('../../IDR_evolution/data/matrices/LG.paml', norm=True)
rmatrix = freqs * ematrix

# Load IQ-TREE matrices
records = {}
for label in labels:
    file_labels = []
    for path in os.listdir('../../IDR_evolution/analysis/evofit/iqtree_fit/out/'):
        match = re.match(f'({label}{labels_suffix})\.iqtree', path)
        if match:
            file_labels.append(match.group(1))

    ematrix_stack = []
    rmatrix_stack = []
    freqs_stack = []
    length_stack = []
    for file_label in sorted(file_labels):
        record = read_iqtree(f'../../IDR_evolution/analysis/evofit/iqtree_fit/out/{file_label}.iqtree', norm=True)
        ematrix, freqs = record['ematrix'], record['freqs']
        rmatrix = freqs * ematrix

        tree = skbio.read(f'../../IDR_evolution/analysis/evofit/iqtree_fit/out/{file_label}.treefile', 'newick', skbio.TreeNode)
        length = tree.descending_branch_length()

        ematrix_stack.append(ematrix)
        rmatrix_stack.append(rmatrix)
        freqs_stack.append(freqs)
        length_stack.append(length)
    records[label] = Record(label,
                            np.stack(ematrix_stack),
                            np.stack(rmatrix_stack),
                            np.stack(freqs_stack),
                            np.stack(length_stack))

# Make plots
if not os.path.exists('out/'):
    os.mkdir('out/')

# RE-ORDER SYMBOLS BY DISORDER RATIO
freqs1 = records['50R_disorder'].freqs.mean(axis=0)
freqs2 = records['50R_order'].freqs.mean(axis=0)
sym2ratio = {sym: ratio for sym, ratio in zip(paml_order, freqs1 / freqs2)}
sym2idx = {sym: idx for idx, sym in enumerate(paml_order)}
alphabet = sorted(paml_order, key=lambda x: sym2ratio[x])
ix = [sym2idx[sym] for sym in alphabet]
ixgrid = np.ix_(ix, ix)
for label, record in records.items():
    records[label] = Record(label,
                            record.ematrix[:, ixgrid[0], ixgrid[1]],
                            record.rmatrix[:, ixgrid[0], ixgrid[1]],
                            record.freqs[:, ix],
                            record.length)

# === MAIN FIGURE ===
plots = [(records['50R_disorder'], records['50R_order'], 'ematrix', 'exchangeability'),
         (records['50R_disorder'], records['50R_order'], 'rmatrix', 'rate')]

fig = plt.figure(figsize=(7.5, 6.25))
gs = plt.GridSpec(1 + len(plots), 3)
rect = (0.1, 0.1, 0.7, 0.8)

width = 0.2
bars = [records['50R_disorder'], records['50R_order']]
subfig = fig.add_subfigure(gs[0, :])
ax = subfig.add_axes((0.1, 0.125, 0.85, 0.8))
for i, record in enumerate(bars):
    freqs = record.freqs.mean(axis=0)
    std = record.freqs.std(axis=0)
    label = record.label.split('_')[1]
    dx = -(len(bars) - 1) / 2 + i
    ax.bar([x+width*dx for x in range(len(alphabet))], freqs, yerr=std, label=label, width=width)
ax.set_xticks(range(len(alphabet)), alphabet)
ax.set_xlabel('Amino acid')
ax.set_ylabel('Frequency')
ax.legend()
subfig.suptitle('A', x=0.0125, y=0.975, fontweight='bold')  # Half usual value because panel is full-width

panel_labels = ['B', 'C', 'D', 'E', 'F', 'G']
for gs_idx, plot in enumerate(plots):
    record1, record2, data_label, title_label = plot
    matrix1 = getattr(record1, data_label).mean(axis=0)
    matrix2 = getattr(record2, data_label).mean(axis=0)
    vmax = max(matrix1.max(), matrix2.max())
    ratio = np.log10(matrix1 / matrix2)
    vext = np.nanmax(np.abs(ratio))

    subfig = fig.add_subfigure(gs[gs_idx+1, 0])
    ax = subfig.add_axes(rect)
    im = ax.imshow(matrix1, vmin=0, vmax=vmax, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=6)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=6, ha='center')
    ax.set_title(f'disorder {title_label}', fontsize=8, pad=3)
    subfig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
    subfig.suptitle(panel_labels[3*gs_idx], x=0.0375, y=0.975, fontweight='bold')  # 1.5x because panel is third length

    subfig = fig.add_subfigure(gs[gs_idx+1, 1])
    ax = subfig.add_axes(rect)
    im = ax.imshow(matrix2, vmin=0, vmax=vmax, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=6)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=6, ha='center')
    ax.set_title(f'order {title_label}', fontsize=8, pad=3)
    subfig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
    subfig.suptitle(panel_labels[3*gs_idx+1], x=0.0375, y=0.975, fontweight='bold')  # 1.5x because panel is third length

    subfig = fig.add_subfigure(gs[1+gs_idx, 2])
    ax = subfig.add_axes(rect)
    im = ax.imshow(ratio, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=6)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=6, ha='center')
    ax.set_title(f'{title_label} log ratio', fontsize=8, pad=3)
    subfig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
    subfig.suptitle(panel_labels[3*gs_idx+2], x=0.0375, y=0.975, fontweight='bold')  # 1.5x because panel is third length
fig.savefig('out/substitution.png', dpi=dpi)
fig.savefig('out/substitution.tiff', dpi=dpi)
plt.close()

# === HEATMAP ===
plots = [('ematrix', 'exchangeability'),
         ('rmatrix', 'rate')]
for data_label, title_label in plots:
    vmax = max([getattr(record, data_label).mean(axis=0).max() for record in records.values()])
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.5), layout='constrained')
    for ax, record in zip(axs.ravel(), records.values()):
        data = getattr(record, data_label)
        ax.imshow(data.mean(axis=0), vmax=vmax, cmap='Greys')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(record.label)
    fig.colorbar(ScalarMappable(Normalize(0, vmax), cmap='Greys'), ax=axs, fraction=0.025)
    fig.savefig(f'out/heatmap_{data_label}.png', dpi=dpi)
    fig.savefig(f'out/heatmap_{data_label}.tiff', dpi=dpi)
    plt.close()

# === CORRELATION GRID ===
fig = plt.figure(figsize=(7.5, 3))
gs = plt.GridSpec(1, 2)
rect = (0.25, 0.1, 0.55, 0.85)

plots = [('ematrix', 'exchangeability', 'A'),
         ('rmatrix', 'rate', 'B')]
for gs_idx, plot in enumerate(plots):
    data_label, title_label, panel_label = plot
    corr = np.zeros((len(labels), len(labels)))
    for i, label1 in enumerate(labels):
        matrix1 = getattr(records[label1], data_label).mean(axis=0)
        for j, label2 in enumerate(labels[:i+1]):
            matrix2 = getattr(records[label2], data_label).mean(axis=0)
            r = np.corrcoef(matrix1.ravel(), matrix2.ravel())
            corr[i, j] = r[0, 1]
            corr[j, i] = r[0, 1]

    subfig = fig.add_subfigure(gs[0, gs_idx])
    ax = subfig.add_axes(rect)
    im = ax.imshow(corr)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=30, rotation_mode='anchor',
                  horizontalalignment='right', verticalalignment='center')
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    subfig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
    subfig.suptitle(panel_label, x=0.025, y=0.975, fontweight='bold')
fig.savefig(f'out/heatmap_corr.png', dpi=dpi)
fig.savefig(f'out/heatmap_corr.tiff', dpi=dpi)
plt.close()

# === VARIATION ===
plots = [(records['50R_disorder'], records['50R_order'], 'ematrix'),
         (records['50R_disorder'], records['50R_order'], 'rmatrix')]
for record1, record2, data_label in plots:
    fig = plt.figure(figsize=(7.5, 3.5))
    gs = plt.GridSpec(2, 3)
    rect = (0.1, 0.1, 0.7, 0.8)

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for gs_idx, record in enumerate([record1, record2]):
        data = getattr(record, data_label)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)

        subfig = fig.add_subfigure(gs[gs_idx, 0])
        ax = subfig.add_axes(rect)
        im = ax.imshow(mean, cmap='Greys')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=6)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=6, ha='center')
        fig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
        subfig.suptitle(panel_labels[3*gs_idx], x=0.0375, y=0.975, fontweight='bold')

        subfig = fig.add_subfigure(gs[gs_idx, 1])
        ax = subfig.add_axes(rect)
        im = ax.imshow(std / mean, cmap='Greys')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=6)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=6, ha='center')
        fig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))
        subfig.suptitle(panel_labels[3*gs_idx+1], x=0.0375, y=0.975, fontweight='bold')

        subfig = fig.add_subfigure(gs[gs_idx, 2])
        ax = subfig.add_axes((0.275, 0.25, 0.675, 0.65))
        ax.scatter(mean, std / mean, s=10, alpha=0.5, edgecolor='none')
        ax.set_xlabel('Mean')
        ax.set_ylabel('Coefficient of variation')
        subfig.suptitle(panel_labels[3*gs_idx+2], x=0.0375, y=0.975, fontweight='bold')

    fig.savefig(f'out/CV_{data_label}.png', dpi=dpi)
    fig.savefig(f'out/CV_{data_label}.tiff', dpi=dpi)
    plt.close()
