"""Make figure of distribution of conserved features and sample alignments."""

import json
import os
import re
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import skbio
from src.draw import plot_msa
from src.utils import read_fasta

pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 2
min_indel_rate = 0.1

tree_template = skbio.read('../../IDR_evolution/data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

with open('../../IDR_evolution/analysis/brownian/simulate_stats/out/BM/critvals.json') as file:
    critvals = json.load(file)

# Load regions as segments
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        for ppid in fields['ppids'].split(','):
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'ppid': ppid})
all_segments = pd.DataFrame(rows)
all_regions = all_segments.drop('ppid', axis=1).drop_duplicates()

asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

models = pd.read_table(f'../../IDR_evolution/analysis//brownian/model_compute/out/models_{min_length}.tsv', header=[0, 1])
models = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
models = models.set_index(['OGid', 'start', 'stop', 'disorder'])

feature_groups = {}
feature_labels = []
nonmotif_labels = []
with open(f'../../IDR_evolution/analysis/brownian/model_compute/out/models_{min_length}.tsv') as file:
    column_labels = file.readline().rstrip('\n').split('\t')
    group_labels = file.readline().rstrip('\n').split('\t')
for column_label, group_label in zip(column_labels, group_labels):
    if not column_label.endswith('_loglikelihood_BM') or group_label == 'ids_group':
        continue
    feature_label = column_label.removesuffix('_loglikelihood_BM')
    try:
        feature_groups[group_label].append(feature_label)
    except KeyError:
        feature_groups[group_label] = [feature_label]
    feature_labels.append(feature_label)
    if group_label != 'motifs_group':
        nonmotif_labels.append(feature_label)

columns = {}
for feature_label in feature_labels:
    columns[f'{feature_label}_delta_loglikelihood'] = models[f'{feature_label}_loglikelihood_OU'] - models[
        f'{feature_label}_loglikelihood_BM']
    columns[f'{feature_label}_sigma2_ratio'] = models[f'{feature_label}_sigma2_BM'] / models[
        f'{feature_label}_sigma2_OU']
models = pd.concat([models, pd.DataFrame(columns)], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

# === DISTRIBUTION FIGURE ===
fig = plt.figure(figsize=(7.5, 5))
gs = plt.GridSpec(2, 1, height_ratios=[0.6, 0.4])

# --- PANEL A: Bar graph of fraction of regions with a significant feature ---
column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
ys_95 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).mean()
ys_99 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).mean()
xs = list(range(len(column_labels)))
labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

subfig = fig.add_subfigure(gs[0])
ax = subfig.subplots(gridspec_kw={'left': 0.1, 'right': 0.99, 'bottom': 0.425, 'top': 0.99})
ax.bar(xs, ys_99, label='1%', color='C0')
ax.bar(xs, ys_95 - ys_99, bottom=ys_99, label='5%', color='C1')
ax.set_xmargin(0.005)
ax.set_xticks(xs, labels, fontsize=5.5,
              rotation=60, rotation_mode='anchor', ha='right', va='center')
ax.set_ylabel('Fraction of regions')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=2,
          title='Type I error rate', fontsize=8, title_fontsize=8)
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# --- PANEL B: Distribution of number of significant features in regions ---
column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]

counts_95 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q95']).sum(axis=1).value_counts()
xs_95 = np.arange(counts_95.index.min(), counts_95.index.max() + 1)
ys_95 = stats.binom.pmf(xs_95, n=len(column_labels), p=0.05) * counts_95.values.sum()

counts_99 = (models.loc[pdidx[:, :, :, True], column_labels] > critvals['q99']).sum(axis=1).value_counts()
xs_99 = np.arange(counts_99.index.min(), counts_99.index.max() + 1)
ys_99 = stats.binom.pmf(xs_99, n=len(column_labels), p=0.01) * counts_99.values.sum()

subfig = fig.add_subfigure(gs[1, 0])
axs = subfig.subplots(1, 2, gridspec_kw={'left': 0.1, 'right': 0.99, 'bottom': 0.225, 'top': 0.95})

width = 0.35
plots = [(counts_99, xs_99, ys_99, 'observed\n(1% type I error)', 'C0'),
         (counts_95, xs_95, ys_95, 'observed\n(5% type I error)', 'C1')]
for ax, (counts, xs, ys, label, color) in zip(axs, plots):
    ax.bar(counts.index - width / 2, counts.values, width=width, label=label, color=color)
    ax.bar(xs + width / 2, ys, width=width, label='random', color='C9')
    ax.set_xmargin(0.01)
    ax.set_xlabel('Number of significant features')
    ax.set_ylabel('Number of regions')
    ax.legend(fontsize=8)
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/conserved.png', dpi=300)
fig.savefig('out/conserved.tiff', dpi=300)
plt.close()

# === ALIGNMENT FIGURE ===
plot_msa_kwargs_common = {'tree_kwargs': {'linewidth': 0.75, 'tip_labels': False, 'xmin_pad': 0.025, 'xmax_pad': 0.025},
                          'msa_legend': True}
plot_msa_kwargs = {'left': 0.12, 'right': 0.85, 'top': 0.925, 'bottom': 0.075, 'anchor': (0, 0.5),
                   'hspace': 0.01,
                   'tree_position': 0.02, 'tree_width': 0.1,
                   'legend_kwargs': {'bbox_to_anchor': (0.875, 0.5), 'loc': 'center left', 'fontsize': 5,
                                     'handletextpad': 0.5, 'markerscale': 0.75, 'handlelength': 1},
                   **plot_msa_kwargs_common}

records = [('1944', 474, 623, True),
           ('0E60', 370, 508, True)]

fig_width, fig_height = 7.5, 7.5
fig = plt.figure(figsize=(fig_width, fig_height))
gs = plt.GridSpec(2*len(records), 1)

for idx, (OGid, start, stop, _) in enumerate(records):
    # --- PANEL 1: Sample region ---
    # Get segments in region
    conditions = ((all_segments['OGid'] == OGid) &
                  (all_segments['start'] == start) &
                  (all_segments['stop'] == stop))
    ppids = set(all_segments.loc[conditions, 'ppid'])

    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../IDR_evolution/data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        if ppid in ppids:
            msa.append({'ppid': ppid, 'spid': spid, 'seq': seq[start:stop]})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    # Load tree
    tree = tree_template.shear([record['spid'] for record in msa])
    for node in tree.postorder():  # Ensure tree is ordered as in original
        if node.is_tip():
            node.value = tip_order[node.name]
        else:
            node.children = sorted(node.children, key=lambda x: x.value)
            node.value = sum([child.value for child in node.children])

    subfig = fig.add_subfigure(gs[2*idx])
    plot_msa([record['seq'] for record in msa],
             x_start=start,
             fig=subfig, figsize=(fig_width, fig_height / gs.nrows),
             tree=tree,
             **plot_msa_kwargs)
    subfig.suptitle(ascii_uppercase[2*idx], x=0.0125, y=0.975, fontweight='bold')

    # --- Panel 2: delta loglikelihood ---
    column_labels = [f'{feature_label}_delta_loglikelihood' for feature_label in feature_labels]
    ys = models.loc[pdidx[OGid, start, stop, True], column_labels]
    xs = list(range(len(column_labels)))
    labels = [label.removesuffix('_delta_loglikelihood') for label in column_labels]

    subfig = fig.add_subfigure(gs[2*idx+1])
    ax = subfig.subplots(gridspec_kw={'left': 0.075, 'right': 0.99, 'bottom': 0.45, 'top': 0.95})
    ax.bar(xs, ys, facecolor='none', edgecolor='black', linewidth=0.75)
    ax.axhline(critvals['q99'], label='1%', color='C0', linewidth=0.75, linestyle='--')
    ax.axhline(critvals['q95'], label='5%', color='C1', linewidth=0.75, linestyle='--')
    ax.set_xmargin(0.005)
    ax.set_xticks(xs, labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_ylabel('$\mathregular{\log L_{OU} / L_{BM}}$')
    ax.legend(title='Type I error', title_fontsize=8, fontsize=8)
    subfig.suptitle(ascii_uppercase[2*idx+1], x=0.0125, y=0.975, fontweight='bold')

fig.savefig('out/alignment.png', dpi=300)
fig.savefig('out/alignment.tiff', dpi=300)
plt.close()
