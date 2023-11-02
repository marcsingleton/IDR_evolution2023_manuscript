"""Make figure of cluster heatmap."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from src.brownian.linkage import make_tree
from src.draw import plot_tree

pdidx = pd.IndexSlice
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

clusters = [('15150', 'A', ''),
            ('15107', 'B', ''),
            ('15104', 'C', ''),
            ('15056', 'D', ''),
            ('14889', 'E', ''),
            ('15053', 'F', ''),
            ('15072', 'G', ''),
            ('14741', 'H', ''),
            ('14379', 'I', ''),
            ('15123', 'J', ''),
            ('15153', 'K', ''),
            ('14916', 'L', '')]

color1 = '#4e79a7'

# Load regions
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

models = pd.read_table(f'../../IDR_evolution/analysis/brownian/model_compute/out/models_{min_length}.tsv', header=[0, 1])
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
    columns[f'{feature_label}_delta_loglikelihood'] = models[f'{feature_label}_loglikelihood_OU'] - models[f'{feature_label}_loglikelihood_BM']
    columns[f'{feature_label}_sigma2_ratio'] = models[f'{feature_label}_sigma2_BM'] / models[f'{feature_label}_sigma2_OU']
models = pd.concat([models, pd.DataFrame(columns)], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

# === ASR RATE HISTOGRAM ===
fig = plt.figure(figsize=(7.5, 3))
gs = plt.GridSpec(1, 2)
rect = (0.15, 0.25, 0.825, 0.7)

subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.add_axes(rect)
xs = asr_rates.loc[asr_rates['disorder'], 'aa_rate_mean']
ax.axvspan(min_aa_rate, xs.max(), color='#e6e6e6')
ax.hist(xs, bins=100)
ax.set_xlabel('Average amino acid rate in region')
ax.set_ylabel('Number of regions')
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

subfig = fig.add_subfigure(gs[0, 1], facecolor='none')
ax = subfig.add_axes(rect)
xs = asr_rates.loc[asr_rates['disorder'], 'indel_rate_mean']
ax.axvspan(min_indel_rate, xs.max(), color='#e6e6e6')
ax.hist(xs, bins=100)
ax.set_xlabel('Average indel rate in region')
ax.set_ylabel('Number of regions')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

fig.legend(handles=[Patch(facecolor=color1, label='disorder')], bbox_to_anchor=(0.5, -0.025), loc='lower center')
fig.savefig(f'out/hierarchy_histogram.png')
fig.savefig(f'out/hierarchy_histogram.tiff')
plt.close()

# === HIERARCHICAL HEATMAP ===
legend_args = {'aa_group': ('Amino acid content', 'grey', ''),
               'charge_group': ('Charge properties', 'black', ''),
               'physchem_group': ('Physiochemical properties', 'white', ''),
               'complexity_group': ('Repeats and complexity', 'white', 4 * '.'),
               'motifs_group': ('Motifs', 'white', 4 * '\\')}
group_labels = ['aa_group', 'charge_group', 'motifs_group', 'physchem_group', 'complexity_group']
group_labels_nonmotif = ['aa_group', 'charge_group', 'physchem_group', 'complexity_group']
gridspec_kw = {'width_ratios': [0.1, 0.65, 0.25], 'wspace': 0,
               'height_ratios': [0.975, 0.025], 'hspace': 0.01,
               'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.125}

column_labels = []
for group_label in group_labels:
    column_labels.extend([f'{feature_label}_delta_loglikelihood' for feature_label in feature_groups[group_label]])
data = models.loc[pdidx[:, :, :, True], column_labels]  # Re-arrange columns
array = np.nan_to_num(data.to_numpy(), nan=1)

cdm = pdist(array, metric='correlation')
lm = linkage(cdm, method='average')

# Convert to tree and calculate some useful data structures
tree = make_tree(lm)
tip_order = [int(tip.name) for tip in tree.tips()]
cluster_nodes = set()
for root_id, _, _ in clusters:
    root_node = tree.find(root_id)
    cluster_nodes.update(root_node.traverse())

# Get branch colors
node2color, node2tips = {}, {}
for node in tree.postorder():
    if node.is_tip():
        tips = 1
    else:
        tips = sum([node2tips[child] for child in node.children])
    node2tips[node] = tips
    if node in cluster_nodes:
        cmap = plt.colormaps['Reds_r']
    else:
        cmap = plt.colormaps['Greys_r']
    node2color[node] = cmap(max(0., (11 - tips) / 10))

fig, axs = plt.subplots(2, 3, figsize=(7.5, 7.5), gridspec_kw=gridspec_kw)

# Tree
ax = axs[0, 0]
plot_tree(tree, ax=ax, linecolor=node2color, linewidth=0.2, tip_labels=False,
          xmin_pad=0.025, xmax_pad=0)
ax.sharey(axs[0, 1])
ax.set_ylabel('Disorder regions')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Heatmap
ax = axs[0, 1]
im = ax.imshow(array[tip_order], aspect='auto', cmap=plt.colormaps['inferno'], interpolation='none')
ax.xaxis.set_label_position('top')
ax.set_xlabel('Features')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Corner axes
for ax in [axs[1, 0], axs[1, 2]]:
    ax.set_visible(False)

# Cluster blocks
ax = axs[0, 2]
id2idx = {tip.name: idx for idx, tip in enumerate(tree.tips())}
for root_id, cluster_id, cluster_label in clusters:
    root_node = tree.find(root_id)
    tips = list(root_node.tips())
    upper_idx = id2idx[tips[0].name]
    lower_idx = id2idx[tips[-1].name]

    rect = plt.Rectangle((0.05, upper_idx), 0.2, lower_idx - upper_idx, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(0.325, (upper_idx + lower_idx) / 2, cluster_id, va='center_baseline', ha='center', fontsize='medium', fontweight='bold')
    ax.text(0.4, (upper_idx + lower_idx) / 2, cluster_label, va='center_baseline', fontsize='x-small')
ax.sharey(axs[0, 1])
ax.set_axis_off()

# Legend
ax = axs[1, 1]
x = 0
handles = []
for group_label in group_labels:
    label, color, hatch = legend_args[group_label]
    dx = len(feature_groups[group_label]) / len(column_labels)
    rect = Rectangle((x, 0), dx, 1, label=label, facecolor=color, hatch=hatch,
                     edgecolor='black', linewidth=0.75, clip_on=False)
    ax.add_patch(rect)
    handles.append(rect)
    x += dx
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.25, 0), fontsize=8)
ax.set_axis_off()

# Colorbar
xcenter = gridspec_kw['width_ratios'][0] + gridspec_kw['width_ratios'][1] * 0.75
width = 0.2
ycenter = gridspec_kw['bottom'] / 2
height = 0.015
cax = fig.add_axes((xcenter - width / 2, ycenter - height / 2, width, height))
cax.set_title('$\mathregular{\log L_{OU} / L_{BM}}$', fontdict={'fontsize': 10})
fig.colorbar(im, cax=cax, orientation='horizontal')

fig.savefig(f'out/hierarchy.png', dpi=600)
fig.savefig(f'out/hierarchy.tiff', dpi=600)
plt.close()
