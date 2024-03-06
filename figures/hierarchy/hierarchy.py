"""Make figure of cluster heatmap."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from matplotlib.patches import Patch, Rectangle
from src.draw import plot_tree


def get_id_map(path):
    id2ids = {}
    with open(path) as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
            node_id = int(fields['node_id'])
            id2ids[node_id] = (OGid, start, stop, disorder)
    return id2ids


pdidx = pd.IndexSlice
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

clusters = [('15126', '1'),
            ('15136', '2'),
            ('15107', '3'),
            ('15065', '4'),
            ('14930', '5'),
            ('14935', '6'),
            ('14890', '7'),
            ('14971', '8'),
            ('15086', '9'),
            ('14939', '10'),
            ('15104', '11'),
            ('15056', '12'),
            ('15100', '13'),
            ('15132', '14'),
            ('14889', '15'),
            ('15134', '16'),
            ('15053', '17'),
            ('15072', '18'),
            ('14948', '19'),
            ('15081', '20'),
            ('14731', '21'),
            ('15146', '22'),
            ('14741', '23'),
            ('14379', '24'),
            ('14944', '25'),
            ('15083', '26'),
            ('14988', '27'),
            ('15165', '28'),
            ('15123', '29'),
            ('14743', '30'),
            ('15098', '31'),
            ('15153', '32'),
            ('15035', '33'),
            ('15062', '34'),
            ('15159', '35'),
            ('15047', '36'),
            ('14916', '37'),
            ('15026', '38'),
            ('15102', '39')]

# Load regions
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

# Filter by rates
asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

# Load models
models = pd.read_table(f'../../IDR_evolution/analysis/brownian/model_compute/out/models_{min_length}.tsv', header=[0, 1])
models = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
models = models.set_index(['OGid', 'start', 'stop', 'disorder'])

# Extract labels
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

# Calculate delta loglikelihood
columns = {}
for feature_label in feature_labels:
    columns[f'{feature_label}_delta_loglikelihood'] = models[f'{feature_label}_loglikelihood_OU'] - models[f'{feature_label}_loglikelihood_BM']
models = pd.concat([models, pd.DataFrame(columns)], axis=1)

# Load cluster trees
tree_LR = skbio.read('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_delta_loglikelihood_all_correlation.nwk',
                     'newick', skbio.TreeNode)
id2ids_LR = get_id_map('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_delta_loglikelihood_all_correlation.tsv')

tree_mu = skbio.read('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_mu_OU_all_correlation.nwk',
                     'newick', skbio.TreeNode)
id2ids_mu = get_id_map('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_mu_OU_all_correlation.tsv')

if not os.path.exists('out/'):
    os.mkdir('out/')

# === ASR RATE FILTERS ===
disorder = asr_rates.loc[asr_rates['disorder'], ['aa_rate_mean', 'indel_rate_mean']]
aa_idx = disorder['aa_rate_mean'] > min_aa_rate
indel_idx = disorder['indel_rate_mean'] > min_indel_rate

output = f"""\
# disorder regions: {len(disorder)}

pass aa_rate filter
#: {aa_idx.sum()}
%: {aa_idx.sum() / len(disorder):.2f}

pass indel_rate filter
#: {indel_idx.sum()}
%: {indel_idx.sum() / len(disorder):.2f}

pass either filter
#: {(aa_idx | indel_idx).sum()}
%: {(aa_idx | indel_idx).sum() / len(disorder):.2f}
"""
with open('out/filters.txt', 'w') as file:
    file.write(output)

# === ASR RATE HISTOGRAM ===
fig = plt.figure(figsize=(7.5, 3))
gs = plt.GridSpec(1, 2)
rect = (0.15, 0.25, 0.825, 0.7)
color1 = '#4e79a7'

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
metagroups = {'all': ['aa_group', 'charge_group', 'physchem_group', 'complexity_group', 'motifs_group'],
              'nonmotif': ['aa_group', 'charge_group', 'physchem_group', 'complexity_group']}
gridspec_kw = {'width_ratios': [0.1, 0.85, 0.1], 'wspace': 0,
               'height_ratios': [0.975, 0.025], 'hspace': 0.01,
               'left': 0.05, 'right': 0.99, 'top': 0.95, 'bottom': 0.15}

cmap = plt.colormaps['RdBu']
cmap.set_under((1.0, 0.0, 1.0))
cmap.set_over((0.0, 1.0, 1.0))
heatmap_args = {'delta_loglikelihood': {'cmap': plt.colormaps['inferno']},
                'mu_OU': {'cmap': cmap, 'vmin': -3, 'vmax': 3}}
colorbar_labels = {'delta_loglikelihood': '$\mathregular{\log L_{OU} / L_{BM}}$',
                   'mu_OU': '$z$-score of $\mathregular{\mu_{OU}}$'}
colorbar_args = {'delta_loglikelihood': {},
                 'mu_OU': {'extend': 'both', 'extendrect': True, 'extendfrac': 0.03}}

# Make data sets
data_set_parameters = [('delta_loglikelihood', 'all', tree_LR, id2ids_LR, False),
                       ('mu_OU', 'all', tree_mu, id2ids_mu, True)]
data_sets = {}
for statistic_label, metagroup_label, tree, id2ids, normalize in data_set_parameters:
    row_labels = []
    for node in tree.tips():
        row_labels.append(id2ids[int(node.name)])

    column_labels = []
    for group_label in metagroups[metagroup_label]:
        column_labels.extend([f'{feature_label}_delta_loglikelihood' for feature_label in feature_groups[group_label]])
    data = models.loc[row_labels, column_labels]  # Re-arrange rows and columns

    if normalize:
        data = (data - data.mean()) / data.std()

    data_sets[f'{statistic_label}_{metagroup_label}'] = data

# Make plots
# File labels are explicitly provided for backwards compatibility when generating different versions
# The original plot has the same file name so the most recent version of the repo can properly generate PLOS_v1
plots = [('delta_loglikelihood', 'all', 'hierarchy', tree_LR, clusters),
         ('mu_OU', 'all', 'hierarchy_mu_OU', tree_mu, [])]
for statistic_label, metagroup_label, file_label, tree, clusters in plots:
    data = data_sets[f'{statistic_label}_{metagroup_label}']
    array = np.nan_to_num(data.to_numpy(), nan=1)

    # Calculate some useful data structures
    cluster_nodes = set()
    node2root = {}
    for root_id, _ in clusters:
        root_node = tree.find(root_id)
        for node in root_node.traverse():
            cluster_nodes.add(node)
            node2root[node] = root_node

    # Get branch colors
    cmaps = [plt.colormaps[name] for name in ['Blues_r', 'Oranges_r', 'Greens_r', 'Reds_r', 'Purples_r']]
    id2color = {root_id: cmaps[i % len(cmaps)] for i, (root_id, _) in enumerate(clusters)}
    node2color, node2tips = {}, {}
    for node in tree.postorder():
        if node.is_tip():
            tips = 1
        else:
            tips = sum([node2tips[child] for child in node.children])
        node2tips[node] = tips
        if node in cluster_nodes:
            cmap = id2color[node2root[node].name]
        else:
            cmap = plt.colormaps['Greys_r']
        node2color[node] = cmap(max(0., (11 - tips) / 10))

    fig, axs = plt.subplots(2, 3, figsize=(5.2, 6), gridspec_kw=gridspec_kw)

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
    im = ax.imshow(array, aspect='auto', interpolation='none', **heatmap_args[statistic_label])
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
    for root_id, cluster_id in clusters:
        root_node = tree.find(root_id)
        tips = list(root_node.tips())
        upper_idx = id2idx[tips[0].name]
        lower_idx = id2idx[tips[-1].name]
        if len(tips) < 75:
            continue

        rect = plt.Rectangle((0.05, upper_idx), 0.2, lower_idx - upper_idx, facecolor='white',
                             edgecolor='black', linewidth=0.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.45, (upper_idx + lower_idx) / 2, cluster_id, va='center_baseline', ha='center', fontsize=6)
    ax.sharey(axs[0, 1])
    ax.set_axis_off()

    # Legend
    ax = axs[1, 1]
    x = 0
    handles = []
    for group_label in metagroups[metagroup_label]:
        label, color, hatch = legend_args[group_label]
        dx = len(feature_groups[group_label]) / len(data.columns)
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
    cax.set_title(colorbar_labels[statistic_label], fontsize=10)
    fig.colorbar(im, cax=cax, orientation='horizontal', **colorbar_args[statistic_label])

    fig.savefig(f'out/{file_label}.png', dpi=600)
    fig.savefig(f'out/{file_label}.tiff', dpi=600)
    plt.close()
