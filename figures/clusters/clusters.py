"""Plot individual clusters with features."""

import os
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio

pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

tree_template = skbio.read('../../IDR_evolution/data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

clusters = [('15056', '12'),
            ('14889', '15'),
            ('14741', '23'),
            ('14379', '24')]

# Load regions
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields[
                                                                                                     'disorder'] == 'True'
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
models = pd.read_table(f'../../IDR_evolution/analysis//brownian/model_compute/out/models_{min_length}.tsv', header=[0, 1])
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

# Load cluster tree
tree_cluster = skbio.read('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_all_correlation.nwk',
                          'newick', skbio.TreeNode)
id2ids = {}
with open('../../IDR_evolution/analysis/brownian/model_stats/out/regions_30/hierarchy/heatmap_all_correlation.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        node_id = int(fields['node_id'])
        id2ids[node_id] = (OGid, start, stop, disorder)

prefix = 'out/'
if not os.path.exists(prefix):
    os.mkdir(prefix)

group_labels = ['aa_group', 'charge_group', 'physchem_group', 'complexity_group', 'motifs_group']
gridspec_kw = {'left': 0.025, 'right': 0.995, 'bottom': 0.375, 'top': 0.85, 'hspace': 0}
cmap = plt.colormaps['RdBu']
cmap.set_under((1.0, 0.0, 1.0))
cmap.set_over((0.0, 1.0, 1.0))

column_labels = []
for group_label in group_labels:
    column_labels.extend([f'{feature_label}_delta_loglikelihood' for feature_label in feature_groups[group_label]])
array = models[column_labels].to_numpy()
vmin, vmax = array.min(), array.max()

fig = plt.figure(figsize=(7.5, 8.75))
gs = plt.GridSpec(len(clusters), 1)
for idx, (root_id, cluster_id) in enumerate(clusters):
    row_labels = []
    for node in tree_cluster.find(root_id).tips():
        row_labels.append(id2ids[int(node.name)])
    xs_labels = []
    for group_label in group_labels:
        xs_labels.extend(feature_groups[group_label])

    # --- SUBPANEL 1: delta loglikelihood heatmap ---
    column_labels = []
    for group_label in group_labels:
        column_labels.extend([f'{feature_label}_delta_loglikelihood' for feature_label in feature_groups[group_label]])
    data = models.loc[row_labels, column_labels]  # Re-arrange rows and columns
    array = np.nan_to_num(data.to_numpy(), nan=1)

    subfig = fig.add_subfigure(gs[idx])
    axs = subfig.subplots(2, 1, gridspec_kw=gridspec_kw)

    ax = axs[0]
    im0 = ax.imshow(array, aspect='auto', cmap=plt.colormaps['inferno'], interpolation='none',
                    vmin=vmin, vmax=vmax)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.text(0, 1.05, f'Cluster {cluster_id}', fontsize=8, transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- SUBPANEL 2: feature heatmap ---
    column_labels = []
    for group_label in group_labels:
        column_labels.extend([f'{feature_label}_mu_OU' for feature_label in feature_groups[group_label]])
    data = models.loc[row_labels, column_labels]   # Re-arrange rows and columns
    data = (data - models[column_labels].mean(axis=0)) / models[column_labels].std(axis=0)
    array = np.nan_to_num(data.to_numpy(), nan=1)

    ax = axs[1]
    im1 = ax.imshow(array, aspect='auto', cmap=cmap, interpolation='none',
                    vmin=-3, vmax=3)
    ax.set_xticks(range(len(xs_labels)), xs_labels, fontsize=5.5,
                  rotation=60, rotation_mode='anchor', ha='right', va='center')
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- COLORBARS ---
    ax = axs[0]
    cax = ax.inset_axes((0.65, 1.25, 0.15, 0.1))
    cbar = subfig.colorbar(im0, cax=cax, orientation='horizontal')
    cax.set_title('$\mathregular{\log L_{OU} / L_{BM}}$', fontsize=8, pad=1)
    cax.tick_params(labelsize=5.5, pad=1, length=2)
    cax = ax.inset_axes((0.85, 1.25, 0.15, 0.1))
    subfig.colorbar(im1, cax=cax, orientation='horizontal',
                    extend='both', extendrect=True, extendfrac=0.03)
    cax.set_title('$z$-score of $\mathregular{\mu_{OU}}$', fontsize=8, pad=1)
    cax.tick_params(labelsize=5.5, pad=1, length=2)

    subfig.suptitle(ascii_uppercase[idx], x=0.0125, y=0.975, fontweight='bold')

fig.savefig('out/clusters.png', dpi=300)
fig.savefig('out/clusters.tiff', dpi=300)
plt.close()
