"""Make figures for basic region statistics."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from numpy import linspace, sqrt
from src.utils import read_fasta

pdidx = pd.IndexSlice
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

# Load regions as segments
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        for ppid in fields['ppids'].split(','):
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder,
                         'ppid': ppid, 'min_length': min_length})
all_segments = pd.DataFrame(rows)
all_regions = all_segments[['OGid', 'start', 'stop', 'disorder']].drop_duplicates()

# Get segment lengths
rows = []
for OGid, group in all_segments[['OGid', 'start', 'stop']].drop_duplicates().groupby('OGid'):
    msa = read_fasta(f'../../IDR_evolution/data/alignments/fastas/{OGid}.afa')
    msa = [(re.search(ppid_regex, header).group(1), seq) for header, seq in msa]

    for row in group.itertuples():
        region_length = row.stop - row.start
        for ppid, seq in msa:
            segment = seq[row.start:row.stop]
            segment_length = region_length - segment.count('.') - segment.count('-')
            rows.append({'OGid': OGid, 'start': row.start, 'stop': row.stop, 'ppid': ppid, 'length': segment_length})
all_lengths = pd.DataFrame(rows)

segments = all_segments[all_segments['min_length'] == min_length]
segments = segments.merge(all_lengths, how='left', on=['OGid', 'start', 'stop', 'ppid'])
regions = segments.groupby(['OGid', 'start', 'stop', 'disorder'])

means = regions.mean()
disorder = means.loc[pdidx[:, :, :, True], :]
order = means.loc[pdidx[:, :, :, False], :]

asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 3.5))
gs = plt.GridSpec(1, 2)

# Mean region length histogram
subfig = fig.add_subfigure(gs[0, 0])
axs = subfig.subplots(2, 1, sharex=True, gridspec_kw={'left': 0.2, 'bottom': 0.15})
xmin, xmax = means['length'].min(), means['length'].max()
axs[0].hist(disorder['length'], bins=linspace(xmin, xmax, 100), color='C0', label='disorder')
axs[1].hist(order['length'], bins=linspace(xmin, xmax, 100), color='C1', label='order')
axs[1].set_xlabel('Mean length of region')
for ax in axs:
    ax.set_ylabel('Number of regions')
    ax.legend()
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# ASR rate boxplot
x1 = (asr_rates.loc[asr_rates['disorder'] == True, 'aa_rate_mean'] +
      asr_rates.loc[asr_rates['disorder'] == True, 'indel_rate_mean']).dropna()
x2 = (asr_rates.loc[asr_rates['disorder'] == False, 'aa_rate_mean'] +
      asr_rates.loc[asr_rates['disorder'] == False, 'indel_rate_mean']).dropna()

# Manual calculation of log p-value due to underflow
# Code adapted from SciPy stats example in mannwhitneyu function
nx1, nx2 = len(x1), len(x2)
N = nx1 + nx2
U1, p = stats.mannwhitneyu(x1, x2, alternative='greater')
U2 = nx1 * nx2 - U1
U = min(U1, U2)
z = (U - nx1 * nx2 / 2 + 0.5) / sqrt(nx1 * nx2 * (N + 1) / 12)
logp = 2 * stats.norm.logcdf(z)
with open('out/utest.txt', 'w') as file:
    file.write(f'log p-value (Mann-Whitney U test): {logp}\n')

subfig = fig.add_subfigure(gs[0, 1])
ax = subfig.subplots()
ax.boxplot([x1, x2], labels=['disorder', 'order'])
ax.set_ylabel('Total substitution rate')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/regions.png', dpi=300)
fig.savefig('out/regions.tiff', dpi=300)
plt.close()
