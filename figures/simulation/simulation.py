"""Make figure for BM/OU hypothesis testing simulation."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from src.brownian.simulate.sampling import sigma2_min, sigma2_max, sigma2_delta, sigma2_num
from src.brownian.simulate.sampling import alpha_min, alpha_max, alpha_delta, alpha_num

df_BM = pd.read_table('../../IDR_evolution/analysis/brownian/simulate_compute/out/models_BM.tsv')
df_OU = pd.read_table('../../IDR_evolution/analysis/brownian/simulate_compute/out/models_OU.tsv')
dfs = [df_BM, df_OU]

for df in dfs:
    df['delta_loglikelihood'] = df['loglikelihood_hat_OU'] - df['loglikelihood_hat_BM']

groups_BM = df_BM.groupby('sigma2_id')
groups_OU = df_OU.groupby(['sigma2_id', 'alpha_id'])

if not os.path.exists('out/'):
    os.mkdir('out/')

# === MAIN FIGURE ===
fig = plt.figure(figsize=(7.5, 4.5))
gs = plt.GridSpec(2, 3)
gridspec_kw = {'left': 0.2, 'right': 0.9, 'bottom': 0.2, 'top': 0.9}

# BM MODEL PLOTS
# --- PANEL A: Violin plots of sigma2_BM ---
dataset = [group['sigma2_hat_BM'].apply(np.log10).to_list() for _, group in groups_BM]
positions = groups_BM['sigma2'].mean().apply(np.log10).to_list()

subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
violins = ax.violinplot(dataset, positions, widths=0.75*sigma2_delta, showmedians=True)
for key in ['cmins', 'cmaxes', 'cmedians', 'cbars']:
    violins[key].set_linewidth(0.75)
ax.set_xlabel('true $\mathregular{\log_{10}(\sigma^2_{BM})}$')
ax.set_ylabel('estimated $\mathregular{\log_{10}(\sigma^2_{BM})}$')
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# --- PANEL B: Type I error as function of cutoff ---
delta_loglikelihood_min, delta_loglikelihood_max = df_BM['delta_loglikelihood'].min(), df_BM['delta_loglikelihood'].max()
cutoffs = np.linspace(delta_loglikelihood_min, delta_loglikelihood_max, 50)

errors = []
for cutoff in cutoffs:
    errors.append(groups_BM['delta_loglikelihood'].aggregate(lambda x: (x > cutoff).mean()))
errors = pd.DataFrame(errors).reset_index(drop=True)

subfig = fig.add_subfigure(gs[0, 1], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
id2value = groups_BM['sigma2'].mean().apply(np.log10).to_dict()
cmap = ListedColormap(plt.colormaps['viridis'].colors[:240])
norm = Normalize(sigma2_min, sigma2_max)
get_color = lambda x: cmap(norm(x))
for sigma2_id in errors:
    ax.plot(cutoffs, errors[sigma2_id], color=get_color(id2value[sigma2_id]), alpha=0.75)
ax.set_xlabel('$\mathregular{\log L_{OU} - \log L_{BM}}$ cutoff')
ax.set_ylabel('Type I error')
subfig.colorbar(ScalarMappable(norm=norm, cmap=cmap))
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

# --- PANEL C: Type I error as function of cutoff (merge) ---
errors = []
for cutoff in cutoffs:
    errors.append((df_BM['delta_loglikelihood'] > cutoff).mean())
q95 = df_BM['delta_loglikelihood'].quantile(0.95)
q99 = df_BM['delta_loglikelihood'].quantile(0.99)

subfig = fig.add_subfigure(gs[0, 2], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
ax.plot(cutoffs, errors)
ax.axvline(q95, color='C1', label='5%')
ax.axvline(q99, color='C2', label='1%')
ax.set_xlabel('$\mathregular{\log L_{OU} - \log L_{BM}}$ cutoff')
ax.set_ylabel('Type I error')
ax.legend(title='Type I error',
          loc='center', bbox_to_anchor=(0.9, 0.5),
          title_fontsize=8, fontsize=8)
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

# OU MODEL PLOTS
df_parameter = groups_OU[['sigma2_hat_OU', 'alpha_hat_OU', 'sigma2', 'alpha']].mean().sort_index(level=['sigma2_id', 'alpha_id'])
df_error = (groups_OU['delta_loglikelihood'].aggregate(**{'q95': lambda x: (x < q95).mean(),
                                                          'q99': lambda x: (x < q99).mean()})
                                            .sort_index(level=['sigma2_id', 'alpha_id']))
extent = (alpha_min-alpha_delta/2, alpha_max+alpha_delta/2,
          sigma2_min-sigma2_delta/2, sigma2_max+sigma2_delta/2)

# --- PANEL D: Heatmap of sigma2 ---
array = np.log10((df_parameter['sigma2_hat_OU'] / df_parameter['sigma2']).to_numpy()).reshape((sigma2_num, alpha_num))
vext = max(abs(array.min()), abs(array.max()))

subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
im = ax.imshow(array,
               extent=extent, origin='lower', aspect='auto',
               cmap='RdBu', vmin=-vext, vmax=vext)
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
fig.colorbar(im)
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

# --- PANEL E: Heatmap of alpha ---
array = np.log10((df_parameter['alpha_hat_OU'] / df_parameter['alpha']).to_numpy()).reshape((sigma2_num, alpha_num))
vext = max(abs(array.min()), abs(array.max()))

subfig = fig.add_subfigure(gs[1, 1], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
im = ax.imshow(array,
               extent=extent, origin='lower', aspect='auto',
               cmap='RdBu', vmin=-vext, vmax=vext)
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
fig.colorbar(im)
subfig.suptitle('E', x=0.025, y=0.975, fontweight='bold')

# --- PANEL F: Heatmap of type II errors ---
subfig = fig.add_subfigure(gs[1, 2], facecolor='none')
ax = subfig.subplots(gridspec_kw=gridspec_kw)
array = df_error['q99'].to_numpy().reshape((sigma2_num, alpha_num))
im = ax.imshow(array, extent=extent, origin='lower', aspect='auto')
ax.set_xlabel('true $\mathregular{\log_{10}(\\alpha)}$')
ax.set_ylabel('true $\mathregular{\log_{10}(\sigma^2_{OU})}$')
fig.colorbar(im)
subfig.suptitle('F', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/simulation.png', dpi=300)
fig.savefig('out/simulation.tiff', dpi=300)
plt.close()
