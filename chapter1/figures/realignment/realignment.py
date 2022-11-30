"""Make figure of alignments before and after alignment with HMMer."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import skbio
from src.draw import plot_msa
from src.utils import read_fasta

records = [('23D9', 'A', 731, 1158, 704, 1000),
           ('01E0', 'B', 0, 344, 0, 94),
           ('07C3', 'C', 0, 484, 0, 211),
           ('0430', 'D', 0, 476, 0, 126)]
msa_path1 = '../../orthology_inference/analysis/ortho_MSA/get_repseqs/out/'
msa_path2 = '../../orthology_inference/analysis/ortho_MSA/realign_hmmer/out/mafft/'
spid_regex = r'spid=([a-z]+)'

adjust_left1 = 0.02
adjust_left2 = 0.05
adjust_bottom = 0.1
adjust_right1 = 0.95
adjust_right2 = 0.85
adjust_top = 0.9

hspace = 0.3
x_labelsize = 5
legend_position = (0.86, 0.5)
legend_fontsize = 5
legend_handletextpad = 0.5
legend_markerscale = 1

fig_width = 3.75
fig_height = 2.5
figsize_effective1 = (fig_width * (adjust_right1 - adjust_left1), fig_height * (adjust_top - adjust_bottom))
figsize_effective2 = (fig_width * (adjust_right2 - adjust_left2), fig_height * (adjust_top - adjust_bottom))
dpi = 600

panel_label_fontsize = 'large'
panel_label_offset = 0.025
suptitle_fontsize = 'medium'

tree_order = skbio.read('../../orthology_inference/analysis/ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_order.tips())}

if not os.path.exists('out/'):
    os.mkdir('out/')

for OGid, panel_label, start1, stop1, start2, stop2 in records:
        path1 = f'{msa_path1}/{OGid}.afa'
        path2 = f'{msa_path2}/{OGid}.afa'

        msa1 = []
        for header, seq in read_fasta(path1):
            spid = re.search(spid_regex, header).group(1)
            msa1.append({'spid': spid, 'seq': seq})
        msa1 = sorted(msa1, key=lambda x: tip_order[x['spid']])

        msa2 = []
        for header, seq in read_fasta(path2):
            spid = re.search(spid_regex, header).group(1)
            msa2.append({'spid': spid, 'seq': seq})
        msa2 = sorted(msa2, key=lambda x: tip_order[x['spid']])

        # Plot original MSA
        fig = plot_msa([record['seq'][start1:stop1] for record in msa1],
                       x_start=start1,
                       figsize=figsize_effective1, hspace=hspace,
                       x_labelsize=x_labelsize)
        fig.text(panel_label_offset / fig_height, 1 - panel_label_offset / fig_width, panel_label, fontsize=panel_label_fontsize, fontweight='bold',
                 horizontalalignment='left', verticalalignment='top')
        fig.set_size_inches((fig_width, fig_height))
        plt.subplots_adjust(left=adjust_left1, bottom=adjust_bottom, right=adjust_right1, top=adjust_top)
        plt.suptitle('Before', fontsize=suptitle_fontsize)
        plt.savefig(f'out/{panel_label}1_{OGid}.png', dpi=dpi)
        plt.savefig(f'out/{panel_label}1_{OGid}.tiff', dpi=dpi)
        plt.close()

        # Plot re-aligned MSA
        fig = plot_msa([record['seq'][start2:stop2] for record in msa2],
                       x_start=start2,
                       figsize=figsize_effective2, hspace=hspace,
                       x_labelsize=x_labelsize,
                       msa_legend=True, legend_kwargs={'bbox_to_anchor': legend_position, 'loc': 'center left', 'fontsize': legend_fontsize,
                                                       'handletextpad': legend_handletextpad, 'markerscale': legend_markerscale, 'handlelength': 1})
        fig.set_size_inches((fig_width, fig_height))
        plt.subplots_adjust(left=adjust_left2, bottom=adjust_bottom, right=adjust_right2, top=adjust_top)
        plt.suptitle('After', fontsize=suptitle_fontsize)
        plt.savefig(f'out/{panel_label}2_{OGid}.png', dpi=dpi)
        plt.savefig(f'out/{panel_label}2_{OGid}.tiff', dpi=dpi)
        plt.close()

images = []
for record in records[:-1]:
    OGid, panel_label = record[0], record[1]
    image1 = plt.imread(f'out/{panel_label}1_{OGid}.png')
    image2 = plt.imread(f'out/{panel_label}2_{OGid}.png')
    images.append(np.concatenate([image1, image2], axis=1))
image = np.concatenate(images)
plt.imsave('out/merged.png', image)
