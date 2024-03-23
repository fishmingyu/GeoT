import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
# /mnt/disk3/xinchen/projects/torch_index_scatter/benchmark/benchmark_cpp/../../data/eval_data/amazon_photo_idx.npy,1,1,8,4,8,0.00904704,26.3249
header = ['dataset', 'features', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
df = pd.read_csv('../SC24-Result/Ablation/sr_result.csv', header=None, names=header)

# split the dataset column
df['dataset'] = df['dataset'].str.split('/').str[-1].str.split('_idx.npy').str[0]

# select the dataset amazon_photo and features == 64
df_amz = df[(df['dataset'] == 'amazon_photo') & (df['features'] == 64)]

# select the dataset ogbn-arxiv and features == 64
df_arxiv = df[(df['dataset'] == 'ogbn-arxiv') & (df['features'] == 64)]

# set config1 == 1, config4 == 4, we draw the heatmap for the gflops
df_amz = df_amz[(df_amz['config1'] == 1) & (df_amz['config4'] == 4)]
df_arxiv = df_arxiv[(df_arxiv['config1'] == 1) & (df_arxiv['config4'] == 4)]

# pivot the table
df_amz = df_amz.pivot(index='config2', columns='config3', values='gflops')
df_arxiv = df_arxiv.pivot(index='config2', columns='config3', values='gflops')

# plot the heatmap
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
fig, ax = plt.subplots(2, 1, figsize=(6, 5))

# set mako palette
color = sns.color_palette("GnBu", as_cmap=True)


sns.heatmap(df_amz, ax=ax[0], annot=True, fmt=".3f", cmap=color)
ax[0].set_title('Amazon-Photo')
ax[0].set_ylabel('$T_N$')
ax[0].set_xlabel('')
sns.heatmap(df_arxiv, ax=ax[1], annot=True, fmt=".3f", cmap=color)
ax[1].set_title('ogbn-arxiv')
ax[1].set_xlabel('$M_t$')
ax[1].set_ylabel('$T_N$')

# set font size of subfigure title
for i in range(2):
    ax[i].title.set_fontsize(14)
    ax[i].title.set_fontweight('bold') 
    # label size
    ax[i].xaxis.label.set_fontsize(12)
    ax[i].yaxis.label.set_fontsize(12)

plt.tight_layout()

plt.savefig('heatmap.png', dpi=300)