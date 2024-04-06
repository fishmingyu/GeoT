import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style

# Applying Seaborn styles
sns.set_style("whitegrid")


def process_df(dataset, feature):
    filename = "../SC24-Result/breakdown/{}_{}_sparse_breakdown.csv".format(dataset, feature)
    filename_GS = "../SC24-Result/breakdown/{}_{}_sparse_GS_breakdown.csv".format(dataset, feature)
    df_pyg = pd.read_csv(filename)
    df_GS = pd.read_csv(filename_GS)
    new_percent = {}
    new_percent['SpMM'] = df_pyg['Percentage'][df_pyg['Function'] == 'torch_sparse::spmm_sum'].values[0]
    new_percent['MatMul'] = df_pyg['Percentage'][df_pyg['Function'] == 'aten::mm'].values[0]
    new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])
    df_pyg = pd.DataFrame(new_percent, index=['{}_pyg'.format(dataset)])
    new_percent = {}
    new_percent['SpMM'] = df_GS['Percentage'][df_GS['Function'] == 'torch_index_scatter::gather_weight_scatter'].values[0]
    new_percent['MatMul'] = df_GS['Percentage'][df_GS['Function'] == 'aten::mm'].values[0]
    new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])
    df_GS = pd.DataFrame(new_percent, index=['{}_GS'.format(dataset)])
    df_pyg['Tech'] = 'PyG'
    df_GS['Tech'] = 'GeoT'
    df_combined = pd.concat([df_pyg, df_GS], ignore_index=True)
    # convert to uppercase for the first letter, and add feature size
    if dataset == 'ogbn-arxiv':
        dataset = 'arxiv'
    if dataset == 'reddit2':
        dataset = 'reddit'
    dataset = dataset[0].upper() + dataset[1:] + '-' + str(feature)
    df_combined['Dataset'] = dataset
    return df_combined

features = [32, 64]
datasets = ['flickr', 'ogbn-arxiv', 'reddit2']

# combine all the DataFrames
df_list = []
for dataset in datasets:
    for feature in features:
        df_combined = process_df(dataset, feature)
        df_list.append(df_combined)

# Concatenate all the DataFrames
df_combined = pd.concat(df_list, ignore_index=True)

# melt the DataFrame
df_long = pd.melt(df_combined, id_vars=['Dataset', 'Tech'], var_name='Category', value_name='Percentage')

# Now, plotting with seaborn.objects, facetting by 'Dataset'
plot = (
    so.Plot(df_long, x="Tech", y="Percentage", color="Category")
    .facet("Dataset")
    .add(so.Bar(edgewidth=0, width=0.5), so.Stack())
    .scale(color="mako")
    .label(legend="Category", x="", y="Percentage (%)")
    # .layout(size=(12, 6))
)

from matplotlib import font_manager

font_dirs = ["/usr/share/fonts/truetype/msttcorefonts/"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


plot.theme(axes_style("white"))
so.Plot.config.theme.update(axes_style("ticks"))
# configure font type and size
theme_dict = {
    "axes.titlesize" : 23,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "font.size": 16,
    'font.family': 'Arial',
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'legend.title_fontsize': 17,
    'legend.edgecolor': 'white',
    'legend.fancybox': False,
    'legend.loc': 'best',
    'figure.figsize': (12.5, 6.5),
}
so.Plot.config.theme.update(theme_dict)
plot.save("breakdown.pdf", bbox_inches='tight')

# calculate the SpMM percentage reduction of GeoT over PyG
"""
     SpMM  MatMul  Others  Tech    Dataset
0   51.33    8.44   40.23   PyG  Flickr-32
1   11.43   12.54   76.03  GeoT  Flickr-32
2   56.10   10.62   33.28   PyG  Flickr-64
3   14.59   18.01   67.40  GeoT  Flickr-64
4   73.12    3.17   23.71   PyG   Arxiv-32
5   22.08    8.25   69.67  GeoT   Arxiv-32
6   75.57    4.18   20.25   PyG   Arxiv-64
7   23.94   11.86   64.20  GeoT   Arxiv-64
8   83.44    2.24   14.32   PyG  Reddit-32
9   56.89    5.55   37.56  GeoT  Reddit-32
10  86.63    2.96   10.41   PyG  Reddit-64
11  62.70    7.89   29.41  GeoT  Reddit-64
"""
# only select the SpMM, Tech, and Dataset columns
df_spmm = df_combined[['SpMM', 'Tech', 'Dataset']]
# calculate the SpMM percentage reduction of GeoT over PyG
df_spmm = df_spmm.pivot(index='Dataset', columns='Tech', values='SpMM')

df_spmm['Reduction'] = (df_spmm['PyG'] - df_spmm['GeoT'])

# average the reduction
reduction = df_spmm['Reduction'].mean()
# maximum reduction
max_reduction = df_spmm['Reduction'].max()

print(f"Average SpMM percentage reduction of GeoT over PyG: {reduction:.2f}")
print(f"Maximum SpMM percentage reduction of GeoT over PyG: {max_reduction:.2f}")
