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
    new_percent['MatMul'] = df_pyg['Percentage'][df_pyg['Function'] == 'aten::addmm'].values[0] + df_pyg['Percentage'][df_pyg['Function'] == 'aten::mm'].values[0]
    new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])
    df_pyg = pd.DataFrame(new_percent, index=['{}_pyg'.format(dataset)])
    new_percent = {}
    new_percent['SpMM'] = df_GS['Percentage'][df_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]
    new_percent['MatMul'] = df_GS['Percentage'][df_GS['Function'] == 'aten::addmm'].values[0] + df_GS['Percentage'][df_GS['Function'] == 'aten::mm'].values[0]
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
