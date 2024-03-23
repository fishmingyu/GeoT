import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style

# Applying Seaborn styles
sns.set_style("whitegrid")

# Read the CSV file
df_filckr_pyg = pd.read_csv("../SC24-Result/breakdown/flickr_sparse_breakdown.csv")
df_filckr_GS = pd.read_csv("../SC24-Result/breakdown/flickr_sparse_GS_breakdown.csv")

new_percent = {}

# for pyg, calculate the "torch_sparse::spmm_sum" percentage using the tag "SpMM"
# check "torch_sparse::spmm_sum" is in the value of the function column
# then get the percentage
new_percent['SpMM'] = df_filckr_pyg['Percentage'][df_filckr_pyg['Function'] == 'torch_sparse::spmm_sum'].values[0]

# add up "aten::addmm" and "aten::mm" using the tag "MatMul"
# check "aten::addmm" and "aten::mm" are in the value of the function column
# then get the percentage
new_percent['MatMul'] = df_filckr_pyg['Percentage'][df_filckr_pyg['Function'] == 'aten::addmm'].values[0] + df_filckr_pyg['Percentage'][df_filckr_pyg['Function'] == 'aten::mm'].values[0]

# use 100 - (SpMM + MatMul + Format) to calculate the "Others" percentage
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

# Create a new dataframe with the new percentages
df_filckr_pyg = pd.DataFrame(new_percent, index=['filckr_pyg'])

# now we process the GeoT data
new_percent = {}

# for GeoT, calculate the "torch_index_scatter::gather_scatter" percentage using the tag "SpMM"
new_percent['SpMM'] = df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]

# add up "aten::addmm" and "aten::mm" using the tag "MatMul"
new_percent['MatMul'] = df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'aten::addmm'].values[0] + df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'aten::mm'].values[0]

# use 100 - (SpMM + MatMul + Format) to calculate the "Others" percentage
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

# Create a new dataframe with the new percentages
df_filckr_GS = pd.DataFrame(new_percent, index=['filckr_GS'])


df_filckr_pyg['Tech'] = 'PyG(sparse)'
df_filckr_GS['Tech'] = 'GeoT'

# Concatenate the two DataFrames vertically
df_combined_filckr = pd.concat([df_filckr_pyg, df_filckr_GS], ignore_index=True)
df_combined_filckr['Dataset'] = 'Flickr'

# now do the same process for dataset reddit2

df_reddit2_pyg = pd.read_csv("../SC24-Result/breakdown/reddit2_sparse_breakdown.csv")
df_reddit2_GS = pd.read_csv("../SC24-Result/breakdown/reddit2_sparse_GS_breakdown.csv")

new_percent = {}

new_percent['SpMM'] = df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'torch_sparse::spmm_sum'].values[0]
new_percent['MatMul'] = df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'aten::addmm'].values[0] + df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'aten::mm'].values[0]
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

df_reddit2_pyg = pd.DataFrame(new_percent, index=['reddit2_pyg'])

new_percent = {}

new_percent['SpMM'] = df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]
new_percent['MatMul'] = df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'aten::addmm'].values[0] + df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'aten::mm'].values[0]
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

df_reddit2_GS = pd.DataFrame(new_percent, index=['reddit2_GS'])

df_reddit2_pyg['Tech'] = 'PyG(sparse)'
df_reddit2_GS['Tech'] = 'GeoT'

df_combined_reddit2 = pd.concat([df_reddit2_pyg, df_reddit2_GS], ignore_index=True)
df_combined_reddit2['Dataset'] = 'Reddit2'


# do the same process for the ogbn-arxiv dataset
df_arxiv_pyg = pd.read_csv("../SC24-Result/breakdown/ogbn-arxiv_sparse_breakdown.csv")
df_arxiv_GS = pd.read_csv("../SC24-Result/breakdown/ogbn-arxiv_sparse_GS_breakdown.csv")

new_percent = {}

new_percent['SpMM'] = df_arxiv_pyg['Percentage'][df_arxiv_pyg['Function'] == 'torch_sparse::spmm_sum'].values[0]
new_percent['MatMul'] = df_arxiv_pyg['Percentage'][df_arxiv_pyg['Function'] == 'aten::addmm'].values[0] + df_arxiv_pyg['Percentage'][df_arxiv_pyg['Function'] == 'aten::mm'].values[0]
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

df_arxiv_pyg = pd.DataFrame(new_percent, index=['arxiv_pyg'])

new_percent = {}

new_percent['SpMM'] = df_arxiv_GS['Percentage'][df_arxiv_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]
new_percent['MatMul'] = df_arxiv_GS['Percentage'][df_arxiv_GS['Function'] == 'aten::addmm'].values[0] + df_arxiv_GS['Percentage'][df_arxiv_GS['Function'] == 'aten::mm'].values[0]
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'])

df_arxiv_GS = pd.DataFrame(new_percent, index=['arxiv_GS'])

df_arxiv_pyg['Tech'] = 'PyG(sparse)'
df_arxiv_GS['Tech'] = 'GeoT'

df_combined_arxiv = pd.concat([df_arxiv_pyg, df_arxiv_GS], ignore_index=True)

df_combined_arxiv['Dataset'] = 'ogbn-arxiv'

# Concatenate filckr, reddit2, and ogbn-arxiv DataFrames
df_combined = pd.concat([df_combined_filckr, df_combined_reddit2, df_combined_arxiv], ignore_index=True)
# melt the DataFrame
df_long = pd.melt(df_combined, id_vars=['Tech', 'Dataset'], var_name='Category', value_name='Percentage')


# Now, plotting with seaborn.objects, facetting by 'Dataset'
plot = (
    so.Plot(df_long, x="Tech", y="Percentage", color="Category")
    .facet("Dataset")
    .add(so.Bar(edgewidth=0, width=0.6), so.Stack())
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
    "axes.titlesize" : 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "font.size": 14,
    'font.family': 'Arial',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'legend.edgecolor': 'white',
    'legend.fancybox': False,
    'legend.loc': 'upper center',
    'figure.figsize': (12, 6),
}
so.Plot.config.theme.update(theme_dict)
plot.save("breakdown.pdf", bbox_inches='tight')
