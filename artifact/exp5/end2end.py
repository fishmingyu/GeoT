import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# three models
# ['GCN', 'GIN', 'GraphSAGE']
# three datasets
# ['flickr', 'reddit2', 'ogbn-arxiv']

# load model_result.csv
header = ["model", "dataset", "hidden_size", "use_sparse", "use_geo", "time"]
df = pd.read_csv("../SC24-Result/A100/model_result.csv", header=None, names=header)

# if use_sparse == false, set method == 'PyG_Dense'
# if use_sparse == true, and use_geo == false, set method == 'PyG_Sparse'
# if use_sparse == true, and use_geo == true, set method == 'GeoT'

df['Method'] = df.apply(lambda x: 'PyG_Dense' if x['use_sparse'] == False else ('PyG_Sparse' if x['use_geo'] == False else 'GeoT'), axis=1)

# only leave the dataset, Method, and value columns

# filter the dataframe
# leave feature 32 and 64
df_filtered = df[df['hidden_size'].isin([32, 64])]
# only leave the dataset, model, Method, feature_size, and time columns
df_filtered = df_filtered[['dataset', 'model', 'Method', 'hidden_size', 'time']]

# reindex
df_filtered = df_filtered.reset_index(drop=True)

# calculate the normalized speedup based on PyG_Sparse
# we need to rearrange the method order from ['PyG_Dense', 'PyG_Sparse', 'GeoT'] to ['PyG_Sparse', 'PyG_Dense', 'GeoT']
df_filtered['Method'] = pd.Categorical(df_filtered['Method'], ['PyG_Sparse', 'PyG_Dense', 'GeoT'])
df_filtered = df_filtered.sort_values(by='Method')
# calculate the normalized speedup
df_filtered['normalized_speedup'] = df_filtered.groupby(['dataset', 'model', 'hidden_size'])['time'].transform(lambda x: 1/ ( x / x.iloc[0]))

# rearrange the dataset order
df_filtered['dataset'] = pd.Categorical(df_filtered['dataset'], ['flickr', 'ogbn-arxiv', 'reddit2',])

# rename the method
df_filtered['model'] = df_filtered.apply(lambda x: x['model'] + '-' + str(x['hidden_size']), axis=1)
# then we drop the hidden_size column
df_filtered = df_filtered.drop(columns=['hidden_size'])

# sort the model
df_filtered['model'] = pd.Categorical(df_filtered['model'], ['GCN-32', 'GIN-32', 'GraphSAGE-32', 'GCN-64', 'GIN-64', 'GraphSAGE-64'])

# set the font family to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# remove the background grid
sns.set_theme(style="whitegrid")

# Create a barplot with FacetGrid
g = sns.FacetGrid(df_filtered, col="model", col_wrap=3, height=2.1, aspect=1, sharey=True, sharex=True)
g.map_dataframe(sns.barplot, x="dataset", y="normalized_speedup", hue="Method", palette="mako")

# leave x axis label empty
g.set_xlabels('')

# Set y-axis label
g.set_ylabels("Normalized Speedup")

# add the title
# g.figure.suptitle('SpMM Speedup (Normalized by cuSPARSE)', fontsize=17, fontweight='bold')
# plt.subplots_adjust(top=0.9)

# plt.legend(title="", title_fontsize=8, fontsize=8, fancybox=False, shadow=False, edgecolor='white', loc='upper right')

# Rotate x-axis labels
for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelrotation=40, labelsize=12)  # Rotate x-axis labels
    ax.xaxis.label.set_size(10)  # Adjust font size of x-axis labels
    ax.yaxis.label.set_size(10)  # Adjust font size of y-axis labels
    # Dynamically set y-axis limits based on maximum speedup for each feature size
    # ax.set_ylim(0, max_speedup[feature_size] * 1.1)  # Adjust 1.1 for padding
    # remove the "Model = " prefix
    ax.set_title(ax.get_title().split('=')[1].strip(), fontsize=12, fontweight='bold')  # Adjust subplot title size
    # ax.set_title(ax.get_title(), fontsize=18, fontweight='bold')  # Adjust subplot title size
    # set x_ticks font size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)  # Adjust y-tick label size

g.axes[4].legend(title="", title_fontsize=8, fontsize=8, fancybox=False, shadow=False, edgecolor='white', loc='upper left', framealpha=0.5)

# save the figure
plt.savefig('end2end.pdf', dpi=300, bbox_inches='tight')