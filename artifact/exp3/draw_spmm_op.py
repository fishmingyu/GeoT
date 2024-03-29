import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_path = "../SC24-Result/A100/benchop_spmm.csv"

header = ["dataset", "feature_size", "geos", "cusparse", "pyg_sparse"]
df = pd.read_csv(csv_path, header=None, names=header)

# Filter the dataframe to include only feature sizes 16, 32, and 64
df_filtered = df[df['feature_size'].isin([16, 32, 64, 128])].copy()

# if the dataset == amazon_photo, we rename it to amazon
df_filtered['dataset'] = df_filtered['dataset'].apply(lambda x: 'amazon' if x == 'amazon_photo' else x)

df_melted = pd.melt(df_filtered, id_vars=["dataset", "feature_size"], value_vars=["cusparse", "pyg_sparse", "geos"], var_name="Method")

method_name_mapping = {
    "geos": "GeoT",
    "cusparse": "cuSPARSE",
    "pyg_sparse": "PyG_SpMM"
}

df_melted['Method'] = df_melted['Method'].map(method_name_mapping)

# Normalize by 'torch(cusparse)' for each dataset and feature size
df_melted['normalized_speedup'] = df_melted.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)

max_speedup = df_melted.groupby('feature_size')['normalized_speedup'].max()

# Set global font to Arial (ensure Arial is available on your system)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set a base font size

# remove the background grid
sns.set_theme(style="whitegrid")

# Create a barplot with FacetGrid
g = sns.FacetGrid(df_melted, col="feature_size", col_wrap=4, height=4, aspect=1.2, sharey=False)
g.map_dataframe(sns.barplot, x="dataset", y="normalized_speedup", hue="Method", palette="mako")
# leave x axis label empty
g.set_xlabels('')
# Set y-axis label
g.set_ylabels("Normalized Speedup")
g.set_titles("Feature Size = {col_name}", fontsize=16, fontweight='bold')
# g.add_legend(title="", fontsize='12')
# sns.move_legend(g, "right", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=11)

# add the title
# g.figure.suptitle('SpMM Speedup (Normalized by cuSPARSE)', fontsize=17, fontweight='bold')
plt.subplots_adjust(top=0.9)
plt.legend(title="", title_fontsize=11, fontsize=11, fancybox=False, shadow=False, edgecolor='white', loc='upper right')

# Rotate x-axis labels
for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)  # Rotate x-axis labels
    ax.xaxis.label.set_size(13)  # Adjust font size of x-axis labels
    # Dynamically set y-axis limits based on maximum speedup for each feature size
    feature_size = int(ax.get_title().split('=')[1].strip())
    ax.set_ylim(0, max_speedup[feature_size] * 1.1)  # Adjust 1.1 for padding
    ax.set_title(ax.get_title(), fontsize=18, fontweight='bold')  # Adjust subplot title size
    # set x_ticks font size
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)  # Adjust y-tick label size

# only set once
for ax in g.axes.flatten():
    ax.set_ylabel("Normalized Speedup", fontsize=16, fontweight='normal')
    break

# Adjust layout to prevent clipping of x-axis labels
g.figure.subplots_adjust(bottom=0.2)

# Save the plot
plt.savefig("spmm_speedup.pdf", dpi=300, bbox_inches='tight')

# calculate the speedup across all datasets and feature sizes
geomean_speedup = df_melted.groupby(['Method'])['normalized_speedup'].apply(lambda x: x.prod() ** (1 / len(x)))
# GeoT vs cuSPARSE
speedup = geomean_speedup['GeoT'] / geomean_speedup['cuSPARSE']
print(f"Speedup of GeoT over cuSPARSE: {speedup:.2f}")

# GeoT vs PyG_SpMM
speedup = geomean_speedup['GeoT'] / geomean_speedup['PyG_SpMM']
print(f"Speedup of GeoT over PyG_SpMM: {speedup:.2f}")

# GeoT vs cuSPARSE
# filter GeoT and cuSPARSE
df_filtered = df_melted[df_melted['Method'].isin(['GeoT', 'cuSPARSE'])].copy()
# recalculated the normalized speedup
df_filtered['speedup'] = df_filtered.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)
# calculate the max speedup overall
max_speedup = df_filtered['speedup'].max()
print(f"Max speedup of GeoT vs cuSPARSE: {max_speedup:.2f}")

# GeoT vs PyG_SpMM
# filter GeoT and PyG_SpMM
df_filtered = df_melted[df_melted['Method'].isin(['GeoT', 'PyG_SpMM'])].copy()
# recalculated the normalized speedup
df_filtered['speedup'] = df_filtered.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)
# calculate the max speedup overall
max_speedup = df_filtered['speedup'].max()
print(f"Max speedup of GeoT vs PyG_SpMM: {max_speedup:.2f}")
