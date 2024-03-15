import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_path = "../SC24-Result/A100/benchop_spmm.csv"

header = ["dataset", "feature_size", "geos", "cusparse", "pyg_sparse"]
df = pd.read_csv(csv_path, header=None, names=header)

# Filter the dataframe to include only feature sizes 16, 32, and 64
df_filtered = df[df['feature_size'].isin([16, 32, 64])]

df_melted = pd.melt(df_filtered, id_vars=["dataset", "feature_size"], value_vars=["cusparse", "pyg_sparse", "geos"], var_name="Method")

# Normalize by 'torch(cusparse)' for each dataset and feature size
df_melted['normalized_speedup'] = df_melted.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)

max_speedup = df_melted.groupby('feature_size')['normalized_speedup'].max()

# Set global font to Arial (ensure Arial is available on your system)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set a base font size

# remove the background grid
sns.set_theme(style="ticks")  # Set the Seaborn style

# Create a barplot with FacetGrid
g = sns.FacetGrid(df_melted, col="feature_size", col_wrap=1, height=3, aspect=1.6)
g.map_dataframe(sns.barplot, x="dataset", y="normalized_speedup", hue="Method", palette="mako")
# leave x axis label empty
g.set_xlabels('')
# Set y-axis label
g.set_ylabels("Normalized Speedup")
g.set_titles("Feature Size = {col_name}", fontsize=15)
g.add_legend(title="Method", title_fontsize='13', fontsize='11')


# add the title
g.figure.suptitle('SpMM Speedup (Normalized by cuSPARSE)', fontsize=17, fontweight='bold')
plt.subplots_adjust(top=0.9)

# Rotate x-axis labels
for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelrotation=50)
    ax.xaxis.label.set_size(12)  # Adjust font size of x-axis labels
    # Dynamically set y-axis limits based on maximum speedup for each feature size
    feature_size = int(ax.get_title().split('=')[1].strip())
    ax.set_ylim(0, max_speedup[feature_size] * 1.1)  # Adjust 1.1 for padding


# Adjust layout to prevent clipping of x-axis labels
g.figure.subplots_adjust(bottom=0.12)

# Save the plot
plt.savefig("spmm_speedup.png", dpi=300)  # Increase dpi for higher resolution if needed