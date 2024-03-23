import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

csv_path = "../SC24-Result/A100/benchop_index_scatter.csv"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

color_palette = sns.color_palette("mako", n_colors=4)  # Adjust the number of colors as needed

# header dataset,feature_size,pyg_scatter_reduce,pyg_segment_coo,torch_scatter_reduce,index_scatter_reduce
df = pd.read_csv(csv_path)
# Assuming the DataFrame 'df' is already loaded and structured as specified
df_melted = df.melt(id_vars=["dataset", "feature_size"], var_name="method", value_name="time")
# Determine the order for the feature sizes and methods
feature_order = sorted(df_melted['feature_size'].unique())
method_name_mapping = {
    "pyg_scatter_reduce": "pyg_scatter_reduce",
    "pyg_segment_coo": "pyg_segment_coo",  # This seems to be the same, adjust if needed
    "torch_scatter_reduce": "torch_scatter_reduce",
    "index_scatter_reduce": "GeoT"
}

# Apply the mapping to the 'method' column
df_melted['method'] = df_melted['method'].map(method_name_mapping)
method_order = df_melted['method'].unique()  # Adjust this if you have a preferred order

# Normalize by 'pyg_scatter_reduce' for each dataset and feature size
df_melted['normalized_speedup'] = df_melted.groupby(['dataset', 'feature_size'])['time'].transform(lambda x: 1 / (x / x.iloc[0]))

sns.set_theme(style="whitegrid") # Set the Seaborn style
# Set global font to Arial (ensure Arial is available on your system)
plt.rcParams['font.family'] = 'Arial'

g = sns.FacetGrid(df_melted, col="dataset", col_wrap=4, height=4, aspect=1.5)
g.map_dataframe(sns.barplot, x="feature_size", y="normalized_speedup", hue="method", palette=color_palette, errorbar=None, order=feature_order, hue_order=method_order)

# Improve the legend
# g.add_legend(title="", title_fontsize=15, label_order=method_order, fontsize=13)
# Adjust the legend position to upper center
# sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=14)

plt.legend(title="", title_fontsize=15, fontsize=13, fancybox=False, shadow=False, edgecolor='white', loc='upper center')
plt.subplots_adjust(top=0.9)
# g.figure.suptitle('Segment Reduce Speedup (Normalized by PyG Scatter Reduce)', fontsize=18, fontweight='bold')

# Adjust labels and titles
for ax in g.axes.flatten():
    # Optionally, you can refine how the tick labels are displayed based on your data
    feature_sizes = df["feature_size"].unique()
    ax.set_xticks(range(len(feature_sizes)))  # Ensure there's a tick for each feature size
    ax.set_xticklabels(feature_sizes, rotation=45)
    ax.set_xlabel("Feature Size", fontsize=16, fontweight='normal')
    ax.set_ylabel("Normalized Speedup", fontsize=16, fontweight='normal')
    ax.tick_params(axis='x', labelsize=14)  # Adjust x-tick label size
    ax.tick_params(axis='y', labelsize=14)  # Adjust y-tick label size
    ax.set_title(ax.get_title(), fontsize=17, fontweight='bold')  # Adjust subplot title size 

plt.savefig("index_scatter_benchmark.pdf", dpi=300, bbox_inches='tight')
