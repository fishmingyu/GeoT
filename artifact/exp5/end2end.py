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
# leave feature 64
df_filtered = df[df['hidden_size'] == 64]
# only leave the dataset, model, Method, and time columns
df_filtered = df_filtered[['dataset', 'model', 'Method', 'time']]

# reindex
df_filtered = df_filtered.reset_index(drop=True)

# calculate the normalized speedup
df_filtered['normalized_speedup'] = df_filtered.groupby(['dataset', 'model'])['time'].transform(lambda x: 1 /(x / x.iloc[0]))

# melt the dataframe
df_melted = pd.melt(df_filtered, id_vars=["dataset", "model", "Method"], value_vars=["normalized_speedup"], var_name="Metric")

# remove the background grid
sns.set_theme(style="ticks")  # Set the Seaborn style

# Create the FacetGrid object and map the barplot
g = sns.FacetGrid(df_melted, col="model", col_wrap=3, height=3.2, aspect=0.7, sharey=False)
g.map_dataframe(sns.barplot, x="dataset", y="value", hue="Method", palette="mako")

# Remove the background grid and set styles
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Set labels
g.set_xlabels('')
g.set_ylabels("Normalized Speedup")
# legend font
# g.add_legend(title="Method", title_fontsize='9', fontsize='9')
# sns.move_legend(g, "right", bbox_to_anchor=(1, 0.5), ncol=1, fontsize=9, title_fontsize=9)
plt.legend(title="Method", title_fontsize=9, fontsize=9, fancybox=False, shadow=False, edgecolor='white', loc='upper center')

# Iterate over axes to adjust y-axis limits independently
for ax in g.axes.flatten():
    # Get the maximum value within this subplot to set the y-axis limit
    max_val = max([p.get_height() for p in ax.patches])
    ax.set_ylim(0, max_val * 1.1)  # Add 10% padding above the tallest bar
    # revise the subfigure title
    model = ax.get_title().split('=')[1].strip()
    ax.set_title(model, fontsize=13, fontweight='bold')
    # Rotate x-axis labels and adjust font size
    ax.tick_params(axis='x', labelrotation=50)
    # Adjust xtick label size
    ax.tick_params(axis='x', labelsize=10)
    # Adjust ylabel size
    ax.yaxis.label.set_size(11)


# Adjust layout to prevent clipping of x-axis labels
g.figure.subplots_adjust(bottom=0.25)

# Save the plot
plt.savefig("model_inference_speedup.pdf", dpi=300, bbox_inches='tight')  # Increase dpi for higher resolution if needed