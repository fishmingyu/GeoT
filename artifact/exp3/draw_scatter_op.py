import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

csv_path = "../../benchmark/benchop_index_scatter.csv"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# header dataset,feature_size,pyg_scatter_reduce,pyg_segment_coo,torch_scatter_reduce,index_scatter_reduce
df = pd.read_csv(csv_path)
header = ["dataset", "feature_size", "pyg_scatter_reduce", "pyg_segment_coo", "torch_scatter_reduce", "index_scatter_reduce"]

# Setting up the plot
fig, axs = plt.subplots(4, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle('Index Scatter Benchmark (Normalized by PyG Scatter Reduce)')
datasets = df["dataset"].unique()
width = 0.15  # Width of the bars

for i, dataset in enumerate(datasets):
    sub_df = df[df["dataset"] == dataset]
    ax = axs[i // 2, i % 2]
    ax.set_title(dataset)

    # Calculate bar positions
    r1 = np.arange(len(sub_df["feature_size"]))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    # Normalize by 'pyg_scatter_reduce'
    pyg_scatter_reduce_norm = sub_df["pyg_scatter_reduce"] / sub_df["pyg_scatter_reduce"]
    pyg_segment_coo_norm = sub_df["pyg_segment_coo"] / sub_df["pyg_scatter_reduce"]
    torch_scatter_reduce_norm = sub_df["torch_scatter_reduce"] / sub_df["pyg_scatter_reduce"]
    index_scatter_reduce_norm = sub_df["index_scatter_reduce"] / sub_df["pyg_scatter_reduce"]

    # Plotting the bars
    ax.bar(r1, pyg_scatter_reduce_norm, color='b', width=width, edgecolor='grey', label='PyG Scatter Reduce')
    ax.bar(r2, pyg_segment_coo_norm, color='r', width=width, edgecolor='grey', label='PyG Segment COO')
    ax.bar(r3, torch_scatter_reduce_norm, color='g', width=width, edgecolor='grey', label='Torch Scatter Reduce')
    ax.bar(r4, index_scatter_reduce_norm, color='y', width=width, edgecolor='grey', label='GeoS')

    # Adding labels
    ax.set_xlabel('Feature Size', fontweight='bold')
    ax.set_xticks([r + width for r in range(len(sub_df["feature_size"]))])
    ax.set_xticklabels(sub_df["feature_size"])
    ax.legend()

fig.savefig("benchmark_scatter.png")
