import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

csv_path = "../../benchmark/benchop_spmm.csv"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming 'csv_path' is the path to your CSV file and it's already defined.
header = ["dataset", "feature_size", "geos", "torch(cusparse)", "pyg_sparse"]
df = pd.read_csv(csv_path, header=None, names=header)

# Setting up the plot
fig, axs = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
fig.suptitle('SPMM Benchmark (Normalized by GEOS)')

datasets = df["dataset"].unique()
width = 0.25  # Width of the bars

for i, dataset in enumerate(datasets):
    sub_df = df[df["dataset"] == dataset]
    ax = axs[i // 3, i % 3]
    ax.set_title(dataset)

    # Calculate bar positions
    r1 = np.arange(len(sub_df["feature_size"]))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    # Normalize by 'geos'
    geos_norm = sub_df["geos"] / sub_df["geos"]
    torch_norm = sub_df["torch(cusparse)"] / sub_df["geos"]
    pyg_sparse_norm = sub_df["pyg_sparse"] / sub_df["geos"]

    # Plotting the bars
    ax.bar(r1, geos_norm, color='b', width=width, edgecolor='grey', label='GEOS')
    ax.bar(r2, torch_norm, color='r', width=width, edgecolor='grey', label='Torch(cusparse)')
    ax.bar(r3, pyg_sparse_norm, color='g', width=width, edgecolor='grey', label='PyG Sparse')

    # Adding labels
    ax.set_xlabel('Feature Size', fontweight='bold')
    ax.set_xticks([r + width for r in range(len(sub_df["feature_size"]))])
    ax.set_xticklabels(sub_df["feature_size"])
    ax.legend()

fig.savefig("benchmark_spmm.png")
