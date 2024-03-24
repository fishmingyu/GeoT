import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

a100_index_scatter = "../SC24-Result/A100/benchop_index_scatter.csv"
h100_index_scatter = "../SC24-Result/H100/benchop_index_scatter.csv"
rtx3090Ti_index_scatter = "../SC24-Result/3090Ti/benchop_index_scatter.csv"

# for index scatter, we set baseline as pyg_scatter_reduce
df_a100_index_scatter = pd.read_csv(a100_index_scatter)
df_h100_index_scatter = pd.read_csv(h100_index_scatter)
df_rtx3090Ti_index_scatter = pd.read_csv(rtx3090Ti_index_scatter)


df_a100_index_scatter_melted = df_a100_index_scatter.melt(id_vars=["dataset", "feature_size"], var_name="method", value_name="time")
df_h100_index_scatter_melted = df_h100_index_scatter.melt(id_vars=["dataset", "feature_size"], var_name="method", value_name="time")
df_rtx3090Ti_index_scatter_melted = df_rtx3090Ti_index_scatter.melt(id_vars=["dataset", "feature_size"], var_name="method", value_name="time")
df_a100_index_scatter_melted['GPU'] = 'A100'
df_h100_index_scatter_melted['GPU'] = 'H100'
df_rtx3090Ti_index_scatter_melted['GPU'] = 'RTX3090Ti'


df_a100_index_scatter_melted['normalized_speedup'] = df_a100_index_scatter_melted.groupby(['dataset', 'feature_size'])['time'].transform(lambda x: 1 / (x / x.iloc[0]))
df_h100_index_scatter_melted['normalized_speedup'] = df_h100_index_scatter_melted.groupby(['dataset', 'feature_size'])['time'].transform(lambda x: 1 / (x / x.iloc[0]))
df_rtx3090Ti_index_scatter_melted['normalized_speedup'] = df_rtx3090Ti_index_scatter_melted.groupby(['dataset', 'feature_size'])['time'].transform(lambda x: 1 / (x / x.iloc[0]))

# only leave the GPU and normalized_speedup columns for method==index_scatter_reduce
df_a100_index_scatter = df_a100_index_scatter_melted[df_a100_index_scatter_melted['method'] == 'index_scatter_reduce'][['GPU', 'normalized_speedup']]
df_h100_index_scatter = df_h100_index_scatter_melted[df_h100_index_scatter_melted['method'] == 'index_scatter_reduce'][['GPU', 'normalized_speedup']]
df_rtx3090Ti_index_scatter = df_rtx3090Ti_index_scatter_melted[df_rtx3090Ti_index_scatter_melted['method'] == 'index_scatter_reduce'][['GPU', 'normalized_speedup']]

# Combine the dataframes
df_combined = pd.concat([df_a100_index_scatter, df_h100_index_scatter, df_rtx3090Ti_index_scatter], ignore_index=True)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# draw another subfigure based on spmm
a100_spmm = "../SC24-Result/A100/benchop_spmm.csv"
h100_spmm = "../SC24-Result/H100/benchop_spmm.csv"
rtx3090Ti_spmm = "../SC24-Result/3090Ti/benchop_spmm.csv"

header = ["dataset", "feature_size", "geos", "cusparse", "pyg_sparse"]
df_a100_spmm = pd.read_csv(a100_spmm, header=None, names=header)
df_h100_spmm = pd.read_csv(h100_spmm, header=None, names=header)
df_rtx3090Ti_spmm = pd.read_csv(rtx3090Ti_spmm, header=None, names=header)

# no filter
df_a100_spmm_melted = pd.melt(df_a100_spmm, id_vars=["dataset", "feature_size"], value_vars=["cusparse", "pyg_sparse", "geos"], var_name="Method")
df_h100_spmm_melted = pd.melt(df_h100_spmm, id_vars=["dataset", "feature_size"], value_vars=["cusparse", "pyg_sparse", "geos"], var_name="Method")
df_rtx3090Ti_spmm_melted = pd.melt(df_rtx3090Ti_spmm, id_vars=["dataset", "feature_size"], value_vars=["cusparse", "pyg_sparse", "geos"], var_name="Method")

df_a100_spmm_melted['GPU'] = 'A100'
df_h100_spmm_melted['GPU'] = 'H100'
df_rtx3090Ti_spmm_melted['GPU'] = 'RTX3090Ti'

df_a100_spmm_melted['normalized_speedup'] = df_a100_spmm_melted.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)
df_h100_spmm_melted['normalized_speedup'] = df_h100_spmm_melted.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)
df_rtx3090Ti_spmm_melted['normalized_speedup'] = df_rtx3090Ti_spmm_melted.groupby(['dataset', 'feature_size'])['value'].transform(lambda x: x / x.iloc[0] if x.name == 'cusparse' else x.iloc[0] / x)

# only leave the GPU and normalized_speedup columns for method==geos
df_a100_spmm = df_a100_spmm_melted[df_a100_spmm_melted['Method'] == 'geos'][['GPU', 'normalized_speedup']]
df_h100_spmm = df_h100_spmm_melted[df_h100_spmm_melted['Method'] == 'geos'][['GPU', 'normalized_speedup']]
df_rtx3090Ti_spmm = df_rtx3090Ti_spmm_melted[df_rtx3090Ti_spmm_melted['Method'] == 'geos'][['GPU', 'normalized_speedup']]

# Combine the dataframes
df_combined_spmm = pd.concat([df_a100_spmm, df_h100_spmm, df_rtx3090Ti_spmm], ignore_index=True)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
# Example plotting code for the first subplot
sns.stripplot(x='GPU', y='normalized_speedup', data=df_combined, ax=axs[0], hue='GPU', palette='mako')
axs[0].set_title('Segment Reduce', fontsize=16, fontweight='bold', fontname='Arial')
axs[0].set_ylabel('Normalized Speedup', fontsize=14,  fontname='Arial')
axs[0].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)

# Example plotting code for the second subplot
sns.stripplot(x='GPU', y='normalized_speedup', data=df_combined_spmm, ax=axs[1], hue='GPU', palette='mako')
axs[1].set_title('SpMM', fontsize=16, fontweight='bold', fontname='Arial')
axs[1].set_ylabel('')
axs[1].tick_params(axis='x', labelsize=12)

plt.tight_layout()

# set x-axis label with none
axs[0].set_xlabel('')
axs[1].set_xlabel('')

# plt.subplots_adjust(top=0.9)

plt.ylim(bottom=0)  # Set the bottom limit of the y-axis to 0
# Save the plot
plt.savefig("portability_speedup_comparison.pdf", dpi=300, bbox_inches='tight')

