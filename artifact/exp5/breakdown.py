import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# add up "aten::sort" and "torch_sparse::ind2ptr" using the tag "Format"
# check "aten::sort" and "torch_sparse::ind2ptr" are in the value of the function column
# then get the percentage
new_percent['Format'] = df_filckr_pyg['Percentage'][df_filckr_pyg['Function'] == 'aten::sort'].values[0] + df_filckr_pyg['Percentage'][df_filckr_pyg['Function'] == 'torch_sparse::ind2ptr'].values[0]

# use 100 - (SpMM + MatMul + Format) to calculate the "Others" percentage
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'] + new_percent['Format'])

# Create a new dataframe with the new percentages
df_filckr_pyg = pd.DataFrame(new_percent, index=['filckr_pyg'])

# now we process the GS data
new_percent = {}

# for GS, calculate the "torch_index_scatter::gather_scatter" percentage using the tag "SpMM"
new_percent['SpMM'] = df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]

# add up "aten::addmm" and "aten::mm" using the tag "MatMul"
new_percent['MatMul'] = df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'aten::addmm'].values[0] + df_filckr_GS['Percentage'][df_filckr_GS['Function'] == 'aten::mm'].values[0]

# No format percentage in GS
new_percent['Format'] = 0

# use 100 - (SpMM + MatMul + Format) to calculate the "Others" percentage
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'] + new_percent['Format'])

# Create a new dataframe with the new percentages
df_filckr_GS = pd.DataFrame(new_percent, index=['filckr_GS'])

# Set global font to Arial (ensure Arial is available on your system)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set a base font size

df_filckr_pyg['Tech'] = 'PyG'
df_filckr_GS['Tech'] = 'GS'

# Concatenate the two DataFrames vertically
df_combined_filckr = pd.concat([df_filckr_pyg, df_filckr_GS], ignore_index=True)

print(df_combined_filckr)

# now do the same process for dataset reddit2

df_reddit2_pyg = pd.read_csv("../SC24-Result/breakdown/reddit2_sparse_breakdown.csv")
df_reddit2_GS = pd.read_csv("../SC24-Result/breakdown/reddit2_sparse_GS_breakdown.csv")

new_percent = {}

new_percent['SpMM'] = df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'torch_sparse::spmm_sum'].values[0]
new_percent['MatMul'] = df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'aten::addmm'].values[0] + df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'aten::mm'].values[0]
new_percent['Format'] = df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'aten::sort'].values[0] + df_reddit2_pyg['Percentage'][df_reddit2_pyg['Function'] == 'torch_sparse::ind2ptr'].values[0]
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'] + new_percent['Format'])

df_reddit2_pyg = pd.DataFrame(new_percent, index=['reddit2_pyg'])

new_percent = {}

new_percent['SpMM'] = df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'torch_index_scatter::gather_scatter'].values[0]
new_percent['MatMul'] = df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'aten::addmm'].values[0] + df_reddit2_GS['Percentage'][df_reddit2_GS['Function'] == 'aten::mm'].values[0]
new_percent['Format'] = 0
new_percent['Others'] = 100 - (new_percent['SpMM'] + new_percent['MatMul'] + new_percent['Format'])

df_reddit2_GS = pd.DataFrame(new_percent, index=['reddit2_GS'])

df_reddit2_pyg['Tech'] = 'PyG'
df_reddit2_GS['Tech'] = 'GS'

df_combined_reddit2 = pd.concat([df_reddit2_pyg, df_reddit2_GS], ignore_index=True)
print(df_combined_reddit2)