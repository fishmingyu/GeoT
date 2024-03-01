import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager


# Add title and labels
# config font
fontpaths = "/usr/share/fonts/truetype/msttcorefonts/"
font_manager.findSystemFonts(fontpaths=fontpaths, fontext='ttf')
font_manager.findfont("Arial")

# Read sr_result_groundtrue.csv
df_sr = pd.read_csv('sr_result_groundtrue.csv')

# Read pr_result_groundtrue.csv
df_pr = pd.read_csv('pr_result_groundtrue.csv')

# Calculate the average gflops for each feature size
df_sr = df_sr.groupby('feature_size').agg({'gflops': np.mean}).reset_index()
df_pr = df_pr.groupby('feature_size').agg({'gflops': np.mean}).reset_index()

# Prune the data when feature <= 32
df_sr = df_sr[df_sr['feature_size'] <= 32]
df_pr = df_pr[df_pr['feature_size'] <= 32]

# # Merge dataframes for easier plotting
# df = pd.merge(df_sr, df_pr, on='feature_size', suffixes=('_SR', '_PR'))

# # Set up the bar plot
sns.set_style("whitegrid")
plt.figure(figsize=(15, 6))

# Add a column to differentiate between SR and PR
df_sr['Type'] = 'SR'
df_pr['Type'] = 'PR'

# Concatenate SR and PR dataframes
df_combined = pd.concat([df_sr, df_pr])

# Set up the figure
plt.figure(figsize=(10, 6))

# Plot bars using sns.barplot
sns.barplot(x='feature_size', y='gflops', hue='Type', data=df_combined, palette={'SR': 'skyblue', 'PR': 'salmon'})


plt.title('Average GFLOPS on different Feature Size', fontname='Arial')
plt.xlabel('Feature Size', fontname='Arial')
plt.ylabel('GFLOPS', fontname='Arial')

# Adjust x-axis ticks and labels
# plt.xticks([r + bar_width/2 + offset/2 for r in range(len(df))], df['feature_size'])

# Show legend
plt.legend()

# Save the plot as a high-resolution PDF
plt.savefig('avg_gflops_feature_barplot.pdf', dpi=300)
