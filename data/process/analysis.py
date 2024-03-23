import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib.ticker import SymmetricalLogLocator

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
# Plot SR data using lineplot with markers and color #B0F2BC
plt.figure(figsize=(6, 6))

# Plot SR data using lineplot with markers and color #B0F2BC
sns.lineplot(x='feature_size', y='gflops', data=df_sr, marker='o', markersize=10, label='SR', color='#B0F2BC')

# Plot PR data using lineplot with markers and color #38B2A3
sns.lineplot(x='feature_size', y='gflops', data=df_pr, marker='^', markersize=10, label='PR', color='#38B2A3')

# Annotate each point for SR data
for index, row in df_sr.iterrows():
    plt.text(row['feature_size'], row['gflops'], f"{row['gflops']:.2f}", color='#007200', fontsize=9, ha='center')

# Annotate each point for PR data
for index, row in df_pr.iterrows():
    plt.text(row['feature_size'], row['gflops'], f"{row['gflops']:.2f}", color='#003366', fontsize=9, ha='center')



# Add title and labels with larger font size
plt.title('Feature Size vs GFLOPS', fontsize=18, fontname='Arial',fontweight='bold')
plt.xlabel('Feature Size', fontsize=14, fontname='Arial',fontweight='bold')
plt.ylabel('GFLOPS', fontsize=14, fontname='Arial',fontweight='bold')

# Set legend font size
plt.legend(fontsize=12)
# Adjust x-axis ticks and labels

plt.xscale('symlog', base=2)
plt.xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'], fontsize=12)
plt.yticks(fontsize=12)
# Save the plot as a high-resolution PDF
plt.savefig('avg_gflops_feature_barplot.pdf', dpi=300)
