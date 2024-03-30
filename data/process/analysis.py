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
sns.lineplot(x='feature_size', y='gflops', data=df_sr, marker='o', markersize=12, label='SR', color='#B0F2BC')

# Plot PR data using lineplot with markers and color #38B2A3
sns.lineplot(x='feature_size', y='gflops', data=df_pr, marker='^', markersize=12, label='PR', color='#38B2A3')

# Annotate each point for SR data
for index, row in df_sr.iterrows():
    # if feature_size == 4, move the annotation to the lower center
    if row['feature_size'] == 4:
        plt.text(row['feature_size'], row['gflops']-5, f"{row['gflops']:.0f}", color='#007200', fontsize=12, ha='center', fontweight='bold')
    else:
        plt.text(row['feature_size'], row['gflops'], f"{row['gflops']:.0f}", color='#007200', fontsize=12, ha='center', fontweight='bold')

# Annotate each point for PR data
for index, row in df_pr.iterrows():
    plt.text(row['feature_size'], row['gflops'], f"{row['gflops']:.0f}", color='#003366', fontsize=12, ha='center', fontweight='bold')



# Add title and labels with larger font size
plt.title('Feature Size vs GFLOPS', fontsize=20, fontname='Arial',fontweight='bold')
plt.xlabel('Feature Size', fontsize=16, fontname='Arial')
plt.ylabel('GFLOPS', fontsize=16, fontname='Arial')

# Set legend font size
plt.legend(fontsize=14)
# Adjust x-axis ticks and labels

plt.xscale('symlog', base=2)
plt.xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'], fontsize=14)
plt.yticks(fontsize=14)
# Save the plot as a high-resolution PDF
plt.savefig('avg_gflops_feature_barplot.pdf', dpi=300, bbox_inches='tight')
