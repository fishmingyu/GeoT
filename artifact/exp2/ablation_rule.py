import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read pr_result and sr_result
pr_header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5', 'time', 'gflops']
sr_header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']

pr_result = pd.read_csv('../SC24-Result/Ablation/pr_result.csv', header=None, names=pr_header)
sr_result = pd.read_csv('../SC24-Result/Ablation/sr_result.csv', header=None, names=sr_header)
# find the best tuning result for each dataset and feature size
pr_best = pr_result.groupby(['dataname', 'feature_size'])['time'].min().reset_index()
sr_best = sr_result.groupby(['dataname', 'feature_size'])['time'].min().reset_index()
# process the dataname
pr_best['dataname'] = pr_best['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])
sr_best['dataname'] = sr_best['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])

# compare the pr_best and sr_best, if the pr_best is better, then use pr_best, otherwise use sr_best
# pr_best does not have data when feature_size > 32
# so we need to use sr_best for these cases
# merge the two dataframes
best = pd.merge(pr_best, sr_best, on=['dataname', 'feature_size'], suffixes=('_pr', '_sr'), how='outer')
# keep the minimum time of the two
best['tune_time'] = best[['time_pr', 'time_sr']].min(axis=1)
best = best.drop(columns=['time_pr', 'time_sr'])
# sort the best dataframe by 'dataname' and 'feature_size'
best = best.sort_values(by=['dataname', 'feature_size'])

# read rule_result.csv, but only leave the naive rule
path = '../SC24-Result/Ablation/rule_result.csv'

# read file
header = ['dataname', 'rule', 'feature_size', 'size', 'time', 'gflops']
# no header in the file
df = pd.read_csv(path, header=None, names=header)

# first process the dataname, only keep the part after the last '/', and remove the '_idx.npy' suffix
df['dataname'] = df['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])

# only keep the naive rule
naive_df = df[df['rule'] == 'naive']

# only leave the dataname, feature_size, time columns
naive_df = naive_df[['dataname', 'feature_size', 'time']]
# rename the time column to naive_time
naive_df = naive_df.rename(columns={'time': 'naive_time'})
# sort the dataframe by 'dataname' and 'feature_size'
naive_df = naive_df.sort_values(by=['dataname', 'feature_size'])

# read selectrule_result.csv
selectrule_path = 'selectrule_result.csv'
selectrule_df = pd.read_csv(selectrule_path)

# sort the dataframe by 'dataname' and 'feature_size'
selectrule_df = selectrule_df.sort_values(by=['dataname', 'feature_size'])
# rename the time column to dtree_time
selectrule_df = selectrule_df.rename(columns={'time': 'dtree_time'})

# merge the naive_df and selectrule_df and best
speedup_df = pd.merge(naive_df, selectrule_df, on=['dataname', 'feature_size'])
speedup_df = pd.merge(speedup_df, best, on=['dataname', 'feature_size'])

# calculate the speedup over naive
speedup_df['dtree_speedup'] = speedup_df['naive_time'] / speedup_df['dtree_time']
speedup_df['tune_speedup'] = speedup_df['naive_time'] / speedup_df['tune_time']

# only leave the dataname, feature_size, dtree_speedup, tune_speedup columns
speedup_df = speedup_df[['dataname', 'feature_size', 'dtree_speedup', 'tune_speedup']]

# draw a joint plot for the two speedup
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

# draw the regplot with color #40b7ad scatter_kws={'s': 45, 'alpha': 0.3}
g = sns.JointGrid(data=speedup_df, x='dtree_speedup', y='tune_speedup', space=0)
g = g.plot_joint(sns.regplot, color='#40b7ad', scatter_kws={'s': 45, 'alpha': 0.3})
# g = g.plot_marginals(sns.histplot, kde=True, color='#40b7ad')

# set the labels and titles
g.set_axis_labels('Speedup of DTREE', 'Speedup of Tuning', fontsize=18)
# set xticks and yticks fontsize
g.ax_joint.tick_params(axis='x', labelsize=15)
g.ax_joint.tick_params(axis='y', labelsize=15)

plt.savefig('speedup_jointplot.pdf', dpi=300)