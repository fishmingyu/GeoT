import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load file from path
path = '../benchmark/benchmark_cpp/rule_result.csv'

# read file
header = ['dataname', 'rule', 'feature_size', 'size', 'time', 'gflops']
# no header in the file
df = pd.read_csv(path, header=None, names=header)

# first process the dataname, only keep the part after the last '/', and remove the '_idx.npy' suffix
df['dataname'] = df['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])


# group by 'dataname' and 'feature_size'
grouped = df.groupby(['dataname'])

# now each group has data contains 8 features x 2 rules
# we need to calculate the speedup of dtree rule over naive rule
# so we get 8 speedup data from each group

# now write the code to do this
speedup = []
for name, group in grouped:
    # group by 'feature_size'
    feature_grouped = group.groupby(['feature_size'])
    for name, feature_group in feature_grouped:
        # get the naive and dtree rule
        naive = feature_group[feature_group['rule'] == 'naive']
        dtree = feature_group[feature_group['rule'] == 'dtree']
        # calculate the speedup
        speedup.append([naive['dataname'].values[0], naive['feature_size'].values[0], naive['size'].values[0],
                        naive['time'].values[0], dtree['time'].values[0], naive['gflops'].values[0], dtree['gflops'].values[0],
                        naive['time'].values[0] / dtree['time'].values[0]])

# convert to dataframe
speedup_df = pd.DataFrame(speedup, columns=['dataname', 'feature_size', 'size', 'naive_time', 'dtree_time', 'naive_gflops', 'dtree_gflops', 'speedup'])

# plot the speedup with lineplot, each line represents a dataname, x-axis is feature_size, y-axis is speedup
sns.set_theme(style="whitegrid")

# Draw a line plot to show the speedup
plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=speedup_df, x="feature_size", y="speedup", hue="dataname", marker='o')
ax.set_title('Speedup of dtree over naive rule')
plt.savefig('speedup.png')