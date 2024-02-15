import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
column_names = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
df = pd.read_csv('../sr_result.csv', header=None, names=column_names)

# prune the inf and -inf
df = df.replace([float('inf'), float('-inf')], float('0'))
# drop 0
df = df.drop(df[df['gflops'] == 0].index)

# group by dataname and feature_size
grouped = df.groupby(['dataname', 'feature_size'])

# get the top 1 tuple (config2, config4) for each group
idx = grouped.apply(lambda x: x['gflops'].idxmax())
df = df.loc[idx]

# plot the percentage of joint (config2, config4) tuple using pie chart
# draw config2 and config4 as a tuple
config2 = df['config2'].values
config4 = df['config4'].values
config2_4 = list(zip(config2, config4))
config2_4 = pd.Series(config2_4)
config2_4 = config2_4.value_counts()
config2_4.plot.pie(autopct='%.2f')
plt.title('Percentage of joint (config2, config4) tuple')
plt.savefig('sr_config2_4_pie.png')