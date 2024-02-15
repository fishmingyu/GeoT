import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
column_names = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5', 'time', 'gflops']
df = pd.read_csv('../pr_result.csv', header=None, names=column_names)

# Group by 'dataname' and 'feature_size', then apply the function to each group, and get the top 5 
# for config1, config2, config3, config4, config5 respectively


# prune the inf and -inf
df = df.replace([float('inf'), float('-inf')], float('0'))
# drop 0 gflops
df = df[df['gflops'] > 0]
# take the top of gflops under each (dataname, feature_size) group
new_df = df.groupby(['dataname', 'feature_size']).apply(lambda x: x.sort_values(by='gflops', ascending=False).head(1)).reset_index(drop=True)
new_df = new_df.drop(['config1', 'config2', 'config3', 'config4', 'config5'], axis=1)

# given the mean gflops of each K value, draw a line chart
new_df = new_df.groupby(['feature_size']).apply(lambda x: x['gflops'].mean()).reset_index()
new_df.columns = ['feature_size', 'mean_gflops']
plt.plot(new_df['feature_size'], new_df['mean_gflops'])

print(new_df)
print(new_df['feature_size'])
