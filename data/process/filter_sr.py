import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
column_names = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
df = pd.read_csv('../sr_result.csv', header=None, names=column_names)

# group by dataname and feature_size
grouped = df.groupby(['dataname', 'feature_size'])

# get the top 1 tuple
idx = grouped.apply(lambda x: x['gflops'].idxmax())
df = df.loc[idx]

# rename the first column(file), split the file name and only keep the last part
df['dataname'] = df['dataname'].apply(lambda x: x.split("/")[-1])

# add feature to the dataframe
# read the feature.csv file
feature = pd.read_csv('feature.csv', header=None, names=['dataname', 'size', 'max', 'std', 'mean'])

# merge the feature after the 'feature_size' column
df = pd.merge(df, feature, on='dataname')
# reorder the columns, ['dataname', 'feature_size', 'size', 'max', 'std', 'mean', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
df = df[['dataname', 'feature_size', 'size', 'max', 'std', 'mean', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']]


# save the result to a new csv file
df.to_csv('sr_result_groundtrue.csv', index=False)