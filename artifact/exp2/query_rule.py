import pandas as pd
import matplotlib.pyplot as plt
from select_rule import select

# we need to process the dtree rule result
# 1. first read feature.csv
feature_path = 'feature.csv'
# this will tell us the size, max and avg of given dataset
feature_df = pd.read_csv(feature_path)
# process the dataframe using split
feature_df['file'] = feature_df['file'].apply(lambda x: x.split('_idx.npy')[0])

# 2. iterate through all the pairs of dataset and feature size, and query the rule using select function
# if feature_size >= 4, then it will return sr rule(which is 4 configs), otherwise it will return pr rule(which is 5 configs)

datasets = feature_df['file']

feature_list = [1, 2, 4, 8, 16, 32, 64, 128]

# open ground truth pr and sr result
pr_header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5', 'time', 'gflops']
pr_result = pd.read_csv('../SC24-Result/Ablation/pr_result.csv', header=None, names=pr_header)
sr_header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
sr_result = pd.read_csv('../SC24-Result/Ablation/sr_result.csv', header=None, names=sr_header)
# first process the dataname
pr_result['dataname'] = pr_result['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])
sr_result['dataname'] = sr_result['dataname'].apply(lambda x: x.split('/')[-1].split('_idx.npy')[0])

# store the result in a list
result_list = []
for dataset in datasets:
    for feature_size in feature_list:
        size = feature_df[feature_df['file'] == dataset]['size'].values[0]
        avg = feature_df[feature_df['file'] == dataset]['avg'].values[0]
        rule = select(size, feature_size, avg)
        # if return 5 configs, then it is pr rule
        if len(rule) == 5:
            # find the dtree result for this dataset and feature size
            dtree = pr_result[(pr_result['dataname'] == dataset) & (pr_result['feature_size'] == feature_size) & (pr_result['config1'] == rule[0]) 
                              & (pr_result['config2'] == rule[1]) & (pr_result['config3'] == rule[2]) & (pr_result['config4'] == rule[3]) & (pr_result['config5'] == rule[4])]['time'].values
        else:
            # find the dtree result for this dataset and feature size
            dtree = sr_result[(sr_result['dataname'] == dataset) & (sr_result['feature_size'] == feature_size) & (sr_result['config1'] == rule[0]) 
                              & (sr_result['config2'] == rule[1]) & (sr_result['config3'] == rule[2]) & (sr_result['config4'] == rule[3])]['time'].values
        result_list.append([dataset, feature_size, dtree[0]])
    
# create a dataframe
result_df = pd.DataFrame(result_list, columns=['dataname', 'feature_size', 'time'])
# write to csv
result_df.to_csv('selectrule_result.csv', index=False)
