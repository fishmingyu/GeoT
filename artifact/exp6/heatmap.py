import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('../SC24-Results/Ablation/sr_result.csv')

# split the dataset column
"""
/mnt/disk3/xinchen/projects/torch_index_scatter/benchmark/benchmark_cpp/../../data/eval_data/amazon_photo_idx.npy
--> amazon_photo
"""
df['dataset'] = df['dataset'].str.split('/').str[-1].str.split('_').str[0]

# extract amazon_photo dataset
df = df[df['dataset'] == 'amazon_photo']

print(df)