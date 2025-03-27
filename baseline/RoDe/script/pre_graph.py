import torch
import numpy as np
import csv
import os
import pandas as pd
import time
import shutil
current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
start_time = time.time()
for index, row in df.iterrows():
    dataset_name = row['dataSet']
    dataset_path = os.path.join(project_dir, 'Baseline/RoDe/dataset', dataset_name)
    print(dataset_path)
    
    if os.path.isdir(dataset_path):
        tar_path = os.path.join(project_dir, 'Baseline/RoDe/gnn_dataset', f"{dataset_name}.tar.gz")


        shutil.make_archive(
            base_name=tar_path.replace('.tar.gz', ''),
            format='gztar',
            root_dir=os.path.dirname(dataset_path),  # 父目录
            base_dir=os.path.basename(dataset_path)  # 子目录名
        )
        
        print(f"{dataset_name} has been archived to {tar_path}")