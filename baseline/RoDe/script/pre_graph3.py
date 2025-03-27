import torch
import numpy as np
import os
import pandas as pd
import time

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

output_sh_path = project_dir + '/Baseline/RoDe/script/download.sh'
# df = pd.read_csv(project_dir + '/dataset/data_filter2.csv')
# base_url = "https://www.cise.ufl.edu/research/sparse/MM"
# with open(output_sh_path, 'w') as sh_file:
#     for index, row in df.iterrows():
#         data = row['dataSet']
#         wget_command = f"wget {base_url}{'/'}{data}{'/'}{data}.tar.gz\n"
#         sh_file.write(wget_command)
        
        
df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
base_url = "https://dataset-ppopp.oss-cn-beijing.aliyuncs.com/dataset1"
with open(output_sh_path, 'a') as sh_file:
    for index, row in df.iterrows():
        data = row['dataSet']
        wget_command = f"wget {base_url}{'/'}{data}.tar.gz\n"
        sh_file.write(wget_command)
        