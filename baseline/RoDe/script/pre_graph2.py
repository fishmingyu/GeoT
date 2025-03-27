import torch
import numpy as np
import os
import pandas as pd
import time

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
start_time = time.time()

for index, row in df.iterrows():
    data = row['dataSet']
    graph = np.load(project_dir + '/dataset/' + data + '.npz')

    # Build data path
    file_dir = project_dir + "/Baseline/RoDe/dataset/" + data
    file_name = file_dir + '/' + data + '.mtx'
    os.makedirs(file_dir, exist_ok=True)
    
    # Write header information once
    head = f"{graph['num_nodes_src']-0} {graph['num_nodes_dst']-0} {graph['num_edges']-0}\n"
    with open(file_name, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(head)
    
    # Prepare data lines in batch to reduce I/O operations
    src_li = graph['src_li']
    dst_li = graph['dst_li']
    lines = [f"{item1+1} {item2+1} 1\n" for item1, item2 in zip(src_li, dst_li)]
    
    # Write all lines at once
    with open(file_name, 'a') as f:
        f.writelines(lines)
    
    print(f"{data} is success")

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {round(execution_time/60, 2)} minutes")
