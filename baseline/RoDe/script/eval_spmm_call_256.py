import torch
import pandas as pd
import csv
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
baseline_dir = project_dir + '/Baseline/'

df = pd.read_csv(baseline_dir + '/dataset/data_filter.csv')
# df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_128.csv')

#用于存储结果
file_name = project_dir + '/result/Baseline/spmm/rode_spmm_f32_n256.csv'
head = ['dataSet','rows_','columns_','nonzeros_','cuSPARSE','cuSPARSE_gflops','rode','rode_gflops']

with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)
count = 0

start_time = time.time()
for index, row in df.iterrows():
    count+=1
    
    data = [row['Dataset']]
    if data in ['adaptive', 'delaunay_n22', 'rgg_n_2_22_s0'] :
        continue
    with open(file_name, 'a', newline='') as csvfile:
        csvfile.write(','.join(map(str, data)))

    shell_command =project_dir + "/Baseline/RoDe/build/eval/eval_spmm_f32_n256 " + baseline_dir + "dataset/" + row['Dataset'] + '/' + row['Dataset'] + ".mtx >> " + file_name
    
    print(row['Dataset'])
    subprocess.run(shell_command, shell=True)

    
    
end_time = time.time()
execution_time = end_time - start_time

dimN = 256
# Record execution time.
with open("execution_time_base.txt", "a") as file:
    file.write("spmm-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")
