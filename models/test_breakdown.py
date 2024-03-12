import os 
import sys

def run_breakdown():
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv', 'reddit2']

    for dataset in datasets:
            py_file = f"breakdown.py"
            print(f"Running {py_file} with dataset {dataset}")
            # run original model
            os.system(f"python3 {py_file} --dataset {dataset} ")
            # run sparse
            os.system(f"python3 {py_file} --dataset {dataset} --sparse")
            # run GS
            os.system(f"python3 {py_file} --dataset {dataset} --sparse --GS")

if __name__ == "__main__":
    run_breakdown()
    print("All done!")