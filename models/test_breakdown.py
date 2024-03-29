import os 
import sys

def run_breakdown():
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv', 'reddit2']
    features = ['32', '64']
    for dataset in datasets:
        for feature in features:
            py_file = f"breakdown.py"
            print(f"Running {py_file} with dataset {dataset} and hidden_channels {feature}")
            # run sparse
            os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {feature} --sparse")
            # run GS
            os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {feature} --sparse --GS")

if __name__ == "__main__":
    run_breakdown()
    print("All done!")