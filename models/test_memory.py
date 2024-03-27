import os 
import sys

def run_memory():
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv', 'reddit2']
    features = ['32', '64']
    # three models
    models = ['GCN', 'GIN', 'GraphSAGE']

    for dataset in datasets:
        for feature in features:
            for model in models:
                py_file = f"run_memory.py"
                print(f"Running {py_file} with dataset {dataset} and hidden_channels {feature} and model {model}")
                # run original model
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {feature} --model {model}")
                # run sparse
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {feature} --model {model} --sparse")
                # run GS
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {feature} --model {model} --sparse --GS")

if __name__ == "__main__":
    run_memory()
    print("All done!")