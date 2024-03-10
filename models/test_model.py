import os 
import sys

def run_model():
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv', 'reddit2']
    # three models
    models = ['gcn', 'gin', 'graphsage']
    # two hidden_channels
    hidden_channels = [32, 64]

    for dataset in datasets:
        for model in models:
            for hidden_channel in hidden_channels:
                py_file = f"{model}.py"
                print(f"Running {py_file} with dataset {dataset} and hidden_channels {hidden_channel}")
                # run original model
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {hidden_channel}")
                # run sparse
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {hidden_channel} --sparse")
                # run GS
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {hidden_channel} --sparse --GS")

if __name__ == "__main__":
    run_model()
    print("All done!")