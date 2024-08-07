import os 

def run_model():
    with open("model_result.csv", "w") as f:
        f.write("model, dataset, hidden_channels, original_time, compiled_time\n")
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv', 'reddit2']
    # 7 models, 'gat' and 'arma' needs 3-dim gather_weight_scatter
    # models = ['appnp', 'arma', 'gat', 'gcn', 'gin', 'graphsage', 'sgc']
    models = ['appnp', 'gcn', 'gin', 'graphsage', 'sgc']

    # two hidden_channels
    hidden_channels = [32, 64]

    for dataset in datasets:
        for model in models:
            for hidden_channel in hidden_channels:
                py_file = f"test_{model}.py"
                print(f"Running {py_file} with dataset {dataset} and hidden_channels {hidden_channel}")
                # run original model
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {hidden_channel}")

if __name__ == "__main__":
    run_model()
    print("All done!")