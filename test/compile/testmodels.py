import os 
which = 'sparse'
def run_model():
    with open(f"model_result.csv", "w") as f:
        f.write("model,dataset,hidden_channels,type,time\n")
    # three dataset
    datasets = ['flickr', 'ogbn-arxiv']
    # 7 models, 'gat' and 'arma' needs 3-dim gather_weight_scatter
    # models = ['appnp', 'arma', 'gat', 'gcn', 'gin', 'graphsage', 'sgc']
    models = ['appnp', 'gat', 'gcn', 'gin', 'graphsage', 'sgc']

    # two hidden_channels
    hidden_channels = [32, 64]

    for dataset in datasets:
        for model in models:
            for hidden_channel in hidden_channels:
                py_file = f"test_{model}.py"
                print(f"Running {py_file} with dataset {dataset} and hidden_channels {hidden_channel} and type {which}")
                # run original model
                os.system(f"python3 {py_file} --dataset {dataset} --hidden_channels {hidden_channel} --which {which}")

if __name__ == "__main__":
    run_model()
    print("All done!")
