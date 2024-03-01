# import cudf.pandas
# cudf.pandas.install()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse

# Read dataset from sr_result_groundtrue.csv
# The dataset contains the following columns:
# dataname, feature_size, size, max, std, mean, config1, config2, config3, config4, time, gflops
# The dataset is used to train the decision tree regressor
# The decision tree regressor is used to predict the (config1, config2, config3, config4) given the feature_size, size, max, (std, mean)

# Step 1: Load the dataset
# Step 2: Split the dataset into features and target
# Step 3: Split the dataset into training and testing sets
# Step 4: Train the decision tree regressor

sr_config1_map = {1:0, 2:1}
sr_config2_map = {8:0, 16:1, 32:2, 64:3}
sr_config3_map = {4:0, 8:1, 16:2, 32:3}
sr_config4_map = {2:0, 4:1, 8:2}

def sr_map_df(df):
    df['config1'] = df['config1'].map(sr_config1_map)
    df['config2'] = df['config2'].map(sr_config2_map)
    df['config3'] = df['config3'].map(sr_config3_map)
    df['config4'] = df['config4'].map(sr_config4_map)
    return df

def sr_inverse_map_df(df):
    df['config1'] = df['config1'].map({0:1, 1:2})
    df['config2'] = df['config2'].map({0:8, 1:16, 2:32, 3:64})
    df['config3'] = df['config3'].map({0:4, 1:8, 2:16, 3:32})
    df['config4'] = df['config4'].map({0:2, 1:4, 2:8})
    return df

def predict(dt, X_test):
    y_pred = dt.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    y_pred = pd.DataFrame(y_pred, columns=['config1', 'config2', 'config3', 'config4'])
    y_pred = sr_inverse_map_df(y_pred)
    return y_pred


if __name__ == "__main__":
    # add arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sr_result_groundtrue.csv", help="input file")
    parser.add_argument("--export", type=bool, default=True, help="export file")
    parser.add_argument("--eval", type=bool, default=True, help="eval file")
    args = parser.parse_args()

    # Step 1: Load the dataset (has header)
    df = pd.read_csv('sr_result_groundtrue.csv')
    df = sr_map_df(df)
    # prune the data when feature <= 

    # Step 2: Split the dataset into features and target
    # add new avg = size / max
    df['avg'] = df['size'] / df['max']
    features = df[['feature_size', 'avg']]
    # convert config1, config2, config3, config4 using the map

    target = df[['config1', 'config2', 'config3', 'config4']]

    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    # get the ground truth gflops
    y_test_gflops = df['gflops'].iloc[y_test.index]
    # reindex y_test_gflops
    y_test_gflops = y_test_gflops.reset_index(drop=True)
    # get the dataname of X_test
    X_test_data = df.iloc[X_test.index]
    X_test_data = X_test_data[['dataname', 'feature_size']]
    # reindex X_test_data
    X_test_data = X_test_data.reset_index(drop=True)

    # Step 4: Train the decision tree regressor
    dt = DecisionTreeRegressor(max_depth=5)
    dt.fit(X_train, y_train)

    # Step 5: Predict the (config1, config2, config3, config4) given the feature_size, size, max, std, mean
    y_pred = predict(dt, X_test)

    # export the decision tree regressor to graphviz
    if args.export:
        from sklearn.tree import export_graphviz
        import graphviz
        export_graphviz(dt, out_file="sr.dot", feature_names=features.columns, filled=True, rounded=True)
        with open("sr.dot") as f:
            dot_graph = f.read()
        # save to a file
        graph = graphviz.Source(dot_graph)
        graph.render("sr")

    # now merge the y_pred with X_test_data
    eval_df = pd.concat([X_test_data, y_pred], axis=1)
    # merge eval_df and y_test_gflops
    eval_df = pd.concat([eval_df, y_test_gflops], axis=1)
    # rename the gflops column to be 'ground_truth'
    eval_df = eval_df.rename(columns={'gflops': 'ground_truth'})

    # # Step 6: Evaluate the model
    if not args.eval:
        print("Skip evaluation")
        exit(0)
    # firstly, read sr_result.csv
    header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'time', 'gflops']
    predicted = pd.read_csv('../sr_result.csv', header=None, names=header)

    # secondly, search the gflops of given ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4'] in every row of eval_df
    eval_gflops = []

    for i in range(len(eval_df)):
        row = eval_df.iloc[i]
        dataname = row['dataname']
        feature_size = row['feature_size']
        config1 = row['config1']
        config2 = row['config2']
        config3 = row['config3']
        config4 = row['config4']
        # search the gflops of given ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4'] in every row of predicted
        # first get the dataframe of the given dataname
        # we need to split the predicted dataname to get the last part
        predicted['dataname'] = predicted['dataname'].apply(lambda x: x.split("/")[-1])
        config4_df = predicted[(predicted['dataname'] == dataname) & (predicted['feature_size'] == feature_size) & (predicted['config1'] == config1) & (predicted['config2'] == config2) & (predicted['config3'] == config3) & (predicted['config4'] == config4)]
        # then get the gflops of the given config4
        gflops = config4_df['gflops'].values
        if len(gflops) == 0:
            print(f"Cannot find the gflops of {dataname} {feature_size} {config1} {config2} {config3} {config4}")
            eval_gflops.append(0)
        else:
            print(f"Find the gflops of {dataname} {feature_size} {config1} {config2} {config3} {config4}, gflops: {gflops[0]}")
            eval_gflops.append(gflops[0])

    eval_df['predicted'] = eval_gflops


    # # thirdly, compare the ground truth gflops and the predicted gflops
    mse = mean_squared_error(eval_gflops, y_test_gflops)
    print(f"Mean Squared Error: {mse}")

    # # Step 7: Save the result to a new csv file
    eval_df.to_csv('sr_result_eval.csv', index=False)