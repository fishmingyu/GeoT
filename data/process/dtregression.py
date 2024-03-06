# import cudf.pandas
# cudf.pandas.install()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import argparse
from sklearn.tree import _tree

# Read dataset from sr_result_groundtrue.csv
# The dataset contains the following columns:
# dataname, feature_size, size, max, std, mean, config1, config2, config3, config4, time, gflops
# The dataset is used to train the decision tree regressor
# The decision tree regressor is used to predict the (config1, config2, config3, config4) given the feature_size, size, max, (std, mean)

# Step 1: Load the dataset
# Step 2: Split the dataset into features and target
# Step 3: Split the dataset into training and testing sets
# Step 4: Train the decision tree regressor


def round_to_nearest_power_of_two(x):
    return 2 ** np.round(np.log2(x)).astype(int)

def predict_sr(dt, X_test):
    y_pred = dt.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    y_pred = pd.DataFrame(y_pred, columns=['config1', 'config2', 'config3', 'config4'])
    y_pred = round_to_nearest_power_of_two(y_pred)
    return y_pred


def tree_to_code_sr(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    # Initialize cpp_code with the provided function header
    cpp_code = ""

    def recurse(node, depth):
        nonlocal cpp_code
        indent = "    " * (depth + 2)  # Adjusting indentation to match the header
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node].replace(" ", "_")
            threshold = tree_.threshold[node]
            cpp_code += f"{indent}if ({name} <= {np.round(threshold, 2)}) {{\n"
            recurse(tree_.children_left[node], depth + 1)
            cpp_code += f"{indent}}} else {{\n"
            recurse(tree_.children_right[node], depth + 1)
            cpp_code += f"{indent}}}\n"
        else:
            # we have four values in the leaf node
            # example, segreduce_sr_sorted<scalar_t, 2, 16, 32, 2>(index, src, dst)
            # we need first convert the leaf node use round and mapping
            value = tree_.value[node] # shape (4, 1)
            value = value.flatten()
            config1 = round_to_nearest_power_of_two(value[0])
            config2 = round_to_nearest_power_of_two(value[1])
            config3 = round_to_nearest_power_of_two(value[2])
            config4 = round_to_nearest_power_of_two(value[3])
            cpp_code += f"{indent}segreduce_sr_sorted<scalar_t, {config1}, {config2}, {config3}, {config4}>(index, src, dst);\n"

    recurse(0, 0)

    return cpp_code

def process_sr(args):
    # Step 1: Load the dataset (has header)
    df = pd.read_csv('sr_result_groundtrue.csv')
    # prune the dataset when feature < 8
    df = df[df['feature_size'] >= 8]
    # reindex the dataset
    df = df.reset_index(drop=True)

    # Step 2: Split the dataset into features and target
    # add new avg = size / max
    df['avg'] = df['size'] / df['max']
    features = df[['feature_size', 'size', 'avg']]
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
    dt = DecisionTreeRegressor(max_depth=5, criterion='friedman_mse', splitter='best')
    dt.fit(X_train, y_train)

    # Step 5: Predict the (config1, config2, config3, config4) given the feature_size, size, max, std, mean
    y_pred = predict_sr(dt, X_test)

    # export the decision tree regressor to graphviz
    code = tree_to_code_sr(dt, features.columns)
        
    # now merge the y_pred with X_test_data
    eval_df = pd.concat([X_test_data, y_pred], axis=1)
    # merge eval_df and y_test_gflops
    eval_df = pd.concat([eval_df, y_test_gflops], axis=1)
    # rename the gflops column to be 'ground_truth'
    eval_df = eval_df.rename(columns={'gflops': 'ground_truth'})

    # # Step 6: Evaluate the model
    if not args.eval:
        print("Skip evaluation")
        return code
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
        best_gflops = row['ground_truth']
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
            print(f"Find the gflops of {dataname} {feature_size} {config1} {config2} {config3} {config4}, gflops: {gflops[0]}, best gflops: {best_gflops}")
            eval_gflops.append(gflops[0])

    eval_df['predicted'] = eval_gflops

    # # thirdly, compare the ground truth gflops and the predicted gflops
    rmse = root_mean_squared_error(eval_gflops, y_test_gflops)
    print(f"Root Mean Squared Error: {rmse}")

    # # Step 7: Save the result to a new csv file
    eval_df.to_csv('sr_result_eval.csv', index=False)
    return code


def predict_pr(dt, X_test):
    y_pred = dt.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    y_pred = pd.DataFrame(y_pred, columns=['config1', 'config2', 'config3', 'config4', 'config5'])
    y_pred = round_to_nearest_power_of_two(y_pred)
    return y_pred

def tree_to_code_pr(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    # Initialize cpp_code with the provided function header
    cpp_code = ""

    def recurse(node, depth):
        nonlocal cpp_code
        indent = "    " * (depth + 2)  # Adjusting indentation to match the header
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node].replace(" ", "_")
            threshold = tree_.threshold[node]
            cpp_code += f"{indent}if ({name} <= {np.round(threshold, 2)}) {{\n"
            recurse(tree_.children_left[node], depth + 1)
            cpp_code += f"{indent}}} else {{\n"
            recurse(tree_.children_right[node], depth + 1)
            cpp_code += f"{indent}}}\n"
        else:
            # we have five values in the leaf node
            # example, segreduce_pr_sorted<scalar_t, 2, 2, 2, 2, 16>(index, src, dst)
            # we need first convert the leaf node use round and mapping
            value = tree_.value[node] # shape (5, 1)
            value = value.flatten()
            config1 = round_to_nearest_power_of_two(value[0])
            config2 = round_to_nearest_power_of_two(value[1])
            config3 = round_to_nearest_power_of_two(value[2])
            config4 = round_to_nearest_power_of_two(value[3])
            config5 = round_to_nearest_power_of_two(value[4])
            cpp_code += f"{indent}segreduce_pr_sorted<scalar_t, {config1}, {config2}, {config3}, {config4}, {config5}>(index, src, dst);\n"

    recurse(0, 0)

    return cpp_code

def process_pr(args):
    # Step 1: Load the dataset (has header)
    df = pd.read_csv('pr_result_groundtrue.csv')
    # prune the dataset when feature >= 8
    df = df[df['feature_size'] < 8]
    # reindex the dataset
    df = df.reset_index(drop=True)

    # Step 2: Split the dataset into features and target
    # add new avg = size / max
    df['avg'] = df['size'] / df['max']
    features = df[['feature_size', 'size', 'avg']]
    # convert config1, config2, config3, config4 using the map
    target = df[['config1', 'config2', 'config3', 'config4', 'config5']]

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
    dt = DecisionTreeRegressor(max_depth=5, criterion='friedman_mse', splitter='best')
    dt.fit(X_train, y_train)

    # Step 5: Predict the (config1, config2, config3, config4, config5) given the feature_size, size, max
    y_pred = predict_pr(dt, X_test)

    # export the decision tree regressor to graphviz
    code = tree_to_code_pr(dt, features.columns)
        
    # now merge the y_pred with X_test_data
    eval_df = pd.concat([X_test_data, y_pred], axis=1)
    # merge eval_df and y_test_gflops
    eval_df = pd.concat([eval_df, y_test_gflops], axis=1)
    # rename the gflops column to be
    eval_df = eval_df.rename(columns={'gflops': 'ground_truth'})

    # # Step 6: Evaluate the model
    if not args.eval:
        print("Skip evaluation")
        return code

    # firstly, read pr_result.csv
    header = ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5', 'time', 'gflops']
    predicted = pd.read_csv('../pr_result.csv', header=None, names=header)

    # secondly, search the gflops of given ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5'] in every row of eval_df
    eval_gflops = []

    for i in range(len(eval_df)):
        row = eval_df.iloc[i]
        dataname = row['dataname']
        feature_size = row['feature_size']
        config1 = row['config1']
        config2 = row['config2']
        config3 = row['config3']
        config4 = row['config4']
        config5 = row['config5']
        best_gflops = row['ground_truth']
        # search the gflops of given ['dataname', 'feature_size', 'config1', 'config2', 'config3', 'config4', 'config5'] in every row of predicted
        # first get the dataframe of the given dataname
        # we need to split the predicted dataname to get the last part
        predicted['dataname'] = predicted['dataname'].apply(lambda x: x.split("/")[-1])
        config5_df = predicted[(predicted['dataname'] == dataname) & (predicted['feature_size'] == feature_size) & (predicted['config1'] == config1) & (predicted['config2'] == config2) & (predicted['config3'] == config3) & (predicted['config4'] == config4) & (predicted['config5'] == config5)]
        # then get the gflops of the given config5
        gflops = config5_df['gflops'].values
        if len(gflops) == 0:
            print(f"Cannot find the gflops of {dataname} {feature_size} {config1} {config2} {config3} {config4} {config5}")
            eval_gflops.append(0)
        else:
            print(f"Find the gflops of {dataname} {feature_size} {config1} {config2} {config3} {config4} {config5}, gflops: {gflops[0]}, best gflops: {best_gflops}")
            eval_gflops.append(gflops[0])
        
    eval_df['predicted'] = eval_gflops

    # # thirdly, compare the ground truth gflops and the predicted gflops
    rmse = root_mean_squared_error(eval_gflops, y_test_gflops)
    print(f"Root Mean Squared Error: {rmse}")

    # # Step 7: Save the result to a new csv file
    eval_df.to_csv('pr_result_eval.csv', index=False)

    return code


def code_gen(code_sr, code_pr):
    # Using the provided function header
    function_head = r"""
#pragma once
#include "index_scatter_base.h"
template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
"""
    function_body = r"""    const auto size = index.numel();
    const auto feature_size = src.numel() / size;
    const auto keys = dst.numel() / feature_size;
    int avg = size / keys;
"""
    # if feature_size < 8, gen pr code; else gen sr code
    function_body += "    if (feature_size < 8) {\n"
    function_body += code_pr
    function_body += "    } else {\n"
    function_body += code_sr
    function_body += "    }\n"
    function_tail = r"}"
    return function_head + function_body + function_tail


if __name__ == "__main__":
    # add arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=bool, default=True, help="export file")
    parser.add_argument("--eval", type=bool, default=False, help="eval file")
    parser.add_argument("--sr", type=bool, default=True, help="sr file")
    parser.add_argument("--pr", type=bool, default=True, help="pr file")
    args = parser.parse_args()
    if args.sr:
        code_sr = process_sr(args)
    if args.pr:
        code_pr = process_pr(args)
    
    if args.export:
        code = code_gen(code_sr, code_pr)
        with open('index_scatter_rule.h', 'w') as f:
            f.write(code)
        print("Export index_scatter_rule.h successfully")