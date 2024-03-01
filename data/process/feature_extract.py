
import os
import numpy as np

def feature_extract(data):
    size = len(data)
    max_value = np.max(data)
    std = np.std(data)
    mean = np.mean(data)
    # return the features as a str
    feat = f"{size},{max_value},{std:.2f},{mean:.2f}"
    return feat

# iterate through all files in the directory
# write the results to a csv file
def process():
    path = "../idx_data/"
    files = os.listdir(path)
    with open("feature.csv", "w") as f:
        for file in files:
            data = np.load(path+file)
            feat = feature_extract(data)
            # write the file name and the features to the csv file
            # split the file name, only keep the last part
            file = file.split("/")[-1]
            f.write(file + ",")
            f.write(feat + "\n")

if __name__ == "__main__":
    process()