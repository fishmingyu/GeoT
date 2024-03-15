import os
import numpy as np

def feature_extract(data):
    size = len(data)
    max_value = np.max(data)
    avg = size / max_value
    # return the features as a str
    feat = f"{size},{max_value},{avg}"
    return feat

# iterate through all files in the directory
# write the results to a csv file
def process():
    path = "../../data/eval_data/"
    files = os.listdir(path)
    with open("feature.csv", "w") as f:
        # add header
        f.write("file,size,max,avg\n")
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