import os

# Run the benchmark
# data dir, ../../data/idx_data
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/idx_data")
# iterate over all files in the data dir (follow the alphabetical order to run the benchmark)
file_start = 0
for file in sorted(os.listdir(data_dir)):
    # run the benchmark on each file
    # print ith file, count the current number
    print(f"Running benchmark on {file}, {file_start}th file of {len(os.listdir(data_dir))} files.")
    file_start += 1
    for K in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        # print command
        # print(f"./build/benchmark {os.path.join(data_dir, file)} {K}")
        os.system(f"./build/benchmark {os.path.join(data_dir, file)} {K}")