## Run cpp benchmark

The input dataset format is `.npy`.

### Run benchmark_kernel
```bash
mkdir build
cd build
cmake ..
make
```

Then run the test with
``` bash
python run_benchmark.py
```

### Run ablation
